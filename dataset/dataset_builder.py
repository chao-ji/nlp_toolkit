"""Defines dataset builders."""
import os
import functools

import numpy as np
import tensorflow as tf

from .parse_fn import parse_fn_sequence_pair
from .parse_fn import parse_fn_single_sequence
from .parse_fn import parse_fn_sequence_classification
from .parse_fn import parse_fn_xlnet_pretrain
from .parse_fn import parse_fn_squad
from .parse_fn import parse_fn_sequence_classification_bert


# Buffer size for reading TFRecord files. Should be generally larger than the 
# actual size (in bytes) of each TFRecord file.
_READ_RECORD_BUFFER = 8 * 1000 * 1000

# The minimum boundary for sequence length bucketing.
_MIN_BOUNDARY = 8

# The rate by which the boundary grows for the next bucket.
_BOUNDARY_SCALE = 1.1

CLS_ID = 3
SEP_ID = 4


def create_and_preprocess(filenames, 
                          parse_fn,
                          shuffle, 
                          num_parallel_calls,
                          filter_fn=None, 
                          buffer_size_per_file=None, 
                          random_seed=None):
  """Initialize tf.data.Dataset instance and apply parsing (deserialization), 
  shuffling and filtering preprocessing transforms.

  Args:
    filenames: a list of strings, names of TFRecord files.
    parse_fn: callable, function that deserializes protobuf message.
    shuffle: bool scalar, if False, the training examples will be generated
      deterministically. 
    num_parallel_calls: int scalar, num of TFRecord files to be processed
      concurrently.
    filter_fn: callable, function that applies filtering.
    buffer_size_per_file: int scalar, buffer size used to shuffle records in a 
      single *.tfrecord file.
    random_seed: int scalar, random seed. 

  Returns:
    dataset: a tf.data.Dataset instance. 
  """
  # shape: ()
  dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(
      len(filenames), seed=random_seed)

  # set `options.experimental_deterministic` to False to shuffle the dataset 
  # shape: ()
  options = tf.data.Options()
  options.experimental_deterministic = False if shuffle else True

  if buffer_size_per_file is not None:
    map_fn = lambda filename: tf.data.TFRecordDataset(
        filename, buffer_size=_READ_RECORD_BUFFER).shuffle(
        buffer_size_per_file)
  else:
    map_fn = lambda filename: tf.data.TFRecordDataset(
        filename, buffer_size=_READ_RECORD_BUFFER)

  dataset = dataset.interleave(
      map_fn, cycle_length=num_parallel_calls,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)
  dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)
  # filter out long sequences
  if filter_fn is not None:
    dataset = dataset.filter(filter_fn)

  return dataset


class BaseSequenceTransductionDatasetBuilder(object):
  """Abstract base class for building sequence transduction dataset.

  A sequence transduction dataset produces int 2-tuples of tensors of shape 
  ([batch_size, src_seq_len], [batch_size, tgt_seq_len]), holding the 
  zero-padded token ids for batched source and target sequences. Depending
  on the batching scheme, the subclass `DynamicBatchDatasetBuilder` and
  `StaticBatchDatasetBuilder` produces tensors with *dynamic* or *static*
  `batch_size`.
  """
  def build_dataset(self, filenames):
    """Builds the sequence transduction dataset.

    Args:
      filenames: a list of strings, names of TFRecord files. 

    Returns:
      dataset: a tf.data.Dataset instance, each item is a tuple of two tensors
        of shape [batch_size, src_seq_len] and [batch_size, tgt_seq_len], 
        holding the token ids in source or target sequences. Each row is 
        zero-padded to the length of the longest sequence in each batch, so
        the last column contains at least one non-zero (padded) token.
    """

    filter_fn = lambda x, y: tf.logical_and(
        tf.size(x) <= self._max_length, tf.size(y) <= self._max_length)

    dataset = create_and_preprocess(
        filenames,
        parse_fn=parse_fn_sequence_pair,
        shuffle=self._shuffle,
        num_parallel_calls=self._num_parallel_calls,
        filter_fn=filter_fn,
        buffer_size_per_file=_READ_RECORD_BUFFER,
        random_seed=self._random_seed)

    dataset = self._batch_examples(dataset)
    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


class DynamicBatchDatasetBuilder(BaseSequenceTransductionDatasetBuilder):
  """Builds a tf.data.Dataset instance that reads, zero-pads and batches source 
  and target token ids from pre-generated TFRecord files.

  Note: The produced dataset adopts the dynamic batching scheme (unlike the more
  common static batching) -- the `batch_size` (num of seqs in a batch) may vary 
  across different batches as long as `batch_size * src_seq_len` or `batch_size
  * tgt_seq_len` <= `max_num_tokens`.
  """
  def __init__(self, 
               max_num_tokens, 
               shuffle, 
               max_length, 
               num_parallel_calls, 
               random_seed=None):
    """Constructor.

    Args:
      max_num_tokens: int scalar, the maximum num of tokens in source or target 
        sequences in each batch. 
      shuffle: bool scalar, if False, the training examples will be generated
        deterministically. 
      max_length: int scalar, source or target seqs longer than this will be
        filtered out.
      num_parallel_calls: int scalar, num of TFRecord files to be processed
        concurrently. 
      random_seed: int scalar, random seed.
    """
    self._max_num_tokens = max_num_tokens
    self._shuffle = shuffle
    self._max_length = max_length
    self._num_parallel_calls = num_parallel_calls
    self._random_seed = random_seed
    
  def _batch_examples(self, dataset):
    """Batches the sequence pairs using dynamic batching scheme.

    Args:
      dataset: a tf.data.Dataset instance, each item is a tuple of two int 
        tensors of shape [src_seq_len] and [tgt_seq_len].

    Returns:
      a tf.data.Dataset instance, each item is a tuple of two int tensors of 
        shape [batch_size, src_seq_len] and [batch_size, tgt_seq_len].
    """
    buckets_min, buckets_max = self._create_bucket_bounds()
    bucket_batch_sizes = tf.constant([self._max_num_tokens // (seq_len - 1)
        for seq_len in buckets_max], dtype='int64')

    # mapper
    def example_to_bucket_id(src_token_ids, tgt_token_ids):
      """Maps source and target sequence to bucket id.

      Args:
        src_token_ids: int tensor of shape [src_seq_len].
        tgt_token_ids: int tensor of shape [tgt_seq_len].

      Returns:
        bucket_id: int scalar tensor.
      """
      seq_len = tf.maximum(tf.size(src_token_ids), tf.size(tgt_token_ids))

      flags = tf.logical_and(
          tf.less_equal(buckets_min, seq_len),
          tf.less(seq_len, buckets_max))

      bucket_id = tf.where(flags)[0, 0]
      return bucket_id

    # reducer
    def batching_fn(bucket_id, grouped_dataset):
      """Maps key and dataset to dataset"""
      bucket_batch_size = window_size_fn(bucket_id)
      return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

    def window_size_fn(bucket_id):
      """Maps key to window size."""
      return bucket_batch_sizes[bucket_id]

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))

  def _create_bucket_bounds(
      self, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
    """Creates bucket bounds for sequence length bucketing.

    The buckets are sets of intergers (seq lengths) within intervals like this:

    [0, b_1), [b_1, b_2), ..., [b_n, max_length + 1)

    where b_1 = min_boundary, and b_i+1 = max(b_i * boundary_scale, b_i + 1)

    Args:
      min_boundary: int scalar, the minimum boundary.
      boundary_scale: float scalar, the rate by which the boundary grows.
        
    Returns:
      buckets_lower_bound: list of ints, lower bounds (inclusive) for sequence 
        lengths.
      buckets_upper_bound: list of ints, upper bounds (exclusive) for sequence 
        lengths.
    """
    bucket_boundaries = []
    x = min_boundary
    while x < self._max_length:
      bucket_boundaries.append(x)
      # produce next boundary
      x = max(x + 1, int(x * boundary_scale))

    buckets_lower_bound = [0] + bucket_boundaries
    buckets_upper_bound = bucket_boundaries + [self._max_length + 1]
    return buckets_lower_bound, buckets_upper_bound


class StaticBatchDatasetBuilder(BaseSequenceTransductionDatasetBuilder):
  """Builds a tf.data.Dataset instance that reads, zero-pads and batches source 
  and target token ids from pre-generated TFRecord files.

  Note: The produced dataset adopts the static batching scheme -- the source
  and target token id matrices have shape [batch_size, seq_len] where 
  `batch_size` is fixed across different minibatches.
  """
  def __init__(self,
               batch_size,
               shuffle,
               max_length,
               num_parallel_calls,
               num_buckets=8,
               bucket_width=10,
               random_seed=None):
    """Constructor.

    Args:
      shuffle: bool scalar, if False, the training examples will be generated
        deterministically. 
      max_length: int scalar, source or target seqs longer than this will be
        filtered out.
      num_parallel_calls: int scalar, num of TFRecord files to be processed
        concurrently. 
      num_buckets: int scalar, num of sequence length buckets.
      bucket_width: int scalar, size of each sequence length bucket.
      random_seed: int scalar, random seed.
    """
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._max_length = max_length
    self._num_parallel_calls = num_parallel_calls
    self._num_buckets = num_buckets
    self._bucket_width = bucket_width
    self._random_seed = random_seed

  def _batch_examples(self, dataset):
    """Batches the sequence pairs using dynamic batching scheme.

    Args:
      dataset: a tf.data.Dataset instance, each item is a tuple of two int 
        tensors of shape [src_seq_len] and [tgt_seq_len].

    Returns:
      a tf.data.Dataset instance, each item is a tuple of two int tensors of 
        shape [batch_size, src_seq_len] and [batch_size, tgt_seq_len].
    """
    # mapper
    def example_to_bucket_id(src_token_ids, tgt_token_ids):
      """Maps source and target sequence to bucket id.

      Args:
        src_token_ids: int tensor of shape [src_seq_len].
        tgt_token_ids: int tensor of shape [tgt_seq_len].

      Returns:
        bucket_id: int scalar tensor.
      """
      seq_len = tf.maximum(tf.size(src_token_ids), tf.size(tgt_token_ids))
      bucket_id = seq_len // self._bucket_width
      return tf.cast(tf.minimum(self._num_buckets - 1, bucket_id), 'int64')

    # reducer
    def batching_fn(bucket_id, grouped_dataset):
      """Maps key and dataset to dataset"""
      return grouped_dataset.padded_batch(
          self._batch_size, ([None], [None]), drop_remainder=True)

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=self._batch_size))


class SequenceClassifierDatasetBuilder(object):
  """Builds a tf.data.Dataset instance that generates batched sequences to token
  IDs (without class label) for pretraining or (with class label) for 
  finetuning the sequence classifier for IMDB movie review dataset.
  """
  def __init__(self,
               batch_size,
               shuffle,
               max_length,
               num_parallel_calls,
               num_buckets=10,
               bucket_width=100,
               random_seed=None):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      shuffle: bool scalar, if False, the training examples will be generated
        deterministically.
      max_length: int scalar, sequences longer than this will be filtered out. 
      num_parallel_calls: int scalar, num of TFRecord files to be processed
        concurrently.
      num_buckets: int scalar, num of sequence length buckets. 
      bucket_width: int scalar, size of each sequence length bucket.
      random_seed: int scalar, random seed.
    """
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._max_length = max_length
    self._num_parallel_calls = num_parallel_calls
    self._num_buckets = num_buckets
    self._bucket_width = bucket_width
    self._random_seed = random_seed

  def build_pretrain_dataset(self, filenames):
    """Builds dataset for pretraining.

    Args:
      filenames: a list of strings, names of TFRecord files.
  
    Returns:
      dataset: a tf.data.Dataset instance, each item is an int tensors
        of shape [batch_size, seq_len] holding the token ids. 
    """
    filter_fn = (None if self._max_length is None
        else lambda x: tf.size(x) <= self._max_length)

    dataset = create_and_preprocess(
        filenames,
        parse_fn=parse_fn_single_sequence,
        shuffle=self._shuffle,
        num_parallel_calls=self._num_parallel_calls,
        filter_fn=filter_fn,
        buffer_size_per_file=750,
        random_seed=self._random_seed)

    def example_to_bucket_id(token_ids):
      seq_len = tf.size(token_ids)
      bucket_id = seq_len // self._bucket_width
      return tf.cast(tf.minimum(self._num_buckets - 1, bucket_id), 'int64')

    def batching_fn(bucket_id, grouped_dataset):
      return grouped_dataset.padded_batch(
          self._batch_size, (None,), drop_remainder=True)

    dataset = dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=self._batch_size))
    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

  def build_finetune_dataset(self, filenames):
    """Builds dataset for finetuning.

    Args:
      filenames: a list of strings, names of TFRecord files.

    Returns:
      dataset: a tf.data.Dataset instance, each item is a tuple of int tensors
        of shape [batch_size, seq_len] holding the token ids, and of shape
        [batch_size] holding sequence class labels.
    """
    filter_fn = None
    dataset = create_and_preprocess(
        filenames,
        parse_fn=parse_fn_sequence_classification,
        shuffle=self._shuffle,
        num_parallel_calls=self._num_parallel_calls,
        filter_fn=filter_fn,
        buffer_size_per_file=250,
        random_seed=self._random_seed)

    def example_to_bucket_id(token_ids, _):
      seq_len = tf.size(token_ids)
      bucket_id = seq_len // self._bucket_width
      return tf.cast(tf.minimum(self._num_buckets - 1, bucket_id), 'int64')

    def batching_fn(bucket_id, grouped_dataset):
      return grouped_dataset.padded_batch(
          self._batch_size, ([None], [1]), drop_remainder=True)

    dataset = dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=self._batch_size))
    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class XLNetPretrainDatasetBuilder(object):
  """Builds a tf.data.Dataset instance that generates inputs and targets for
  pretraining XLNet model using permutation language model objective.
  """
  def __init__(self,
               seq_len,
               reuse_len,
               batch_size,
               num_predict,
               perm_size,
               leak_ratio,
               max_num_tokens=5,
               min_num_tokens=1,
               cls_id=3,
               sep_id=4,
               shuffle_files=False):
    """Constructor.

    Args:
      seq_len: int scalar, length of sequence in a batch.
      reuse_len: int scalar, number of token that can be reused as memory.
      batch_size: int scalar, number of sequences in a batch.
      num_predict: int scalar, number of tokens to predict.
      perm_size: int scalar, window size of the permutation.
      leak_ratio: float scalar, percent of masked tokens that are leaked.
      max_num_tokens: int scalar, maximum number of tokens to sample in a span.
      min_num_tokens: int scalar, minimum number of tokens to sample in a span.
      cls_id: int scalar, the ID of the special token CLS.
      sep_id: int scalar, the ID of the special token SEP.
      shuffle_files: bool scalar, whether to shuffle the input TFRecord files,
        defaults to False.
    """
    self._seq_len = seq_len
    self._reuse_len = reuse_len
    self._batch_size = batch_size
    self._num_predict = num_predict
    self._perm_size = perm_size
    self._leak_ratio = leak_ratio
    self._max_num_tokens = max_num_tokens
    self._min_num_tokens = min_num_tokens
    self._cls_id = cls_id
    self._sep_id = sep_id
    self._shuffle_files = shuffle_files

    if reuse_len % perm_size != 0:
      raise ValueError(f'`reuse_len` must be divisible by `perm_size`, got '
          '{reuse_len} and {perm_size}.')

  def _token_span_mask(self, token_ids):
    """Sample the indices of prediction targets (i.e. consecutive tokens) for
    the span-based prediction mask.

    The input sequence is split into multiple sequence spans as follows
    1st span: L..LT..TR..R

    which is immediately followd by

    2nd span: L..LT..TR..R

    ... ...

    Note that each span "L..LT..TR..R" starts with a left context "L..L",
    followed by the prediction targets "T..T", and ended with the right context
    "R..R".

    Args:
      token_ids: int tensor of shape [seq_len], sequence of token IDs in a
        single batch.

    Returns:
      mask: bool tensor of shape [seq_len], vector indicating whether the token
        is the prediction target (1) or not (0).
    """
    # compute the partial prediction constant, i.e. "K" as in the paper
    mask_alpha = self._seq_len / self._num_predict
    round_to_int = lambda x: tf.cast(tf.round(x), 'int64')

    # sample `num_predict` lengths ("L" as in the paper) from the set
    # {min_num_tokens, min_num_tokens + 1, ... max_num_tokens}, such that
    # shorter lengths are more likely than longer ones
    span_len_seq = tf.cast(
        tf.range(self._min_num_tokens, self._max_num_tokens + 1), 'float32')
    probs = 1.0 / (span_len_seq + 1)
    probs /= tf.reduce_sum(probs)
    logits = tf.math.log(probs)
    span_lens = tf.random.categorical(
        logits=logits[tf.newaxis],
        num_samples=self._num_predict,
        dtype='int64')[0] + self._min_num_tokens
    span_lens_float = tf.cast(span_lens, 'float32')

    # each span has length of `span_lens_float[i] * mask_alpha` (i.e. L * K),
    # where `span_lens_float[i]` (i.e. L) is allocated for prediction target,
    # while the remaining `span_lens_float[i] * (mask_alpha - 1)` is allocated
    # as the left plus the right context.
    left_ratio = tf.random.uniform(
        shape=[self._num_predict], minval=0.0, maxval=1.0)
    left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)
    left_ctx_len = round_to_int(left_ctx_len)

    # for each sequence span, compute the total length of prediction targets
    # plus the right context
    right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

    # and the actual start and end indices of prediction targets
    start_indices = (
        tf.cumsum(left_ctx_len) + tf.cumsum(right_offset, exclusive=True))
    end_indices = start_indices + span_lens

    # remove out of range indices
    valid_indices = end_indices < self._seq_len
    start_indices = tf.boolean_mask(start_indices, valid_indices)
    end_indices = tf.boolean_mask(end_indices, valid_indices)

    # shuffle valid indices
    num_valid = tf.cast(tf.shape(start_indices)[0], 'int64')
    order = tf.random.shuffle(tf.range(num_valid, dtype='int64'))
    start_indices = tf.gather(start_indices, order)
    end_indices = tf.gather(end_indices, order)

    mask = self._index_pair_to_mask(
        start_indices, end_indices, token_ids)
    return mask

  def _index_pair_to_mask(self, start_indices, end_indices, token_ids):
    """Convert start and end indices of prediction targets into binary mask.

    Args:
      startindices: int tensor of shape [num_spans], start indices of
        prediction targets.
      end_indices: int tensor of shape [num_spans], end indices of prediction
        targets.
      token_ids: int tensor of shape [seq_len], sequence of token IDs in a
        single batch.

    Returns:
      mask: bool tensor of shape [seq_len], vector indicating whether the token
        is the prediction target (1) or not (0).
    """
    # [seq_len]
    non_func_mask = tf.logical_and(tf.not_equal(token_ids, self._sep_id),
                                   tf.not_equal(token_ids, self._cls_id))

    # [seq_len]
    all_indices = tf.where(
        non_func_mask,
        tf.range(self._seq_len, dtype='int64'),
        tf.constant(-1, shape=[self._seq_len], dtype='int64'))

    # [num_spans, seq_len]
    candidate_matrix = tf.logical_and(
        all_indices[tf.newaxis] >= start_indices[:, tf.newaxis],
        all_indices[tf.newaxis] < end_indices[:, tf.newaxis])

    # [num_spans, seq_len]
    cumsum_matrix = tf.reshape(
        tf.cumsum(tf.reshape(tf.cast(candidate_matrix, 'float32'), [-1])),
        [-1, self._seq_len])

    mask = tf.reduce_any(
        tf.logical_and(candidate_matrix, cumsum_matrix <= self._num_predict),
        axis=0)

    return mask

  def _local_perm(self, mask):
    """Create permutation mask.

    Args:
      mask: bool tensor of shape [mask_seq_len], vector indicating whether the
        token is the prediction target (1) or not (0).

    Returns:
      perm_mask: bool tensor of shape [mask_seq_len, mask_seq_len], where
        the `i`th token cannot attend the `j`th token if `perm_mask[i, j] = 1`.
    """
    # the new indices of each token after a random permutation order is sampled
    index = tf.range(self._reuse_len, dtype='int64')
    index = tf.transpose(tf.reshape(index, [-1, self._perm_size]))
    index = tf.random.shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    # non-prediction targets can always self-attend
    can_attend_self = tf.logical_not(mask)

    smallest_index = tf.constant(-2, dtype='int64', shape=[self._reuse_len])

    if self._leak_ratio > 0:
      # add a small portion of prediciton targets to the set of
      # "can-attend-self" tokens
      leak_tokens = tf.logical_and(
          mask,
          tf.random.uniform([self._reuse_len], maxval=1.0) < self._leak_ratio)
      can_attend_self = tf.logical_or(can_attend_self, leak_tokens)

    to_index = tf.where(can_attend_self, smallest_index, index)
    from_index = tf.where(can_attend_self, to_index + 1, to_index)

    can_attend = from_index[:, tf.newaxis] > to_index[tf.newaxis, :]

    perm_mask = 1.0 - tf.cast(can_attend, 'float32')

    return perm_mask

  def build_dataset(self, filenames):
    """Builds dataset for pretraining XLNet model.

    Args:
      filenames: a list of strings, names of TFRecord files.

    Returns:
      dataset: an instance of tf.data.Dataset, each item is a dict with the
        following entries
        'perm_mask' -> float tensor of shape [batch_size, seq_len, seq_len],
          where the `i`th token cannot attend the `j`th token if
          `perm_mask[b, i, j] = 1`.
        'token_ids' -> int tensor of shape [batch_size, seq_len], sequences of
          token IDs
        'target_mapping' -> float tensor of shape [batch_size, num_predict,
          seq_len], where `target_mapping[b, i]` is the one-hot encoding of the
          index of the prediction target for the `i` prediction task (out of
          `num_predict`). May be zero-padded in the 2nd dimension.
        'target' -> int tensor of shape [batch_size, num_predict], the token
          indices of the prediction targets. May be zero-padded in the 2nd
          dimension.
        'target_mask' -> float tensor of shape [batch_size, num_predict],
          vectors indicating if an entry in `target` is the actual prediction
          target (1) or padded value (0).
        'segment_ids' -> int tensor of shape [batch_size, seq_len], where
          `segment_ids[b]` is an vector of segment IDs for each token in
          `token_ids`.
    """
    def func(example):
      token_ids = example['token_ids']

      # compute permutation mask
      mask = self._token_span_mask(token_ids)
      perm_mask_0 = self._local_perm(mask[:self._reuse_len])
      perm_mask_1 = self._local_perm(mask[self._reuse_len:])
      non_reuse_len = self._seq_len - self._reuse_len
      perm_mask_0 = tf.concat(
          [perm_mask_0, tf.ones([self._reuse_len, non_reuse_len])], axis=1)
      perm_mask_1 = tf.concat(
          [tf.zeros([non_reuse_len, self._reuse_len]), perm_mask_1], axis=1)
      perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)

      # compute target mapping and mask
      indices = tf.range(self._seq_len, dtype='int64')
      indices = tf.boolean_mask(indices, mask)
      actual_num_predict = tf.shape(indices)[0]
      pad_len = self._num_predict - actual_num_predict

      target_mapping = tf.one_hot(indices, self._seq_len, dtype='float32')
      paddings = tf.zeros([pad_len, self._seq_len], dtype='float32')
      target_mapping = tf.concat([target_mapping, paddings], axis=0)

      target = tf.boolean_mask(token_ids, mask)
      paddings = tf.zeros([pad_len], dtype=target.dtype)
      target = tf.concat([target, paddings], axis=0)

      target_mask = tf.concat([
          tf.ones([actual_num_predict], dtype='float32'),
          tf.zeros([pad_len], dtype='float32')], axis=0)

      example['perm_mask'] = perm_mask
      example['target_mapping'] = target_mapping
      example['target'] = target
      example['target_mask'] = target_mask

      for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
          val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
          val = tf.cast(val, 'int32')

        example[key] = val

      return example

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if self._shuffle_files:
      dataset = dataset.shuffle(len(filenames))
    
    dataset = tf.data.TFRecordDataset(dataset)

    dataset = dataset.cache().repeat().map(
        functools.partial(parse_fn_xlnet_pretrain, seq_len=self._seq_len))

    dataset = dataset.map(func)

    dataset = dataset.batch(self._batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


class SquadDatasetBuilder(object):
  def __init__(self, batch_size, seq_len, training=False):
    self._batch_size= batch_size
    self._seq_len = seq_len
    self._training = training

  def build_dataset(self, filenames):

    parse_fn = functools.partial(parse_fn_squad,
                                 seq_len=self._seq_len,
                                 training=self._training) 

    dataset = create_and_preprocess(filenames,
                          parse_fn=parse_fn,
                          shuffle=self._training,
                          num_parallel_calls=1,
                          filter_fn=None,
                          buffer_size_per_file=None,
                          random_seed=None)

    dataset = dataset.batch(self._batch_size)
    if self._training:
      dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


class ClassificationDatasetBuilder(object):
  def __init__(self, batch_size, seq_len, training=False):
    self._batch_size = batch_size
    self._seq_len = seq_len
    self._training = training

  def build_dataset(self, filenames):
    parse_fn = functools.partial(parse_fn_sequence_classification_bert,
                                 seq_len=self._seq_len) 

    dataset = create_and_preprocess(filenames,
        parse_fn=parse_fn,
        shuffle=self._training,
        num_parallel_calls=1,
        filter_fn=None,
        buffer_size_per_file=None,
        random_seed=None)
    dataset = dataset.batch(self._batch_size)
    if self._training:
      dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

