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


def _token_span_mask(inputs, tgt_len, num_predict, min_num_tokens,
                     max_num_tokens, cls_id, sep_id):
  """Sample token spans as prediction targets."""
  mask_alpha = tgt_len / num_predict
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(min_num_tokens, max_num_tokens + 1)
  probs = np.array([1.0 / (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + min_num_tokens

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_float = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

  # Get the actual begin and end indices
  beg_indices = (
      tf.cumsum(left_ctx_len) + tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range indices
  valid_idx_mask = end_indices < tgt_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict, cls_id, sep_id)

def _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len, num_predict, cls_id, sep_id):
  """Turn beg and end indices into actual mask."""
  non_func_mask = tf.logical_and(
      tf.not_equal(inputs, sep_id), tf.not_equal(inputs, cls_id))
  all_indices = tf.where(non_func_mask, tf.range(tgt_len, dtype=tf.int64),
                         tf.constant(-1, shape=[tgt_len], dtype=tf.int64))
  candidate_matrix = tf.cast(
      tf.logical_and(all_indices[None, :] >= beg_indices[:, None],
                     all_indices[None, :] < end_indices[:, None]), tf.float32)
  cumsum_matrix = tf.reshape(
      tf.cumsum(tf.reshape(candidate_matrix, [-1])), [-1, tgt_len])
  masked_matrix = tf.cast(cumsum_matrix <= num_predict, tf.float32)
  target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
  is_masked = tf.cast(target_mask, tf.bool)

  return is_masked, target_mask

def _local_perm(inputs, is_masked, perm_size, seq_len, leak_ratio, cls_id, sep_id):
  index = tf.range(seq_len, dtype=tf.int64)
  index = tf.transpose(tf.reshape(index, [-1, perm_size]))
  index = tf.random.shuffle(index)
  index = tf.reshape(tf.transpose(index), [-1])

  # non-functional tokens
  non_func_tokens = tf.logical_not(
      tf.logical_or(tf.equal(inputs, sep_id), tf.equal(inputs, cls_id)))
  masked_tokens = tf.logical_and(is_masked, non_func_tokens)
  non_masked_or_func_tokens = tf.logical_not(masked_tokens)

  smallest_index = -2 * tf.ones([seq_len], dtype=tf.int64)

  # Similar to BERT, randomly leak some masked tokens
  if leak_ratio > 0:
    leak_tokens = tf.logical_and(
        masked_tokens,
        tf.random.uniform([seq_len], maxval=1.0) < leak_ratio)
    can_attend_self = tf.logical_or(non_masked_or_func_tokens, leak_tokens)
  else:
    can_attend_self = non_masked_or_func_tokens
  to_index = tf.where(can_attend_self, smallest_index, index)
  from_index = tf.where(can_attend_self, to_index + 1, to_index)

  # For masked tokens, can attend if i > j
  # For context tokens, always can attend each other
  can_attend = from_index[:, None] > to_index[None, :]

  # In modeling, 1 indicates cannot attend. Hence, reverse the value here.
  perm_mask = 1.0 - tf.cast(can_attend, tf.float32)

  # Only masked tokens are included in the loss
  target_mask = tf.cast(masked_tokens, tf.float32)

  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  inputs_q = masked_tokens

  return perm_mask, target_mask, inputs_k, inputs_q




class XLNetPretrainDatasetBuilder(object):
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
                sep_id=4):
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

  def build_dataset(self, filenames):
    def func(example):
      token_ids = example.pop('token_ids')
      boundary = None

      
      is_masked, _ = _token_span_mask(token_ids,
                                      self._seq_len,
                                      self._num_predict,
                                      self._min_num_tokens,
                                      self._max_num_tokens,
                                      self._cls_id,
                                      self._sep_id)

      non_reuse_len = self._seq_len - self._reuse_len

      # assert

      perm_mask_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
          token_ids[:self._reuse_len],
          is_masked[:self._reuse_len],
          self._perm_size, self._reuse_len,
          self._leak_ratio, self._cls_id, self._sep_id)

      perm_mask_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
          token_ids[self._reuse_len:],
          is_masked[self._reuse_len:],
          self._perm_size, non_reuse_len,
          self._leak_ratio, self._cls_id, self._sep_id)

      perm_mask_0 = tf.concat(
          [perm_mask_0, tf.ones([self._reuse_len, non_reuse_len])], axis=1)
      perm_mask_1 = tf.concat(
          [tf.zeros([non_reuse_len, self._reuse_len]), perm_mask_1], axis=1)

      perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)
      target_mask = tf.concat([target_mask_0, target_mask_1], axis=0)
      input_k = tf.concat([input_k_0, input_k_1], axis=0)
      input_q = tf.concat([input_q_0, input_q_1], axis=0)

      example['perm_mask'] = tf.reshape(perm_mask, [self._seq_len, self._seq_len])
      example['input_ids'] = tf.reshape(input_k, [self._seq_len])
      example['input_q'] = tf.reshape(input_q, [self._seq_len])

      target = token_ids

      indices = tf.range(self._seq_len, dtype='int64')
      bool_target_mask = tf.cast(target_mask, 'bool')
      indices = tf.boolean_mask(indices, bool_target_mask)     

      actual_num_predict = tf.shape(indices)[0]
      pad_len = self._num_predict - actual_num_predict 

      target_mapping = tf.one_hot(indices, self._seq_len, dtype='float32')
      paddings = tf.zeros([pad_len, self._seq_len], dtype='float32')
      target_mapping = tf.concat([target_mapping, paddings], axis=0)
      example['target_mapping'] = tf.reshape(target_mapping, 
              [self._num_predict, self._seq_len])

      target = tf.boolean_mask(target, bool_target_mask)
      paddings = tf.zeros([pad_len], dtype=target.dtype)
      target = tf.concat([target, paddings], axis=0) 
      example['target'] = tf.reshape(target, [self._num_predict])


      target_mask = tf.concat([
          tf.ones([actual_num_predict], dtype='float32'),
          tf.zeros([pad_len], dtype='float32')], axis=0)
      example['target_mask'] = tf.reshape(target_mask, [self._num_predict])

      for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
          val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
          val = tf.cast(val, 'int32')

        example[key] = val

      return example

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(len(filenames))
    
    dataset = tf.data.TFRecordDataset(dataset)

    dataset = dataset.cache().repeat().map(parse_fn_xlnet_pretrain)

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

