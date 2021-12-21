"""Defines functions parsing serialized examples."""
import tensorflow as tf


def parse_fn_sequence_pair(
    serialized_example, keys=('source', 'target'), dtype=tf.int64):
  """Deserializes a protobuf message into a source and target (batched) sequence
  of token IDs for sequence transduction.

  Args:
    serialized_example: string scalar tensor, serialized example of
      a source-target pair.
    keys: tuple of two strings, keys of the sequence pair parsing dict.
    dtype: string or tensorflow dtype, data type.

  Returns:
    src_tokens_ids: int tensor of shape [src_seq_len], token ids of source
      sequence.
    tgt_tokens_ids: int tensor of shape [tgt_seq_len], token ids of target
      sequence.
  """
  parse_dict = {keys[0]: tf.io.VarLenFeature(tf.int64),
                keys[1]: tf.io.VarLenFeature(tf.int64)}

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  src_token_ids = tf.cast(tf.sparse.to_dense(parsed[keys[0]]), dtype)
  tgt_token_ids = tf.cast(tf.sparse.to_dense(parsed[keys[1]]), dtype)
  return src_token_ids, tgt_token_ids


def parse_fn_single_sequence(serialized_example):
  """Deserializes a protobuf message into a single (batched) sequence of token
  IDs.

  Args:
    serialized_example: string scalar tensor, serialized example of
      a token IDs.

  Returns:
    token_ids: int tensor of shape [seq_len], token ids.
  """
  parse_dict = {'token_ids': tf.io.VarLenFeature(tf.int64)}
  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  token_ids = tf.sparse.to_dense(parsed['token_ids'])
  return token_ids


def parse_fn_sequence_classification(serialized_example):
  """Deserializes a protobuf message into a single (batched) sequence of token
  IDs and class label.

  Args:
    serialized_example: string scalar tensor, serialized example of
      a token IDs and sequence label.

  Returns:
    token_ids: int tensor of shape [seq_len], token_ids.
    label: int scalar tensor, sequence class label.
  """
  parse_dict = {'token_ids': tf.io.VarLenFeature(tf.int64),
                'label': tf.io.VarLenFeature(tf.int64)}

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  token_ids = tf.sparse.to_dense(parsed['token_ids'])
  label = tf.sparse.to_dense(parsed['label'])

  return token_ids, label


def parse_fn_xlnet_pretrain(serialized_example, seq_len=512):
  """Deserializes a protobuf message into a single (batched) sequence of token
  IDs along with their segment IDs.

  Args:
    serialized_example: string scalar tensor, serialized example of
      a token IDs and segment IDs.
    seq_len: (Optional) int scalar, sequence length. Defaults to 512.

  Returns:
    parsed: a dict with the following entries
      'token_ids' -> int tensor of shape [seq_len], sequence of token IDs in a
        single batch.
      'seg_ids' -> int tensor of shape [seq_len], sequence of segment IDs for
        each token in `token_ids`.
  """
  parse_dict = {'token_ids': tf.io.FixedLenFeature([seq_len], 'int64'),
                'seg_ids': tf.io.FixedLenFeature([seq_len], 'int64')}
  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  return parsed


def parse_fn_squad(serialized_example, seq_len=512, training=False):
  """Deserializes a protobuf message into a single (batched) sequence of token
  IDs along with metadata for training or evaluating XLNet model on SQuAD
  dataset.

  Args:
    serialized_example: string scalar tensor, serialized example of
      a token IDs and segment IDs.
    seq_len: (Optional) int scalar, sequence length. Defaults to 512.
    training: (Optional) bool scalar, True for train split and False for dev
      split.

  Returns:
    parsed: a dict with the following entries
      'token_ids' -> int tensor of shape [seq_len], sequence of token IDs in a
        single batch.
      'pad_mask' -> float tensor of shape [seq_len], sequence of 1's and 0's
        where 1's indicate padded (masked) tokens.
      'seg_ids' -> int tensor of shape [seq_len], sequence of segment IDs (
        paragraph, question, CLS, and padded tokens).
      'cls_index' -> scalar int tensor, index of the CLS token.
      'para_mask' -> float tensor of shape [seq_len], sequence of 1's and 0's
        where 1's indicate non-paragraph (masked) tokens.
      and with the following additional entries if training is True
        'start_position': scalar int tensor, token-based start index of answer
          text.
        'end_position': scalar int tensor, token-based end index of answer text.
        'is_impossible': scalar bool tensor, the binary classification label.
  """
  parse_dict = {
      'token_ids': tf.io.FixedLenFeature([seq_len], 'int64'),
      'pad_mask': tf.io.FixedLenFeature([seq_len], 'float32'),
      'seg_ids': tf.io.FixedLenFeature([seq_len], 'int64'),
      'cls_index': tf.io.FixedLenFeature([], 'int64'),
      'para_mask': tf.io.FixedLenFeature([seq_len], 'float32')}

  if training:
    parse_dict['start_position'] = tf.io.FixedLenFeature([], 'int64')
    parse_dict['end_position'] = tf.io.FixedLenFeature([], 'int64')
    parse_dict['is_impossible'] = tf.io.FixedLenFeature([], 'float32')

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)

  for k in parsed.keys():
    if parsed[k].dtype == tf.int64:
      parsed[k] = tf.cast(parsed[k], 'int32')

  return parsed


def parse_fn_sequence_classification_bert(serialized_example, seq_len=512):
  """Deserializes a protobuf message into a single (batched) sequence of token
  IDs along with labels and other satellite data for training sequence
  classification models.

  Args:
    serialized_example: string scalar tensor, serialized example of
      a token IDs and segment IDs.
    seq_len: (Optional) int scalar, sequence length. Defaults to 512.

  Returns:
    parsed: a dict with the following entires
      'token_ids' -> int tensor of shape [seq_len], sequence of token IDs in a
        single batch.
      'pad_mask' -> float tensor of shape [seq_len], sequence of 1's and 0's
        where 1's indicate padded (masked) tokens.
      'seg_ids' -> int tensor of shape [seq_len], sequence of segment IDs.
      'label_ids' -> scalar int tensor, sequence-level label.
  """
  parse_dict = {'token_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
                'pad_mask': tf.io.FixedLenFeature([seq_len], tf.float32),
                'seg_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
                'label_ids': tf.io.FixedLenFeature([], tf.int64)}

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)

  return parsed

