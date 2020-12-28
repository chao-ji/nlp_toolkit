"""Defines functions parsing serialized examples.""" 
import tensorflow as tf


def parse_fn_sequence_pair(serialized_example, keys=('source', 'target')):
  """Deserializes a protobuf message into a source and target (batched) sequence
  of token IDs for sequence transduction.

  Args:
    serialized_example: string scalar tensor, serialized example of 
      a source-target pair.
    keys: tuple of two strings, keys of the sequence pair parsing dict.

  Returns:
    src_tokens_ids: int tensor of shape [src_seq_len], token ids of source
      sequence.
    tgt_tokens_ids: int tensor of shape [tgt_seq_len], token ids of target
      sequence.
  """
  parse_dict = {keys[0]: tf.io.VarLenFeature(tf.int64),
                keys[1]: tf.io.VarLenFeature(tf.int64)}

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  src_token_ids = tf.cast(tf.sparse.to_dense(parsed[keys[0]]), 'float32')
  tgt_token_ids = tf.cast(tf.sparse.to_dense(parsed[keys[1]]), 'float32')
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
