"""Convert text corpus (raw text files) into TFRecord files for language 
modeling.

Usage:
 
  1. Create subword tokenizer from raw text file(s) 
  $ python create_tfrecord_language_model.py \
      --filenames=file1.txt,file2.txt... \
      --subword 

  2. Restore subword tokenizer from existing vocab file `vocab_name`
  $ python create_tfrecord_language_model.py \
      --filenames=file1.txt,file2.txt... \
      --subword \
      --vocab_name=vocab_name \
      --use_exist_vocab

  3. Create whole word tokenizer from raw text file(s)
  $ python create_tfrecord_language_model.py \
      --filenames=file1.txt,file2.txt... \
      
  4. Restore whole word tokenizer from existing vocab file `vocab_name`
      --filenames=file1.txt,file2.txt... \
      --vocab_name=vocab_name
      --use_exist_vocab    
"""
import os

import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

import tokenization
from utils import dict_to_example


FLAGS = flags.FLAGS

flags.DEFINE_list(
    'filenames', None, 'Names of files storing text corpus.')
flags.DEFINE_bool(
    'subword', False, 'Whether to use subword tokenizer. Defaults to False.')
flags.DEFINE_integer(
    'min_count', 0, 'The minimum count required for a token to be included in '
        'the vocabulary.')
flags.DEFINE_integer(
    'target_vocab_size', 32000, 'The desired vocabulary size. Ignored if '
        '`subword` is False.')
flags.DEFINE_integer(
    'threshold', 320, 'If the difference between actual vocab size and '
        '`target_vocab_size` is smaller than this, the binary search '
        'terminates. Ignored if `subword` is False.')
flags.DEFINE_float(
    'file_byte_limit', 1e8, 'Number of bytes to read from each text file. '
        'Ignored if `subword` is False.')
flags.DEFINE_integer(
    'batch_size', 32, 'The number of sequence segments packed in a batch.')
flags.DEFINE_integer(
    'tgt_len', 224, 'Length of sequence segment.')
flags.DEFINE_string(
    'vocab_name', 'vocab', 'Name of the file that the vocabulary will be saved '
        ' to or restored from. Contains one token per line.')
flags.DEFINE_string(
    'output_dir', '.', 'Path to the directory that the generated TFRecord '
        'files will be written to')
flags.DEFINE_string(
    'output_filename', 'data.tfrecord', 'Name of the output file.')
flags.DEFINE_bool(
    'use_exist_vocab', False, 'Whether to create (sub)tokenizer from existing '
        ' vocabualry. Defaults to False.')


def main(_):
  filenames = FLAGS.filenames
  subword = FLAGS.subword
  min_count = FLAGS.min_count
  target_vocab_size = FLAGS.target_vocab_size
  threshold = FLAGS.threshold
  file_byte_limit = FLAGS.file_byte_limit
  batch_size = FLAGS.batch_size
  tgt_len = FLAGS.tgt_len
  vocab_name = FLAGS.vocab_name
  output_dir = FLAGS.output_dir
  output_filename = FLAGS.output_filename
  use_exist_vocab = FLAGS.use_exist_vocab

  if subword:
    if use_exist_vocab:
      print('Restore subtokenizer from existing vocab: %s...' % vocab_name)
      tokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_name)
    else:
      print('Create fresh subtokenizer from raw text files...')
      tokenizer = tokenization.create_subtokenizer_from_raw_text_files(
          filenames,
          target_vocab_size,
          threshold,
          file_byte_limit=file_byte_limit)
      tokenizer.save_to_file(vocab_name)
  else:
    if use_exist_vocab:
      print('Restore whole word tokenizer from existing vocab: %s...' 
          % vocab_name)
      tokenizer = tokenization.restore_tokenizer_from_vocab_files(vocab_name)
    else:
      print('Create fresh whole word tokenizer from raw text files...')
      tokenizer = tokenization.create_tokenizer_from_raw_text_files(
          filenames, min_count=min_count)
      tokenizer.save_to_file(vocab_name)

  data = [] 
  for filename in filenames:
    with open(filename) as f:
      for line in f:
        data.extend(tokenizer.encode(line.strip(), add_eos=True)) 
  data = np.array(data)

  writer = tf.io.TFRecordWriter(os.path.join(output_dir, output_filename))

  num_steps = data.size // batch_size  
  data = data[:batch_size * num_steps]
  data = data.reshape(batch_size, num_steps)

  for i in range(0, data.shape[1] - 1, tgt_len):
    cur_tgt_len = min(data.shape[1] - 1 - i, tgt_len)
    for idx in range(batch_size):
      inputs = data[idx, i:i + cur_tgt_len]
      labels = data[idx, i + 1: i + cur_tgt_len + 1]

      example = dict_to_example({'inputs': inputs, 'labels': labels})
      writer.write(example.SerializeToString())

if __name__ == '__main__':
  flags.mark_flag_as_required('filenames')
  app.run(main)
