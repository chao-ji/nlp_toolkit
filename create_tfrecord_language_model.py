""""Convert text corpus (raw text files) into TFRecord files for language 
modeling.
"""
import os

import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from tokenization import create_tokenizer_from_raw_text_files
from utils import dict_to_example


FLAGS = flags.FLAGS

flags.DEFINE_list(
    'filenames', None, 'Names of files storing text corpus')
flags.DEFINE_integer(
    'min_count', 0, 'The minimum count required for a token to be included in '
        'the vocabulary')
flags.DEFINE_integer(
    'batch_size', 32, 'The number of sequence segments packed in a batch.')
flags.DEFINE_integer(
    'tgt_len', 224, 'Length of sequence segment.')
flags.DEFINE_string(
    'vocab_name', 'vocab', 'Vocabulary will be stored in text file `vocab`, '
        'one token per line')
flags.DEFINE_string(
    'output_dir', None, 'Path to the directory that the generated TFRecord '
        'files will be written to')
flags.DEFINE_string(
    'output_filename', 'lm.tfrecord', 'Name of the output file.')


def main(_):
  filenames = FLAGS.filenames
  min_count = FLAGS.min_count
  batch_size = FLAGS.batch_size
  tgt_len = FLAGS.tgt_len
  vocab_name = FLAGS.vocab_name
  output_dir = FLAGS.output_dir
  output_filename = FLAGS.output_filename
  tokenizer = create_tokenizer_from_raw_text_files(filenames)

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
  flags.mark_flag_as_required('output_dir')
  app.run(main)


