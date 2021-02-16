"""Defines tokenizer that encodes input string into "whole word" token ids, and
decodes the whole word token ids back to the original input string.
"""
import collections

import tensorflow as tf

UNK = '<UNK>'
SOS = '<SOS>'
EOS = '<EOS>'

# vocab index of START-OF-SEQUENCE token (or PADDING token)
SOS_ID = 0
# vocab index of END-OF-SEQUENCE token
EOS_ID = 1


class WhiteSpaceTokenizer(object):
  """Whole word tokenizer that simply splits input string into space-delimited 
  tokens. 

  Example:
    Assuming the ID of EOS is 1

    tokenizer = WhiteSpaceTokenizer(token_list)
    
    string = 'Tokenizer: hello world !'
    tokenizer.encode('Tokenizer: hello world !')                                           

    Out: [24, 73459, 268, 509, 1]    

    ids = [24, 73459, 268, 509, 1]
    original_string = tokenizer.decode(ids)
    
    Out: '<unk> hello world ! <eos>'     

  As shown above, there is no guarantee that `string = decode(encode(string))`
  because `string` may contain unknown tokens. 
  """
  def __init__(self, token_list, lower_case=False):
    """Constructor.

    Args:
      token_list: a list of strings, whole word tokens.
      lower_case: bool scalar, whether to lower-case tokens.
    """
    if lower_case:
      token_list = [token.lower() for token in token_list]

    self._token_list = token_list
    self._token_dict = {token: index for index, token in enumerate(token_list)}
    self._lower_case = lower_case

  @property
  def vocab_size(self):
    return len(self._token_list)

  def save_to_file(self, filename):
    """Save the tokenizer's vocabulary to a '.token' file, from which the same 
    tokenizer can be restored later.
    
    Args:
      filename: string scalar, the name of the vocabulary file.
    """
    with tf.io.gfile.GFile(filename + '.tokens', mode='w') as f:
      for token in self._token_list:
        f.write("%s\n" % token)

  def encode(self, string, add_eos=False):
    """Encodes a raw string into token ids.

    Args:
      string: a string scalar, the raw text to be encoded.
      add_eos: bool scalar, whether or not to add End-Of-Sequence `EOS` token.

    Returns:
      token_ids: a list of ints, the token ids.
    """
    if self._lower_case:
      string = string.lower()

    tokens = string.split()

    token_ids = [self._token_dict[token] if token in self._token_dict 
        else self._token_dict[UNK] for token in tokens]

    if add_eos:
      token_ids.append(EOS_ID)
    return token_ids

  def decode(self, token_ids):
    """Decode a list of token ids back to string.

    Args:
      token_ids: a list of ints, the token ids.

    Returns:
      a string scalar, the decoded string.
    """
    token_list = []

    for token_id in token_ids:
      if token_id < self.vocab_size:
        token_list.append(self._token_list[token_id])
      else:
        raise ValueError('token id %d not in vocab' % token_id)

    return ' '.join(token_list)


def create_tokenizer_from_raw_text_files(filenames,
                                         target_vocab_size=None, 
                                         min_count=0,
                                         file_char_limit=1e6):
  """Builds a vocabulary of whole word tokens from raw text files and creates a
  a tokenizer.

  The special tokens SOS and EOS has fixed IDs 0 and 1, respectively. UNK is 
  used to handle out-of-vocabulary words.

  If `target_vocab_size` is not None, it will be used to truncate the vocabulary 
  to the most frequent `target_vocab_size` tokens. Otherwise, the vocabulary 
  will be left un-truncated.

  Args:
    filenames: a list of strings, names of raw text files.
    target_vocab_size: (Optional) int scalar, the desired vocabulary size. 
      Defaults to None.
    min_count: (Optional) in scalar, the minimum count required for a token to 
      be included in the vocabulary.
    file_char_limit: int scalar, the max num of chars worth of text to be 
      sampled from each raw text file to build the vocabulary.

  Returns:
    tokenizer: a Tokenizer instance. 
  """
  token_counts = collections.Counter()
  token_list = [SOS, EOS]

  for filename in filenames:
    with tf.io.gfile.GFile(filename, mode='r') as f:
      file_char_budget = file_char_limit
      counter = 0
      lines_to_skip = int(f.size() / file_char_budget)
      for line in f:
        if counter < lines_to_skip:
          counter += 1
        else:
          if file_char_budget < 0:
            break
        line = line.strip()
        file_char_budget -= len(line)
        counter = 0

        token_counts.update(line.split())

  has_unk = False
  for token, count in token_counts.most_common(target_vocab_size):
    if count >= min_count:
      token_list.append(token)
      if token == UNK:
        has_unk = True

  if not has_unk:
    token_list.append(UNK)

  tokenizer = WhiteSpaceTokenizer(token_list)

  return tokenizer


def restore_tokenizer_from_vocab_files(filename, lower_case=False):
  """Restores the tokenizer from vocabulary files ('*.token').

  Args:
    filename: string scalar, the name of the vocabulary file.
    lower_case: bool scalar, whether to lower-case tokens.

  Returns:
    tokenizer: a Tokenizer instance.
  """
  token_list = []
  with tf.io.gfile.GFile(filename + '.tokens', mode='r') as f:
    for line in f:
      token_list.append(line.strip())
  tokenizer = WhiteSpaceTokenizer(token_list, lower_case)
  return tokenizer
