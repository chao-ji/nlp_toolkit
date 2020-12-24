"""Defines WholeWordTokenizer class that encodes raw text into whole word token
ids, and decodes the whole word token ids back to the raw text.
"""
import collections

UNK = '<unk>'
EOS = '<eos>'


class WholeWordTokenizer(object):
  """Whole word tokenizer that splits a string into tokens separated by white 
  spaces.

  Example:
    tokenizer = WholeWordTokenizer(token_list)
    
    string = 'Tokenizer: hello world !'
    tokenizer.encode('Tokenizer: hello world !')                                           

    Out: [24, 73459, 268, 509, 0]    

    ids = [24, 73459, 268, 509, 0]
    original_string = tokenizer.decode(ids)
    
    Out: '<unk> hello world ! <eos>'     
  """
  def __init__(self, token_list, lower_case=False):
    """Constructor.

    Args:
      token_list: a list of strings, whole word tokens.
      lower_case: bool scalar, whether to lower-case tokens.
    """
    self._token_list = token_list
    self._token_dict = {token: index for index, token in enumerate(token_list)}
    self._lower_case = lower_case

  @property
  def vocab_size(self):
    return len(self._token_list)

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

    if add_eos:
      tokens = tokens + [EOS]

    token_ids = [self._token_dict[token] if token in self._token_dict 
        else self._token_dict[UNK] for token in tokens]

    return token_ids

  def decode(self, token_ids):
    """Decode a list of token ids back to original string.

    Args:
      token_ids: a list of ints, the token ids.

    Returns:
      a string scalar, the decoded string.
    """
    token_list = [self._token_list[token_id] for token_id in token_ids]

    return ' '.join(token_list)


def create_tokenizer_from_raw_text_files(filenames,
                                         target_vocab_size=None, 
                                         min_count=0):
  """Builds a vocabulary of whole word tokens from raw text files and creates a
  a tokenizer.

  The special EOS token has fixed ID(0) in the vocabulary, and also has special 
  token UNK.

  Args:
    filenames: a list of strings, names of raw text files.
    target_vocab_size: (Optional) int scalar, the desired vocabulary size. 
      Defaults to None.
    min_count: (Optional) in scalar, the minimum count required for a token to 
      be included in the vocabulary.

  Returns:
    tokenizer: a Tokenizer instance. 
  """
  counter = collections.Counter()
  token_list = [EOS]

  for filename in filenames:
    with open(filename, 'r') as f:
      for line in f:
        tokens = line.strip().split()
        counter.update(tokens)

  has_unk = False
  for token, count in counter.most_common(target_vocab_size):
    if count >= min_count:
      token_list.append(token)
      if token == UNK:
        has_unk = True

  if not has_unk:
    token_list.append(UNK)

  tokenizer = WholeWordTokenizer(token_list)

  return tokenizer
