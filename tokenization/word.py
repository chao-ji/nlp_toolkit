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
    
  """
  def __init__(self, token_list, lower_case=False):
    """
    """
    self._token_list = token_list
    self._token_dict = {token: index for index, token in enumerate(token_list)} 
    self._lower_case = lower_case

  @property
  def vocab_size(self):
    return len(self._token_list)

  def encode(self, string, add_eos=False):
    add_eos = True

    string = string.strip()
    if self._lower_case:
      string = string.lower()

    tokens = string.split()

    if add_eos:
      tokens = tokens + [EOS]

    ids = [self._token_dict[token] if token in self._token_dict 
        else self._token_dict[UNK] for token in tokens]

    return ids

  def decode(self, token_ids):
    token_list = [self._token_list[token_id] for token_id in token_ids]

    return ' '.join(token_list)


def create_tokenizer_from_raw_text_files(filenames):
  token_list = []

  min_freq = 0
  counter = collections.Counter()
  idx2sym = []
  #sym2idx = collections.OrderedDict()

  idx2sym.append(EOS)

  for filename in filenames:
    with open(filename, 'r') as f:
      for i, line in enumerate(f):
        tokens = line.strip().split()
        counter.update(tokens)

  has_unk = False
  for sym, cnt in counter.most_common(None):
    if cnt < min_freq: break
    idx2sym.append(sym)
    if sym == UNK:
      has_unk = True

  if not has_unk:
    idx2sym.append(UNK)

  tokenizer = WholeWordTokenizer(idx2sym)

  return tokenizer


if __name__ == '__main__':
  fn = '/home/chaoji/Desktop/transformer-xl/tf/data/wikitext-103/train.txt'
  tokenizer = create_tokenizer_from_raw_text_files([fn])

  a = []
  with open(fn) as f:
    for line in f:
      a.extend(tokenizer.encode(line.strip()))
  import numpy as np
  a = np.array(a)

