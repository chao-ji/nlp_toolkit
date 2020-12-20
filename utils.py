"""Utility functions and classes. """
import heapq

import numpy as np
import tensorflow as tf


class CosineDecayLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Defines cosine decay learning rate."""
  def __init__(
      self, learning_rate, decay_steps, alpha, warmup_steps, warmup_lr):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      decay_steps: int scalar, num of steps to decay over.
      alpha: float scalar, minimum learning rate value as a fraction of 
        learning rate.
      warmup_steps: int scalar, the num of warm-up steps.
      warmup_lr: float scalar, learning rate for warm-up steps. 
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._decay_steps = decay_steps
    self._alpha = alpha
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr

  def __call__(self, global_step):
    """Computes learning rate. 

    Args:
      global_step: int scalar tensor, the current global step.

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
    global_step = tf.cast(global_step, 'float32')

    cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(global_step
        - self._warmup_steps, self._decay_steps) / self._decay_steps))
    decayed = (1 - self._alpha) * cosine_decay + self._alpha
    decayed_learning_rate = self._learning_rate * decayed

    decayed_learning_rate = tf.where(global_step < self._warmup_steps,
                                     self._warmup_lr,
                                     decayed_learning_rate)

    return decayed_learning_rate


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""
  def __init__(self, learning_rate, hidden_size, warmup_steps):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._hidden_size = hidden_size
    self._warmup_steps = tf.cast(warmup_steps, 'float32')

  def __call__(self, global_step):
    """Computes learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: int scalar tensor, the current global step. 

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`. 
    """
    global_step = tf.cast(global_step, 'float32')
    learning_rate = self._learning_rate
    learning_rate *= (self._hidden_size**-0.5)
    # linear warmup
    learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
    # rsqrt decay
    learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
    return learning_rate


def save_attention_weights(filename, data):
  """Saves attention weights data to *.npy file.

  Args:
    filename: string scalar, filename.
    data: a list or tuple or dict of numpy arrays, the attention weights and 
      token ids of input and translated sequence.
  """
  np.save(filename, data)


def dict_to_example(dictionary):
  """Convert dict to protobuf example message.

  Args:
    dictionary: a dict mapping string to list of integers

  Returns:
    a protobuf example message.
  """
  features = {}
  for k, v in dictionary.items():
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def nucleus_sampling(scores, threshold=0.95):
  """Sample from the head of the probability distribution that contains the 
  vast majority of probability mass. See https://arxiv.org/abs/1904.09751 
  for details. The distribution is truncated to the  and re-normalized.

  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    threshold: float scalar, the minimum value of the sum of probability mass
      that the head of the distribution must exceed. 

  Returns:
    next_token_id: int scalar, the sampled id of the next token.
  """
  ids = np.argsort(-scores)
  cumsum = [0.] + np.cumsum(scores[ids]).tolist()
  # search space is any value >= low and <= high
  low, high = 0, len(cumsum) - 2

  while low <= high:
    mid = (low + high) // 2
    sum1 = cumsum[mid]
    sum2 = cumsum[mid + 1]
    if sum1 < threshold and sum2 >= threshold:
      break
    elif sum2 < threshold: # exclude indices <= mid 
      low = mid + 1
    elif sum1 >= threshold: # exclude indices >= mid
      high = mid - 1
    else:
      raise ValueError('Impossible outcome')

  probs = scores[ids[:mid + 1]] / sum2
  next_token_id = np.random.choice(ids[:mid + 1], p=probs)
  return next_token_id


def topk_sampling(scores, k=40):
  """Sample from the top-k tokens with the largest probability. The distribution
   is truncated and re-normalized.

  Args:
    scores: numpy array of shape [vocab_size], the probability distribution (
      sum to one) of all possible next-tokens over the vocabulary.
    k: int scalar, the num of next-tokens with largest probability to sample 
      from.

  Returns:
    next_token_id: int scalar, the sampled id of the next token.
  """
  min_pq = list(zip(scores[:k], range(k)))
  heapq.heapify(min_pq)
  for i in np.arange(k, len(scores)):
    if scores[i] > min_pq[0][0]:
      min_pq[0] = scores[i], i
      heapq.heapify(min_pq)

  probs, ids = list(zip(*min_pq))
  probs = np.array(probs)
  probs /= probs.sum()
  next_token_id = np.random.choice(ids, p=probs)
  return next_token_id
