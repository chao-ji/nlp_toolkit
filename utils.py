"""Utility functions and classes. """
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
