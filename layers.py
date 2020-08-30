"""Functions that add layers/operations to TensorFlow's computation graph."""
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
  """The customized layer that operates in Embedding mode or Logits mode.

  - Embedding mode converts token ids to embedding vectors.
    Input: [batch_size(N), seq_len(T)]
    Weight: [vocab_size(V), hidden_size(D)]
    Output: [batch_size(N), seq_len(T), hidden_size(D)]
  
  - Logits mode converts embedding vectors to logits.
    Input: [batch_size(N), seq_len(T), hidden_size(D)]
    Weight: [vocab_size(V), hidden_size(D)]
    Output: [batch_size(N), seq_len(T), vocab_size(V)]

  Note that Logits mode reuses the same weight matrix in Embedding mode.
  """
  def __init__(self, vocab_size, hidden_size, scale_embeddings=True):
    """Constructor.

    Args:
      vocab_size: int scalar, num of tokens (including SOS and EOS) in the 
        vocabulary.
      hidden_size: int scalar, the hidden size of continuous representation.
      scale_embeddings: bool scalar, whether to scale the embeddings by square 
        root of hidden size. Defaults to True.
    """
    super(EmbeddingLayer, self).__init__()
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._scale_embeddings = scale_embeddings
    self.add_weight('weights',
                    shape=[vocab_size, hidden_size],
                    initializer=tf.keras.initializers.RandomNormal(
                        mean=0., stddev=hidden_size ** -0.5))

  def call(self, inputs, mode):
    """Either converts token ids to embeddings, or embeddings to logits.

    Args:
      inputs: int tensor of shape [batch_size, seq_len] in "embedding" mode, the
        sequences token ids; or float tensor of shape [batch_size, seq_len, 
        hidden_size] in "logits" mode, the sequences in continuous 
        representation.
      mode: string scalar, "embedding" or "logits".

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        "embedding" mode, the sequences in continuous representation; or float 
        tensor of shape [batch_size, seq_len, vocab_size] in "logits" mode, the
        logits preceding the softmax.
    """
    if mode == 'embedding':
      outputs = self._tokens_to_embeddings(inputs)
    elif mode == 'logits':
      outputs = self._embeddings_to_logits(inputs)
    else:
      raise ValueError('Invalid mode {}'.format(mode))
    return outputs

  def _get_vocab_embeddings(self):
    """Returns the embedding matrix (of shape [vocab_size, hidden_size]). Note 
    that SOS token (index 0) has a fixed (not trainable) zero embedding vector.
    """
    return tf.pad(self.weights[0][1:], [[1, 0], [0, 0]])

  def _tokens_to_embeddings(self, inputs):
    """The dense layer that converts token IDs to embedding vectors.

    Args:
      inputs: int tensor of shape [batch_size, seq_len], the sequences token 
        ids.

    Returns:
      embeddings: float tensor of shape [batch_size, seq_len, hidden_size], the
        sequences in continuous representation.
    """
    # [vocab_size, hidden_size]
    embeddings = self._get_vocab_embeddings()

    # [batch_size, seq_len, hidden_size]
    embeddings = tf.gather(embeddings, inputs)

    if self._scale_embeddings:
      embeddings *= self._hidden_size ** 0.5
    embeddings = tf.cast(embeddings, 'float32')
    return embeddings

  def _embeddings_to_logits(self, decoder_outputs):
    """The dense layer preceding the softmax that computes the logits.

    Args:
      decoder_outputs: float tensor of shape [batch_size, tgt_seq_len, 
        hidden_size], the sequences in continuous representation.

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
        logits preceding the softmax.
    """
    # [vocab_size, hidden_size]
    embeddings = self._get_vocab_embeddings()
    batch_size = tf.shape(decoder_outputs)[0]
    tgt_seq_len = tf.shape(decoder_outputs)[1]

    # [batch_size * tgt_seq_len, hidden_size]
    decoder_outputs = tf.reshape(decoder_outputs, [-1, self._hidden_size])
    logits = tf.matmul(decoder_outputs, embeddings, transpose_b=True)
    logits = tf.reshape(logits, [batch_size, tgt_seq_len, self._vocab_size])
    return logits


def compute_loss(labels, logits, smoothing, vocab_size, padding_value=0):
  """Computes average (per-token) cross entropy loss.

  1. Applies label smoothing -- all entries in the groundtruth label tensor  
     get non-zero probability mass.
  2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
     positions are masked, and then the sum of per token loss is normalized by
     the total number of non-padding entries.

  Args:
    labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
      token ids.
    logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
      predicted logits of tokens over the vocabulary.
    smoothing: float scalar, the amount of label smoothing applied to the
      one-hot class labels. 
    vocab_size: int scalar, num of tokens (including SOS and EOS) in the 
      vocabulary.
    padding_value: int scalar, the vocabulary index of the PAD token. 

  Returns:
    loss: float scalar tensor, the per-token cross entropy
  """
  # effective_vocab = vocab - {SOS_ID}
  effective_vocab_size = vocab_size - 1

  # prob mass allocated to the token that should've been predicted 
  on_value = 1.0 - smoothing
  # prob mass allocated to all other tokens
  off_value = smoothing / (effective_vocab_size - 1)

  # [batch_size, tgt_seq_len, vocab_size] 
  labels_one_hot = tf.one_hot(
      labels,
      depth=vocab_size,
      on_value=on_value,
      off_value=off_value)

  # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
  # because SOS_ID should never appear in the decoded sequence
  # [batch_size, tgt_seq_len]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])

  # this is the entropy when the softmax'ed logits == groundtruth labels
  # so it should be deducted from `cross_entropy` to make sure the minimum 
  # possible cross entropy == 0
  normalizing_constant = -(on_value * tf.math.log(on_value) +
      (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
  cross_entropy -= normalizing_constant

  # mask out predictions where the labels == `padding_value`  
  weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
  cross_entropy *= weights
  loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
  return loss
