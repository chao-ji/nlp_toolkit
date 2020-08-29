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
      scale_embeddings: (Optional) whether to scale the embeddings by square 
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
