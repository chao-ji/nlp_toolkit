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
                        mean=0., stddev=hidden_size ** -0.5),
                    dtype='float32',
                    trainable=True)

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
    return tf.pad(self.trainable_variables[0][1:], [[1, 0], [0, 0]])

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
    logits = tf.einsum('NTD,VD->NTV', decoder_outputs, embeddings)
    return logits


class AdaptiveInputSoftmax(tf.keras.layers.Layer):
  """Implements adaptive input representation and adaptive softmax in a single
  layer.

  Adaptive input and adaptive softmax are two closely related techniques 
  designed to speed up computation for neural network based language models (
  e.g. RNN, Transformer) with very large vocabulary, by reducing the hidden size
  of the embedding vectors of LOW-frequency tokens. Briefly, the tokens in the 
  vocabulary are sorted in descending order of frequency, and split up into 
  disjoint partitions (known as "head", "tail1", "tail2", ...), where "head" 
  contains the most high-frequency tokens, and "tail" partitions contain 
  increasingly less frequent tokens, and the token-to-embedding and 
  embedding-to-softmax computation are performed separately for each partition.

  Note that in this implementation the two modules share the same set of 
  weights. For general description, refer to https://arxiv.org/abs/1809.10853 
  and https://arxiv.org/abs/1609.04309

  It operates in three modes:
    - `embedding` mode: converts token ids to embedding vectors.
      Input: [batch_size(N), seq_len(T)]
      Output: [batch_size(N), seq_len(T), hidden_size(D)]
  
    - `softmax` mode: converts embedding vectors to probability distributions 
        over tokens in the vocabulary.
      Input: [batch_size(N), seq_len(T), hidden_size(D)]
      Output: [batch_size(N), seq_len(T), vocab_size(V)]

    - `loss` mode: computes per-token "head loss" and "tail loss" given 
      embedding vectors and groundtruth next-token ids.
      Input: [batch_size(N), seq_len(T), hidden_size(D)], [batch_size(N), 
        seq_len(T)]
      Output: [head_size + tail_size1 + tail_size2 + ...]
  """
  def __init__(self,
               hidden_size,
               cutoffs,
               project_factor=4,
               kernel_initializer='glorot_uniform'):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      cutoffs: list of ints, the boundaries of token indices (tokens are sorted 
        in descending order of frequencies in the vocabulary) with which to 
        split the set of tokens into a head partition and potentially multiple 
        tail partitions. For example, for `cutoff` = [b0, b1, ..., V], where `V` 
        is the vocabulary size, the head contains tokens with indices in 
        `[0, b0)`, and first tail contains tokens with indices in `[b0, b1)`, 
        and so on.
      project_factor: int scalar, the factor by which to decrease the hidden 
        size of token embeddings for different tails. For example, tokens in the
        head has hidden size `hidden_size`, while tokens in the first tail has
        reduced hidden size `hidden_size // project_factor`, and so on.
      kernel_initializer: string scalar, the weight initializer.
    """
    super(AdaptiveInputSoftmax, self).__init__()
    self._hidden_size = hidden_size
    self._cutoffs = cutoffs
    self._project_factor = project_factor
    self._kernel_initializer = 'glorot_uniform'

    self._num_tails = len(self._cutoffs) - 1

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Example:

    Given `cutoffs = [b0, b1, b2, V]` and hidden size `hidden_size`, there is 1 
    head partition, `[0, b0)`, and 3 tail partitions, `[b0, b1)`, `[b1, b2)`, 
    `[b2, V)`. 
    We create trainable weights for 
      - head partition, i.e. `head_weight_proj: [hidden_size, hidden_size]` and 
        `head_weight: [hidden_size, b0 + 3]`
      - tail partition i, i.e. `tail_weight_proj{i}: [hidden_size, 
        project_size{i}]` and `tail_weight{i}`: [projet_size{i}, cutoffs[i + 1]
         - cutoffs[i]], where `project_size{i}` is the reduced hidden size for 
        tail `i`, and `i` = 0, 1, 2 

    Args:
      inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    self.add_weight(name='head_weight_proj',
                    shape=(self._hidden_size, self._hidden_size),
                    initializer=self._kernel_initializer,
                    dtype='float32',
                    trainable=True)

    self.add_weight(name='head_weight',
                    shape=(self._hidden_size,
                           self._cutoffs[0] + self._num_tails),
                    initializer=self._kernel_initializer,
                    dtype='float32',
                    trainable=True)

    for i in range(self._num_tails):
      project_size = max(1,
          self._hidden_size // self._project_factor ** (i + 1))
      self.add_weight(name='tail_weight_proj_%d' % i,
                      shape=(self._hidden_size, project_size),
                      initializer=self._kernel_initializer,
                      dtype='float32',
                      trainable=True)

      tail_size = self._cutoffs[i + 1] - self._cutoffs[i]
      self.add_weight(name='tail_weight_%d' % i,
                      shape=(project_size, tail_size),
                      initializer=self._kernel_initializer,
                      dtype='float32',
                      trainable=True)
    super(AdaptiveInputSoftmax, self).build(inputs_shape)

  def call(self, inputs, labels=None, mode='softmax'):
    """Runs the forward pass with different behavior according to the `mode`. 

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], 
        embedding representation of tokens, which are outputs from the model 
        (e.g. RNN or Transformer), for mode 'softmax' and 'loss'; Or int tensor 
        of shape [batch_size, seq_len], the sequences token ids, for mode 
        'embedding'.
      labels: (Optional) int tensor of shape [batch_size, seq_len], the 
        groundtruth next-token ids. Must be provided for mode 'loss'.
      mode: (Optional) string scalar, either 'embedding', 'softmax' or 'loss'. 

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        "embedding" mode, the sequences in continuous representation; or float
        tensor of shape [batch_size, seq_len, vocab_size], the per-token 
        probability distribution over tokens in vocabulary; or float tensor of 
        shape [head_size + tail1_size + tail2_size + ...]. 
    """
    if mode == 'embedding':
      return self._tokens_to_embedding(inputs)
    elif mode == 'softmax':
      return self._embeddings_to_softmax(inputs)
    elif mode == 'loss':
      return self._compute_loss(inputs, labels)
    else:
      raise ValueError('`mode` must be "embedding", "softmax" or "loss", got %s'
          % mode)

  def _tokens_to_embedding(self, inputs):
    """Converts input token ids to embedding vectors using adaptive input
    representation.

    Args:
      inputs: int tensor of shape [batch_size, seq_len], the sequences token 
        ids.

    Returns:
      embeddings: float tensor of shape [batch_size, seq_len, hidden_size], the
        sequences in continuous representation.
    """
    # [batch_size, seq_len, hidden_size]
    output_shape = tf.concat([tf.shape(inputs), [self._hidden_size]], axis=0)
    embeddings = []

    weights = [tf.transpose(weight) for weight
        in self.trainable_variables[1::2]]
    weights[0] = weights[0][:self._cutoffs[0]]
    weight_projs = [tf.transpose(weight) for weight
        in self.trainable_variables[::2]]

    for i in range(len(self._cutoffs)):
      low, high = 0 if i == 0 else self._cutoffs[i - 1], self._cutoffs[i]
      mask = tf.logical_and(inputs >= low, inputs < high)
      curr_ids = tf.boolean_mask(inputs, mask) - low

      # [num_valid, hidden_size]
      curr_embeddings = tf.matmul(
          tf.gather(weights[i], curr_ids), weight_projs[i])

      # [num_valid, 2]

      mask_idx = tf.cast(tf.where(mask), 'int32')
      # [batch_size, seq_len, hidden_size]
      embeddings.append(tf.scatter_nd(mask_idx, curr_embeddings, output_shape))

    embeddings = tf.add_n(embeddings) * self._hidden_size ** 0.5
    return embeddings

  def _embeddings_to_softmax(self, inputs):
    """Converts the outputs from the model (e.g. RNN, Transformer) to 
    probability distributions over vocabulary tokens using adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input token embeddings computed from the last layer of 
        the model.

    Returns:
      softmax: float tensor of shape [batch_size, seq_len, vocab_size], the 
        per-token probability distribution over tokens in vocabulary. 
    """
    head_weight_proj, head_weight = self.trainable_variables[:2]

    # [batch_size, seq_len, cutoffs[0] + num_tails]
    head_logits = tf.matmul(inputs, head_weight)
    head_softmax = tf.nn.softmax(head_logits)

    softmax_list = [head_softmax[:, :, :self._cutoffs[0]]]
    for i in range(self._num_tails):
      tail_weight_proj = self.trainable_variables[i*2+2]
      tail_weight = self.trainable_variables[i*2+3]

      # [batch_size, seq_len, cutoffs[i + 1] - cutoffs[i]]
      tail_logits = tf.matmul(tf.matmul(inputs, tail_weight_proj), tail_weight)
      tail_softmax = tf.nn.softmax(tail_logits)
      index = self._cutoffs[0] + i
      softmax_list.append(tail_softmax * head_softmax[:, :, index:index+1])

    softmax = tf.concat(softmax_list, axis=2)
    return softmax

  def _compute_loss(self, inputs, labels):
    """Computes the per-token loss using adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        tensor holding input token embeddings computed from the last layer of 
        the model.
      labels: int tensor of shape [batch_size, seq_len], the groundtruth 
        next-token ids. 

    Returns:
      losses: float tensor of shape [head_size + tail1_size + tail2_size + ...],
        the per-token loss, where `head_size = batch_size * seq_len`, and 
        `tail_size{i}` is the num of token ids in `labels` that fall within in
        the range of token indices of tail partition `i`.
    """
    head_weight_proj, head_weight = self.trainable_variables[:2]

    training_losses = []
    head_labels = labels
    for i in range(1, len(self._cutoffs)):
      tail_weight_proj = self.trainable_variables[i*2]
      tail_weight = self.trainable_variables[i*2+1]

      low, high = self._cutoffs[i - 1], self._cutoffs[i]
      mask = tf.logical_and(labels >= low, labels < high)
      head_labels = tf.where(mask, self._cutoffs[0] + i - 1, head_labels)

      tail_inputs = tf.boolean_mask(inputs, mask)
      tail_logits = tf.matmul(tf.matmul(
          tail_inputs, tail_weight_proj), tail_weight)
      tail_labels = tf.boolean_mask(labels - self._cutoffs[i - 1], mask)

      tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tail_labels, logits=tail_logits)
      training_losses.append(tail_loss)

    head_logits = tf.matmul(inputs, head_weight)

    head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=head_labels, logits=head_logits)
    head_loss = tf.reshape(head_loss, [-1])
    training_losses.append(head_loss)

    losses = tf.concat(training_losses, axis=0)
    return losses


class Projection(tf.keras.layers.Layer):
  """Linearly projects a batch of continuously represented sequences of tokens.

  This projection layer operates in either Split mode or Merge mode:

    - Split mode converts the input sequences in the original representation 
      into the multi-headed "query", "key" or "value" for the attention 
      computation. 

      Input: [batch_size(N), seq_len(T), hidden_size(D)] 
      Weight: [hidden_size(D), num_heads(H), size_per_head(S)]
      Output: dot([N*T, D], [D, H*S]) reshape ==> [N, T, H, S]

    - Merge mode performs the opposite action of Split, converting the 
      multi-headed "value" back to the original representation.

      Input: [batch_size(N), seq_len(T), num_heads(H), size_per_head(S)]
      Weight: [num_heads(H), size_per_head(S), hidden_size(D)]
      Output: dot([N*T, H*S], [H*S, D]) reshape ==> [N, T, D]
  """
  def __init__(self,
               num_heads,
               size_per_head,
               kernel_initializer='glorot_uniform',
               mode="split"):
    """Constructor.

    Args:
      num_heads: int scalar, num of attention heads.
      size_per_head: int scalar, the hidden size of each attention head.
      kernel_initializer: string scalar, the weight initializer.
      mode: string scalar, mode of projection ("split" or "merge") . 
    """
    super(Projection, self).__init__()
    if mode not in ('split', 'merge'):
      raise ValueError('"mode" must be either "split" or "merge".')
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = num_heads * size_per_head
    self._kernel_initializer = kernel_initializer
    self._mode = mode

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor, the last element 
        corresponds to the depth.
    """
    depth = inputs_shape[-1]
    if depth is None:
      raise ValueError('The depth of inputs must not be None.')

    if self._mode == 'merge':
      kernel_shape = self._num_heads, self._size_per_head, self._hidden_size
    else:
      kernel_shape = self._hidden_size, self._num_heads, self._size_per_head

    self.add_weight(name='kernel',
                    shape=kernel_shape,
                    initializer=self._kernel_initializer,
                    dtype='float32',
                    trainable=True)
    super(Projection, self).build(inputs_shape)

  def call(self, inputs):
    """Performs the projection.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] in Merge mode, or float tensor of shape [batch_size, 
        seq_len, hidden_size] in Split mode.

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        Merge mode, or float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] int Split mode.
    """
    kernel = self.trainable_variables[0]
    if self._mode == 'merge':
      outputs = tf.einsum('NTHS,HSD->NTD', inputs, kernel)
    else:
      outputs = tf.einsum('NTD,DHS->NTHS', inputs, kernel)
    return outputs


class FeedForwardNetwork(tf.keras.layers.Layer):
  """The Projection layer that consists of a tandem of two dense layers (an
  intermediate layer and an output layer).
  """
  def __init__(self, hidden_size, filter_size, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation, 
        which is also the depth of the output dense layer.
      filter_size: int scalar, the depth of the intermediate dense layer. 
      dropout_rate: float scalar, dropout rate for the Dropout layers. 
    """
    super(FeedForwardNetwork, self).__init__()
    self._hidden_size = hidden_size
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._dense_layer_filter = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu)
    self._dense_layer_output = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training):
    """Performs projection through two dense layers.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        input sequences.
      training: bool scalar, True if in training mode.

    Return:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        output sequences.
    """
    outputs = self._dense_layer_filter(inputs)
    outputs = self._dropout_layer(outputs, training=training)
    outputs = self._dense_layer_output(outputs)
    return outputs


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
