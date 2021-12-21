"""Defines customized layers as building blocks for different models."""
import math

import tensorflow as tf

from .beam_search import NEG_INF
from .utils import rel_shift


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention.

  Given a batch of vector-represented query sequences (tensor of shape [
  batch_size, q_seq_len, hidden_size]) and context sequences (tensor of shape
  [batch_size, c_seq_len, hidden_size]), this layer computes a new
  representation of the query sequences by making them discriminatively attend
  to tokens in the context sequences.

  If the query and context happen to be the same, the result ends up being
  "Self Attention" -- the query sequence attends to itself.
  """
  def __init__(self, hidden_size, num_heads, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(Attention, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._size_per_head = hidden_size // num_heads

    self._dense_layer_query = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_key = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_value = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_output = Projection(
        num_heads, self._size_per_head, mode='merge')
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, query, context, attention_mask, training=False, cache=None):
    """Computes new representation of query sequences.

    Args:
      query: float tensor of shape [batch_size, q_seq_len, hidden_size],
        query sequences.
      context: float tensor of shape [batch_size, c_seq_len, hidden_size]
        , context sequences.
      attention_mask: float tensor of shape [batch_size, num_heads, q_seq_len,
        c_seq_len], populated with either 0 (for tokens to keep) or 1 (for
        tokens to be masked).
      training: (Optional) bool scalar, True if in training mode.
      cache: (Optional) dict with entries
        'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
          size_per_head],
        'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
          size_per_head],
        'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
          num_heads, tgt_seq_len, tgt_seq_len],
        'tgt_src_attention': tensor of shape [batch_size * beam_width,
          num_heads, tgt_seq_len, src_seq_len].
        Must be provided in inference mode when called within decoder layers.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
        new representation of `query`.
    """
    self_attention = True if id(query) == id(context) else False

    # [batch_size, q_seq_len, num_heads, size_per_head]
    q = self._dense_layer_query(query)

    # [batch_size, c_seq_len, num_heads, size_per_head]
    k = self._dense_layer_key(context)
    v = self._dense_layer_value(context)

    if cache is not None and self_attention:
      # concatenate along the `seq_len` dimension
      cache['k'] = k = tf.concat([cache['k'], k], axis=1)
      cache['v'] = v = tf.concat([cache['v'], v], axis=1)

    # [batch_size, num_heads, q_seq_len, c_seq_len]
    attention_weights = tf.einsum('NQHS,NCHS->NHQC', q, k)
    attention_weights *= self._size_per_head ** -0.5
    attention_weights += attention_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
    attention_weights = self._dropout_layer(
        attention_weights, training=training)

    # save attention weights of encoder layers in inference mode
    if not training and cache is None and self_attention:
      setattr(self, '_attention_weights', attention_weights)

    # save attention weights for visualization in inference mode
    if cache is not None:
      if self_attention:
        # [batch_size, num_heads, tgt_seq_len, tgt_seq_len]
        cache['tgt_tgt_attention'] = tf.concat([tf.pad(
            cache['tgt_tgt_attention'], [[0, 0], [0, 0], [0, 0], [0, 1]]),
            attention_weights], axis=2)
      else:
        # [batch_size, num_heads, tgt_src_len, src_seq_len]
        cache['tgt_src_attention'] = tf.concat([
            cache['tgt_src_attention'], attention_weights], axis=2)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum('NHQC,NCHS->NQHS', attention_weights, v)

    # [batch_size, q_seq_len, hidden_size]
    outputs = self._dense_layer_output(outputs)
    return outputs


class RelativeAttention(Attention):
  """Multi-headed attention with relative position encoding. Processes either
  single-stream input as in TransformerXL / fine-tuning XLNet, or two-stream
  input as in pre-training XLNet.
  """
  def __init__(self, hidden_size, num_heads, dropout_rate, for_xlnet=False):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      for_xlnet: (Optional) bool scalar, whether this layer is used for XLNet
        (True) or TransformerXL (False). If True, the positionwise attention
        weight matrix will be further sliced to match the size of other weight
        matrices. Defaults to False.
    """
    super(RelativeAttention, self).__init__(
        hidden_size, num_heads, dropout_rate)
    self._dense_layer_key_position = Projection(
        num_heads, self._size_per_head, mode='split')
    self._for_xlnet = for_xlnet

  def call(self,
           content_stream,
           content_mask,
           context,
           position_encoding,
           content_bias,
           position_bias,
           query_stream=None,
           query_mask=None,
           target_mapping=None,
           segment_encoding=None,
           segment_matrix=None,
           segment_bias=None,
           training=False):
    """Computes the new representation of input sequences.

    Args:
      content_stream: float tensor of shape [batch_size, q_seq_len, hidden_size]
        , the query sequences for TransformerXL or the content stream for
        pre-training XLNet.
      content_mask: float tensor of shape [batch_size, 1, q_seq_len, c_seq_len],
        token mask for content stream.
      context: float tensor of shape [batch_size, c_seq_len, hidden_size], the
        context sequences that the query sequences attend to.
      position_encoding: float tensor of shape [batch_size, r_seq_len,
        hidden_size], the position encoding for the context sequences.
      content_bias: float tensor of shape [num_heads, size_per_head], content
        bias.
      position_bias: float tensor of shape [num_heads, size_per_head], position
        bias.
      query_stream: (Optional) float tensor of shape [batch_size,
        num_predictions, hidden_size], the query stream for pre-training XLNet.
      query_mask: (Optional) float tensor of shape [batch_size, 1, q_seq_len,
        c_seq_len], token mask for query stream.
      target_mapping: (Optional) float tensor of shape [batch_size,
        num_predictions, q_seq_len], one-hot encodings of the indices of
        prediction targets.
      segment_encoding: (Optional) float tensor of shape [2, num_heads,
        size_per_head], embedding vectors of the binary information that
        whether two positions are from the same segment or not.
      segment_matrix: (Optional) bool tensor of shape [batch_size, q_seq_len,
        c_seq_len], binary matrix indicating whether two positions are from the
        same segment or not.
      segment_bias: (Optional) float tensor of shape [num_heads, size_per_head],
        segment bias.
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], for
        single stream input; or a tuple of two tensors of shape [batch_size,
        q_seq_len, hidden_size] and [batch_size, num_predictions, hidden_size].
    """
    # [batch_size, q_seq_len, num_heads, size_per_head]
    q = self._dense_layer_query(content_stream)

    # [batch_size, c_seq_len, num_heads, size_per_head]
    k_content = self._dense_layer_key(context)
    v = self._dense_layer_value(context)

    # [batch_size, r_seq_len, num_heads, size_per_head]
    k_position = self._dense_layer_key_position(position_encoding)

    kwargs = {'k_content': k_content,
              'v': v,
              'k_position': k_position,
              'content_bias': content_bias,
              'position_bias': position_bias,
              'segment_encoding': segment_encoding,
              'segment_matrix': segment_matrix,
              'segment_bias': segment_bias,
              'training': training}

    content_outputs = self._compute_attention(q, content_mask, **kwargs)
    # [batch_size, q_seq_len, hidden_size]
    content_outputs = self._dense_layer_output(content_outputs)

    if query_stream is not None:
      # [batch_size, num_predictions, num_heads, size_per_head]
      q = self._dense_layer_query(query_stream)
      # [batch_size, q_seq_len, num_heads, size_per_head]
      q = tf.einsum('NPHS,NPQ->NQHS', q, target_mapping)
      query_outputs = self._compute_attention(q, query_mask, **kwargs)
      # [batch_size, num_predictions, num_heads, size_per_head]
      query_outputs = tf.einsum('NQHS,NPQ->NPHS', query_outputs, target_mapping)
      # [batch_size, num_predictions, hidden_size]
      query_outputs = self._dense_layer_output(query_outputs)

      return content_outputs, query_outputs
    else:
      return content_outputs

  def _compute_attention(self,
                         q,
                         attention_mask,
                         k_content,
                         v,
                         k_position,
                         content_bias,
                         position_bias,
                         segment_encoding=None,
                         segment_matrix=None,
                         segment_bias=None,
                         training=False):
    """Compute weighted average of multi-headed values.

    Args:
      q: float tensor of shape [batch_size, q_seq_len, num_heads, size_per_head]
        , multi-headed projected query.
      attention_mask: float tensor of shape [batch_size, num_heads, q_seq_len,
        c_seq_len], populated with either 0 (for tokens to keep) or 1 (for
        tokens to be masked).
      k_content: float tensor of shape [batch_size, c_seq_len, num_heads,
        size_per_head], multi-headed projected content-based key.
      v: float tensor of shape [batch_size, c_seq_len, num_heads, size_per_head]
        , multi-headed projected value.
      k_position: float tensor of shape [batch_size, r_seq_len, num_heads,
        size_per_head], multi-headed projected position-based key.
      content_bias: float tensor of shape [num_heads, size_per_head], content
        bias.
      position_bias: float tensor of shape [num_heads, size_per_head], position
        bias.
      segment_encoding: (Optional) float tensor of shape [2, num_heads,
        size_per_head], embedding vectors of the binary information that
        whether two positions are from the same segment or not.
      segment_matrix: (Optional) bool tensor of shape [batch_size, q_seq_len,
        c_seq_len], binary matrix indicating whether two positions are from the
        same segment or not.
      segment_bias: (Optional) float tensor of shape [num_heads, size_per_head],
        segment bias.
      training: (Optional) bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, num_heads,
        size_per_head], weighted average of multi-headed values.
    """
    content_attention = tf.einsum(
        'NQHS,NCHS->NHQC', q + content_bias, k_content)
    position_attention = tf.einsum(
        'NQHS,NRHS->NHQR', q + position_bias, k_position)
    position_attention = rel_shift(position_attention)
    attention_shape = tf.shape(content_attention)
    if self._for_xlnet:
      position_attention = position_attention[..., 1:1 + attention_shape[3]]

    attention_weights = content_attention + position_attention
    if segment_encoding is not None:
      # [batch_size, num_heads, q_seq_len, 2]
      segment_attention = tf.einsum(
          'NQHS,GHS->NHQG', q + segment_bias, segment_encoding)

      # [batch_size, num_heads, q_seq_len, c_seq_len]
      segment_attention = tf.where(
          tf.broadcast_to(segment_matrix[:, tf.newaxis], attention_shape),
          tf.broadcast_to(segment_attention[..., 1:], attention_shape),
          tf.broadcast_to(segment_attention[..., :1], attention_shape))
      attention_weights += segment_attention

    attention_weights = tf.multiply(
        attention_weights, 1.0 / math.sqrt(float(self._size_per_head)))
    attention_weights += attention_mask * NEG_INF
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
    attention_weights = self._dropout_layer(
        attention_weights, training=training)

    outputs = tf.einsum('NHQC,NCHS->NQHS', attention_weights, v)
    return outputs


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

  Note that Logits mode and Embedding mode share the same weight matrix.
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

  def build(self, inputs_shape):
    """Creates weights of this layer.

    Args:
        inputs_shape: tuple of ints or 1-D int tensor. Not used.
    """
    self.add_weight(name='weights',
                    shape=[self._vocab_size, self._hidden_size],
                    initializer=tf.keras.initializers.RandomNormal(
                        mean=0., stddev=self._hidden_size ** -0.5),
                    dtype='float32',
                    trainable=True)
    super(EmbeddingLayer, self).build(inputs_shape)

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
  e.g. RNN, TransformerXL) with very large vocabulary size. The idea is to
  incrementally reduce the hidden size of the embedding vectors of tokens with
  increasingly lower frequency. Briefly, the tokens in the vocabulary are sorted
  in descending order of frequency, and split up into disjoint partitions (
  "head", "tail1", "tail2", ...), where "head" contains the most high-frequency
  tokens, and "tail1", "tail2", ... contain less frequent tokens. The embedding
  vectors (with REDUCED hidden size) of tokens in "tail" partitions will finally
  be projected through a dense layer to the space with the same hidden size as
  in the "head" partition.

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
      Output: [batch_size(N) * seq_len(T) + num_valid_tail1 + num_valid_tail2 +
        ...]
  """
  def __init__(self,
               hidden_size,
               cutoffs,
               project_factor=4,
               kernel_initializer='glorot_uniform',
               scale_embeddings=True):
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
      scale_embeddings: bool scalar, whether to scale the embeddings by square
        root of hidden size. Defaults to True.
    """
    super(AdaptiveInputSoftmax, self).__init__()
    self._hidden_size = hidden_size
    self._cutoffs = cutoffs
    self._project_factor = project_factor
    self._kernel_initializer = kernel_initializer
    self._scale_embeddings = scale_embeddings

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
      - tail partition `i`, i.e. `tail_weight_proj{i}: [hidden_size,
        project_size{i}]` and `tail_weight{i}`: [projet_size{i}, cutoffs[i]
         - cutoffs[i - 1]], where `project_size{i}` is the reduced hidden size
        for tail `i`, and `i` = 1, 2, 3

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
        embedding representation of tokens, which are outputs from the model (
        e.g. RNN or TransformerXL), for mode 'softmax' and 'loss'; Or int tensor
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
        shape [batch_size(N) * seq_len(T) + num_valid_tail1 + num_valid_tail2 +
        ...].
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
    weight_projs = [tf.transpose(weight_proj) for weight_proj
        in self.trainable_variables[::2]]

    for i in range(len(self._cutoffs)):
      low, high = 0 if i == 0 else self._cutoffs[i - 1], self._cutoffs[i]
      mask = tf.logical_and(inputs >= low, inputs < high)

      # [num_valid_entries]
      curr_ids = tf.boolean_mask(inputs, mask) - low

      # [num_valid_entries, hidden_size]
      curr_embeddings = tf.matmul(
          tf.gather(weights[i], curr_ids), weight_projs[i])

      # [num_valid, 2]
      mask_idx = tf.cast(tf.where(mask), 'int32')
      # [batch_size, seq_len, hidden_size]
      embeddings.append(tf.scatter_nd(mask_idx, curr_embeddings, output_shape))

    embeddings = tf.add_n(embeddings)
    if self._scale_embeddings:
        embeddings *= self._hidden_size ** 0.5
    return embeddings

  def _embeddings_to_softmax(self, inputs):
    """Converts the outputs from the model (e.g. RNN, TransformerXL) to
    probability distributions over vocabulary tokens using adaptive softmax.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the
        tensor holding input token embeddings computed from the last layer of
        the model.

    Returns:
      softmax: float tensor of shape [batch_size, seq_len, vocab_size], the
        per-token probability distribution over tokens in vocabulary.
    """
    head_weight = self.trainable_variables[1]

    # [batch_size, seq_len, cutoffs[0] + num_tails]
    head_logits = tf.matmul(inputs, head_weight)
    head_softmax = tf.nn.softmax(head_logits)

    # [batch_size, seq_len, cutoffs[0]]
    softmax_list = [head_softmax[:, :, :self._cutoffs[0]]]
    for i in range(self._num_tails):
      tail_weight_proj = self.trainable_variables[i * 2 + 2]
      tail_weight = self.trainable_variables[i * 2 + 3]

      # [batch_size, seq_len, tail_size]
      tail_logits = tf.matmul(tf.matmul(inputs, tail_weight_proj), tail_weight)
      tail_softmax = tf.nn.softmax(tail_logits)
      index = self._cutoffs[0] + i
      softmax_list.append(tail_softmax * head_softmax[:, :, index : index + 1])

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
      losses: float tensor of shape [batch_size * seq_len + num_valid1 +
        num_valid2 + ...], the per-token loss, where `num_valid{i}` is the
        num of token ids in `labels` that fall within in the range of token
        indices of tail partition `i`.
    """
    head_weight = self.trainable_variables[1]

    training_losses = []
    head_labels = labels

    # computes loss for tails
    for i in range(1, len(self._cutoffs)):
      tail_weight_proj = self.trainable_variables[i * 2]
      tail_weight = self.trainable_variables[i * 2 + 1]

      low, high = self._cutoffs[i - 1], self._cutoffs[i]
      mask = tf.logical_and(labels >= low, labels < high)

      # update the entries in `head_labels` to `cutoffs[0] + i - 1`, i.e. the
      #"head class ID" for tail partition `i`
      # [batch_size, seq_len]
      head_labels = tf.where(mask, self._cutoffs[0] + i - 1, head_labels)

      # [num_valid_entries, hidden_size]
      tail_inputs = tf.boolean_mask(inputs, mask)

      # [num_valid_entries, tail_size]
      tail_logits = tf.matmul(tf.matmul(
          tail_inputs, tail_weight_proj), tail_weight)

      # [num_valid_entries]
      tail_labels = tf.boolean_mask(labels - self._cutoffs[i - 1], mask)

      # [num_valid_entries]
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
      mode: string scalar, mode of projection ("split" or "merge").
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
  def __init__(self,
               hidden_size,
               filter_size,
               dropout_rate,
               filter_activation=tf.nn.relu):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation,
        which is also the depth of the output dense layer.
      filter_size: int scalar, the depth of the intermediate dense layer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      filter_activation: callable or string, activation function of the filter
        dense layer. Defaults to ReLU.
    """
    super(FeedForwardNetwork, self).__init__()
    self._hidden_size = hidden_size
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._filter_activation = filter_activation

    self._dense_layer_filter = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation=filter_activation)
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
