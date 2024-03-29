"""Defines BeamSearch class and utility functions."""
import tensorflow as tf
import numpy as np


# large negative value considered numerically as -inf
NEG_INF = -1e9

CUR_INDEX = "CUR_INDEX"
ACTIVE_SEQ = "ACTIVE_SEQ"
ACTIVE_LOG_PROBS = "ACTIVE_LOG_PROBS"
ACTIVE_CACHE = "ACTIVE_CACHE"
FINISHED_SEQ = "FINISHED_SEQ"
FINISHED_SCORES = "FINISHED_SCORES"
FINISHED_FLAGS = "FINISHED_FLAGS"


class BeamSearch(object):
  """Beam Search Decoder.

  This implementation of beam search adopts the "aggressive" strategy -- we DO
  NOT reduce the number of active threads of searches even if some of them get
  finished (i.e. hitting EOS) and hence should be removed from the candidate
  set of active sequences. Instead, we simply swap them out and bring in new
  still-active searches to keep the size of candidate set unchanged. The benefit
  is that it is possible for active sequences to end up with higher scores than
  previously finished searches. By not shrinking the "frontier" of the search,
  we can maximize the chance of getting top-scoring candidates.

  The loop invariants maintained over the search iterations are as follows:

  * CUR_INDEX: the current index of the iteration.
  * ACTIVE_SEQ: top-scoring active sequences.
  * ACTIVE_LOG_PROBS: log-probs of ACTIVE_SEQ
  * ACTIVE_CACHE: dict storing the cache values used during the ongoing searches
    for active sequences.
  * FINISHED_SEQ: top-scoring finished sequences.
  * FINISHED_SCORES: scores (log-probs / length_norm) of FINISHED_SEQ
  * FINISHED_FLAGS: values indicating whether entries in FINISHED_SEQ and
    FINISHED_SCORES are real finished seqs or just placeholders.
  """
  def __init__(self,
               decoding_fn,
               vocab_size,
               batch_size,
               beam_width,
               alpha,
               max_decode_length,
               eos_id,
               logits_as_scores=True):
    """Constructor.

    Args:
      decoding_fn: a callable, callback function used in the decoding step to
        convert current token ids into scores of next-tokens over the
        vocabulary.

        The input arguments are:
          ids: int tensor of shape [batch_size*beam_width, 1].
          cache: nested dictionary of tensors [batch_size*beam_width, ...].
          kwargs: (Optional) dict, additional optional arguments.
        The function returns a tuple of logits and the updated cache:
          logits: a tensor of shape [batch_size*beam_width, vocab_size].
          updated_cache: nested dictionary with the same structure as the
            input cache.
      vocab_size: int scalar, the size of the vocabulary, used for topk
        computation.
      batch_size: int scalar, the inference batch size.
      beam_width: int scalar, number of beams for beam search.
      alpha: float scalar, defining the strength of length normalization.
      max_decode_length: int scalar, the maximum number of steps to decode
        a sequence.
      eos_id: int scalar, ID of end of sentence token.
      logits_as_scores: bool scalar, whether the outputs of `decoding_fn`
        are logits (True) or probabilities (False).
    """
    self._decoding_fn = decoding_fn
    self._vocab_size = vocab_size
    self._batch_size = batch_size
    self._beam_width = beam_width
    self._alpha = alpha
    self._max_decode_length = max_decode_length
    self._eos_id = eos_id
    self._logits_as_scores = logits_as_scores

    self._doubled_beam_width = 2 * self._beam_width
    self._length_normalization = lambda length: tf.pow(
        (5. + tf.cast(length, 'float32')) / 6., self._alpha)

  def search(self, initial_ids, initial_cache):
    """Searches for sequences with greatest log-probs by keeping track of
    `beam_width` most promising candidates (i.e. beams).

    Args:
      initial_ids: int tensor of shape [batch_size], populated with initial ids
        (e.g. SOS_ID).
      initial_cache: dict, storing tensors of shape [batch_size, ...], i.e. the
        context from which to draw information to predict the scores of the next
        tokens.

    Returns:
      finished_seqs: int tensor of shape [batch_size, beam_width,
        decode_seq_len], the finished decoded sequences over all beams.
      finished_scores: float tensor of shape [batch_size, beam_width], the
        scores of finished decoded sequences over all beams.
      active_cache: a dict with the same structure as the input `initial_cache`,
        except that each tensor's shape is now [batch_size, beam_width, ...]
    """
    state, state_shapes = self._create_initial_state(initial_ids, initial_cache)

    finished_state = map_structure(
        tf.stop_gradient,
        tf.while_loop(self._continue_search,
                      self._search_step,
                      loop_vars=[state],
                      shape_invariants=[state_shapes],
                      parallel_iterations=1))[0]

    active_seqs = finished_state[ACTIVE_SEQ]
    active_log_probs = finished_state[ACTIVE_LOG_PROBS]
    finished_seqs = finished_state[FINISHED_SEQ]
    finished_scores = finished_state[FINISHED_SCORES]
    finished_flags = finished_state[FINISHED_FLAGS]
    active_cache = finished_state[ACTIVE_CACHE]

    # flag it True if any beam in a thread of seach in a batch is finished
    # [batch_size]
    finished_cond = tf.reduce_any(finished_flags, 1)

    # if none of the beams end with finished seqs, we return the remaining
    # active seqs.
    # [batch_size, beam_width, decode_seq_len]
    finished_seqs = tf.where(finished_cond[:, tf.newaxis, tf.newaxis],
        finished_seqs, active_seqs)

    # [batch_size, beam_width]
    finished_scores = tf.where(finished_cond[:, tf.newaxis],
        finished_scores, active_log_probs)

    return finished_seqs, finished_scores, active_cache

  def _create_initial_state(self, initial_ids, initial_cache):
    """Creates initial loop invariant tensors and their shapes. This function
    expands the dimensions and tiles the tensors to match beam width, so that
    each beam has its own state (active and finished seqs, scores, and caches).

    Args:
      initial_ids: see `initial_ids` in `search`.
      initial_cache: see `initial_cache` in `search`.

    Returns:
      state: a dict with the following entries
        'CUR_INDEX': int scalar tensor, initialized to 0.
        'ACTIVE_SEQ': tensor of shape [batch_size, beam_width, 1].
        'ACTIVE_LOG_PROBS': tensor of shape [batch_size, beam_width].
        'ACTIVE_CACHE': a dict of the same structure as input `initial_cache`,
          except that each tensor is expanded and tiled to shape
          [batch_size, beam_width, ...].
        'FINISHED_SEQ': tensor of shape [batch_size, beam_width, 1].
        'FINISHED_SCORES': tensor of shape [batch_size, beam_width].
        'FINISHED_FLAGS': tensor of shape [batch_size, beam_width].
      state_shape_invariants: a dict with the same structure as `state`, where
        the values are the shape of the corresponding tensor.
    """
    cur_index = tf.constant(0)

    active_seq = _tile_beam_width(initial_ids, self._beam_width)
    active_seq = tf.expand_dims(active_seq, axis=2)

    # set the initial log-probs of all beams to -inf except that the first beam
    # set to zero, so that we are effectively using only the first beam in the
    # first decoding step
    # active_log_probs: [batch_size, beam_width]
    active_log_probs = tf.tile(tf.constant(
        [[0.] + [-float("inf")] * (self._beam_width - 1)], dtype='float32'),
        [self._batch_size, 1])

    # expand and tile tensors in `active_cache` to `beam_width`
    active_cache = map_structure(lambda tensor:
        _tile_beam_width(tensor, self._beam_width), initial_cache)

    # initialize `finished_seq` and `finishe_scores` with placeholder values,
    # and `finished_flags` with False values (i.e. no seq is finished yet).
    finished_seq = tf.zeros_like(active_seq, dtype='int32')
    finished_scores = tf.zeros_like(active_log_probs, dtype='float32')
    finished_flags = tf.zeros_like(active_log_probs, dtype='bool')

    state = {CUR_INDEX: cur_index,
             ACTIVE_SEQ: active_seq,
             ACTIVE_LOG_PROBS: active_log_probs,
             ACTIVE_CACHE: active_cache,
             FINISHED_SEQ: finished_seq,
             FINISHED_SCORES: finished_scores,
             FINISHED_FLAGS: finished_flags}

    state_shape_invariants = self._get_state_shape_invariant(active_cache)
    return state, state_shape_invariants

  def _get_state_shape_invariant(self, active_cache):
    """Creates the shape invariant for each state tensor that will be checked
    by `tf.while_loop`. Note it's only required that the second dimension
    equals `beam_width` at the beginning of each iteration.
    """
    def cache_shapes(tensor):
      shape = [None] * len(tensor.shape)
      shape[1] = self._beam_width
      return tf.TensorShape(shape)

    state_shape_invariants = {
        CUR_INDEX:
            tf.TensorShape([]),
        ACTIVE_SEQ:
            tf.TensorShape([None, self._beam_width, None]),
        ACTIVE_LOG_PROBS:
            tf.TensorShape([None, self._beam_width]),
        ACTIVE_CACHE:
            map_structure(cache_shapes, active_cache),
        FINISHED_SEQ:
            tf.TensorShape([None, self._beam_width, None]),
        FINISHED_SCORES:
            tf.TensorShape([None, self._beam_width]),
        FINISHED_FLAGS:
            tf.TensorShape([None, self._beam_width])
    }
    return state_shape_invariants

  def _continue_search(self, state):
    """Determines whether to keep searching or terminate.

    We terminate the search if either of the following is True:
      1. `cur_index` >= `max_decode_length`
      2. It is True that for all concurrent searches in a batch, the worst score
        of finished seqs over all beams > the best score of active seqs over all
        beams -- the remaining candidate active seqs will never outscore the
        current finished seqs (because scores of active seqs will certainly get
        lower as we grow the partial seqs).

    Args:
      state: a dict holding the loop invariant tensors over the decoding
        iterations. See `_create_initial_state` for details.

    Returns:
      a bool scalar tensor, whether to continue search (True) or not (False).
    """
    i = state[CUR_INDEX]

    # active_log_probs: [batch_size, beam_width]
    # finished_scores: [batch_size, beam_width]
    # finished_flags: [batch_size, beam_width]
    active_log_probs = state[ACTIVE_LOG_PROBS]
    finished_scores = state[FINISHED_SCORES]
    finished_flags = state[FINISHED_FLAGS]

    # active_log_probs are always negative, so the best scores of active seqs
    # are achieved when the length penalty is maximal
    # best_active_scores: [batch_size]
    max_length_norm = self._length_normalization(self._max_decode_length)
    best_active_scores = active_log_probs[:, 0] / max_length_norm

    # if there are no finished seqs in a batch, set the worst finished score to
    # negative infinity for that batch
    # finished_batch_flags: [batch_size], True if any beam is finished
    # worst_finished_scores: [batch_size]
    finished_batch_flags = tf.reduce_any(finished_flags, 1)
    worst_finished_scores = tf.reduce_min(finished_scores, axis=1)
    worst_finished_scores = tf.where(
        finished_batch_flags, worst_finished_scores, NEG_INF)

    worst_finished_better_than_best_active = tf.reduce_all(
        tf.greater(worst_finished_scores, best_active_scores))
    return tf.logical_and(
        tf.less(i, self._max_decode_length),
        tf.logical_not(worst_finished_better_than_best_active))

  def _search_step(self, state):
    """Performs a single search step.

    Args:
      state: a dict holding the loop invariant tensors over the decoding
        iterations. See `_create_initial_state` for details.

    Returns:
      a length-1 list holding a dict of the same structure as the input `state`
        with updated tensors.
    """
    new_seq, new_log_probs, new_cache = self._grow_active_seq(state)
    active_state = self._get_new_active_state(new_seq, new_log_probs, new_cache)
    finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)

    new_state = {CUR_INDEX: state[CUR_INDEX] + 1}
    new_state.update(active_state)
    new_state.update(finished_state)

    return [new_state]

  def _grow_active_seq(self, state):
    """Grows the search tree of the active sequences by one level, and gathers
    the top-scoring `2 * beam_width` candidates.

    Note: we may have UP TO `beam_width` finished candidates (i.e. ending with
    EOS_ID) among all `vocab_size * beam_width` candidates, so collecting the
    top-scoring `2 * beam_width` candidates would ensure that there are at least
    `beam_width` candidates that are still active (i.e. not ending with EOS_ID).

    Args:
      state: a dict holding the loop invariant tensors over the decoding
        iterations. See `_create_initial_state` for details.

    Returns:
      topk_seq: int tensor of shape [batch_size, doubled_beam_width,
        cur_index + 2], the token ids of the extended top-scoring
        `doubled_beam_width` sequences.
      topk_log_probs: float tensor of shape [batch_size, doubled_beam_width],
        log-probs of the extended top-scoring `doubled_beam_width` sequences.
      new_cache: dict where values of entries are tensors of shape
        [batch_size, doubled_beam_width, ...].
    """
    i = state[CUR_INDEX]
    # active_seq: [batch_size, beam_width, cur_index + 1]
    # active_log_probs: [batch_size, beam_width]
    # values of entries in dict actice_cache: [batch_size, beam_width, ...]
    active_seq = state[ACTIVE_SEQ]
    active_log_probs = state[ACTIVE_LOG_PROBS]
    active_cache = state[ACTIVE_CACHE]

    # flattening
    # for `active_seq` and `active_cache`, do reshaping
    # [batch_size, beam_width, ...] ==> [batch_size * beam_width, ...]
    flat_active_seq = _flatten_beam_dim(active_seq)
    flat_cache = map_structure(_flatten_beam_dim, active_cache)

    # flat_scores: [batch_size * beam_width, vocab_size]
    # tensors in dict flat_cache may have updated shapes and values.
    flat_scores, flat_cache = self._decoding_fn(
        flat_active_seq[:, -1:], flat_cache, index=i)

    if self._logits_as_scores:
      # SOS should be excluded from the space of valid output tokens, so we push
      # the logits of SOS_ID to -inf so that SOS will never appear in the
      # decoded sequence
      sos_mask = tf.constant(
          [1] + [0] * (self._vocab_size - 1), dtype='float32') * NEG_INF
      flat_scores += sos_mask
      convert_to_logprobs_fn = lambda scores: (
          scores - tf.reduce_logsumexp(scores, axis=2, keepdims=True))

    else:
      sos_mask = tf.constant(
          [0] + [1] * (self._vocab_size - 1), dtype='float32')
      flat_scores *= sos_mask
      convert_to_logprobs_fn = lambda scores: tf.math.log(scores)

    # unflattening
    # scores: [batch_size, beam_width, vocab_size]
    # tensors in `new_cache` now have shape [batch_size, beam_width, ...]
    scores = _unflatten_beam_dim(
        flat_scores, self._batch_size, self._beam_width)
    # covert scores to log probs
    candidate_log_probs = convert_to_logprobs_fn(scores)
    new_cache = map_structure(
        lambda t: _unflatten_beam_dim(t, self._batch_size, self._beam_width),
        flat_cache)

    # log_probs: [batch_size, beam_width, vocab_size]
    log_probs = candidate_log_probs + tf.expand_dims(active_log_probs, axis=2)

    flat_log_probs = tf.reshape(log_probs,
                                [-1, self._beam_width * self._vocab_size])

    # top_log_probs, topk_indices: [batch_size, doubled_beam_width]
    topk_log_probs, topk_indices = tf.nn.top_k(
        flat_log_probs, k=self._doubled_beam_width)

    # get the beam indices for the top `doubled_beam_width` candidates
    topk_beam_indices = topk_indices // self._vocab_size

    # topk_seq: [batch_size, doubled_beam_width, cur_index + 1]
    # tensors in `new_cache` now have shape [batch_size, doubled_beam_width,...]
    topk_seq, new_cache = _gather_beams(
        [active_seq, new_cache], topk_beam_indices)

    # append the top `doubled_beam_width` ids (`topk_ids`) to the growing active
    # seqs (`topk_seq`)
    # topk_ids: [batch_size, doubled_beam_width]
    topk_ids = tf.expand_dims(topk_indices % self._vocab_size, axis=2)
    topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
    return topk_seq, topk_log_probs, new_cache

  def _get_new_active_state(self, new_seq, new_log_probs, new_cache):
    """Gathers the top `beam_width` active sequences from the larger pool of
    `2 * beam_width` candidates.

    Args:
      new_seq: same as `topk_seq` in `_grow_active_seq`.
      new_log_probs: same as `topk_log_probs` in `_grow_active_seq`.
      new_cache: same as `new_cache` in `_grow_active_seq`.

    Returns:
      a dict with the following entries:
        'ACTIVE_SEQ': tensor of the same shape as input `new_seq`, except the
          beam dimension changes to `beam_width` from `2 * beam_width`.
        'ACTIVE_LOG_PROBS': tensor of the same shape as input `new_log_probs`,
          except the beam dimension changes to `beam_width` from
          `2 * beam_width`.
        'ACTIVE_CACHE': nested structure of tensors, where each tensor has the
          same shape as counterpart in input `new_cache`, except the beam
          dimension changes to `beam_width` from `2 * beam_width`.
    """
    # [batch_size, doubled_beam_width]
    new_active_flags = tf.logical_not(tf.equal(new_seq[:, :, -1], self._eos_id))
    top_active_seq, top_active_log_probs, top_active_cache = _gather_topk(
        [new_seq, new_log_probs, new_cache],
        new_log_probs, new_active_flags, self._beam_width)

    return {ACTIVE_SEQ: top_active_seq,
            ACTIVE_LOG_PROBS: top_active_log_probs,
            ACTIVE_CACHE: top_active_cache}

  def _get_new_finished_state(self, state, new_seq, new_log_probs):
    """Gets newly finished seqs (if any) and combines them with previously
    finished seqs, and gathers the top-scoring `beam_width` seqs.

    Args:
      state: a dict holding the loop invariant tensors over the decoding
        iterations. See `_create_initial_state` for details.
      new_seq: same as `topk_seq` in `_grow_active_seq`.
      new_log_probs: same as `topk_log_probs` in `_grow_active_seq`.

    Returns:
      a dict with the following entries:
        'FINISHED_SEQ': tensor of shape [batch_size, beam_width, cur_index + 2].
        'FINISHED_SCORES': tensor of shape [batch_size, beam_width].
        'FINISHED_FLAGS': tensor of shape [batch_size, beam_width].
    """
    i = state[CUR_INDEX]
    # finished_seq: [batch_size, beam_width, cur_index + 1]
    # finished_scores: [batch_size, beam_width]
    # finished_flags: [batch_size, beam_width]
    finished_seq = state[FINISHED_SEQ]
    finished_scores = state[FINISHED_SCORES]
    finished_flags = state[FINISHED_FLAGS]

    # zero-pad the previously finished seqs to shape
    # [batch_size, beam_width, cur_index + 2]
    finished_seq = tf.pad(finished_seq, [[0, 0], [0, 0], [0, 1]])

    # convert log-probs to scores by length normalization
    new_scores = new_log_probs / self._length_normalization(i + 1)

    # flag the newly finished seqs (if any)
    # [batch_size, doubled_beam_width]
    new_finished_flags = tf.equal(new_seq[:, :, -1], self._eos_id)

    # combine previously finished seqs w/ those newly finished (if any)
    # finished_seq: [batch_size, beam_width * 3, cur_index + 2]
    # finished_scores: [batch_size, beam_width * 3]
    # finished_flags: [batch_size, beam_width * 3]
    finished_seq = tf.concat([finished_seq, new_seq], axis=1)
    finished_scores = tf.concat([finished_scores, new_scores], axis=1)
    finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

    top_finished_seq, top_finished_scores, top_finished_flags = _gather_topk(
        [finished_seq, finished_scores, finished_flags],
        finished_scores, finished_flags, self._beam_width)

    return {FINISHED_SEQ: top_finished_seq,
            FINISHED_SCORES: top_finished_scores,
            FINISHED_FLAGS: top_finished_flags}


def _tile_beam_width(tensor, beam_width):
  """Given a tensor of shape [batch_size, ...], expands its dims in axis=1
  and tile along axis=1.

  Args:
    tensor: tensor of shape [batch_size, ...]
    beam_width: int scalar, beam width.

  Returns:
    tiled_tensor: tensor of shape [batch_size, beam_width, ...].
  """
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_width
  tiled_tensor = tf.tile(tensor, tile_dims)
  return tiled_tensor


def _get_partial_static_shape(tensor):
  """Returns the (maybe partially) static shape of the input tensor where
  unknown values are replaced with dynamic shape values.
  """
  static_shape = tensor.get_shape().as_list()
  dynamic_shape = tf.shape(tensor)
  for i, v in enumerate(static_shape):
    if v is None:
      static_shape[i] = dynamic_shape[i]
  return static_shape


def _flatten_beam_dim(tensor):
  """Collapses batch and beam dimension into a single dimension.

  Args:
    tensor: tensor of shape [batch_size, beam_width, ...]

  Returns:
    tensor of shape [batch_size * beam_width, ...]
  """
  shape = _get_partial_static_shape(tensor)
  shape[0] *= shape[1]
  shape.pop(1)
  return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_width):
  """Un-collapses the first dimension back into batch and beam dimension.

  Args:
    tensor: tensor of shape [batch_size * beam_width, ...]
    batch_size: int scalar, batch size.
    beam_width: int scalar, beam width.

  Returns:
    tensor of shape [batch_size, beam_width, ...]
  """
  shape = _get_partial_static_shape(tensor)
  new_shape = [batch_size, beam_width] + shape[1:]
  return tf.reshape(tensor, new_shape)


def _gather_beams(nested, beam_indices):
  """Gathers beams from a nested structure of tensors according to beam indices.

  Args:
    nested: a dict, list, tuple or a tensor, where elements are recursively
      dict, list, tuple or a tensor. All tensors have shape [batch_size,
      beam_width, ...].
    beam_indices: int tensor of shape [batch_size, new_beam_width], holding the
      indices of beams (not necessarily unique) to be gathered for each batch.

  Returns:
    an object of the same structure as `nested`, where each tensor has shape
      [batch_size, new_beam_width, ...].
  """
  batch_size, new_beam_width = tf.shape(beam_indices)

  # For example, given batch_size = 3
  # batch_indices = [[0, 0,...,0],[1, 1,...,1], ...,[2, 2,...,2]]
  # each sublist has length `new_beam_width`
  batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis],
                          [1, new_beam_width])
  indices = tf.stack([batch_indices, beam_indices], axis=2)
  return map_structure(
      lambda state: tf.gather_nd(state, indices), nested)


def _gather_topk(nested, scores, flags, k):
  """Gathers top-k scoring valid beams (the corresponding flag is True).

  For example, given
  scores = [-0.32, -0.59, -0.11, -0.05, -0.96, -0.87]
  flags = [True, False, False, True, False, True]
  k = 4

  The scores for False flags will be pushed to -inf, and the results will be
  scores = [-0.05, -0.32, -0.87, -inf]
  flags = [True, True, True, False]


  Note: if the num of valid seqs across all beams for each batch is less than
  `k`, the result is padded with invalid seqs.

  Args:
    nested: a dict, list, tuple or a tensor, where elements are recursively
      dict, list, tuple or a tensor. All tensors have shape [batch_size,
      beam_width, ...].
    scores: float tensor of shape [batch_size, beam_width], the scores of each
      sequence for a particular batch and beam.
    flags: bool tensor of shape [batch_size, beam_width], indicates the validity
      of each sequence (valid if True).
    k: int scalar, the num of top scoring sequences (<= `beam_width`).

  Returns:
    an object of the same structure as `nested`, where each tensor has shape
      [batch_size, k, ...].
  """
  # push the scores of invalid seqs to NEG_INF, so they will be placed after the
  # valid seqs in `indices`
  scores += tf.cast(tf.logical_not(flags), 'float32') * NEG_INF
  _, indices = tf.nn.top_k(scores, k)
  return _gather_beams(nested, indices)


def map_structure(fn, nested):
  """Recursively executes a function over elements organized in a structure
  recursively composed of dict, list or tuple.

  Args:
    fn: a callable, function to be executed.
    nested: a dict, list, tuple or a tensor, where elements are recursively
      dict, list, tuple or a tensor.

  Returns:
    an object of the same structure as `nested` after applying the function.
  """
  if isinstance(nested, dict):
    d = {}
    for k, v in nested.items():
      d[k] = map_structure(fn, v)
    return d
  elif isinstance(nested, (tuple, list)):
    l = []
    for v in nested:
      l.append(map_structure(fn, v))
    return tuple(l) if isinstance(nested, tuple) else l
  else:
    return fn(nested)
