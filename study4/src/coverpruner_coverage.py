"""CPU-only CoverPruner coverage math; no attention, model or spatial selector.

Paper contract: full-N row calibration and weighted best-representative greedy.
Study4 policies: float32 math, eps_norm=1e-12, equal relevance -> weights1,
exact gain ties -> lowest native visual ordinal, invalid/overflow -> ValueError.
`k` is the TOTAL selected count including seed_indices (25 seeds + k64 ->39
additions). Seed indices are visual ordinals, not merged sequence positions.
"""
from dataclasses import dataclass
import operator

import numpy as np

DEFAULT_TAU = 0.3
DEFAULT_EPS_NORM = 1e-12


def _fp32(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError(f"{name} must contain finite real numbers")
        with np.errstate(over="raise", invalid="raise"):
            result = raw.astype(np.float32, copy=False)
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} is not representable as finite float32") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is not finite float32")
    return result


def _scalar(value, name, *, strictly_positive=False):
    result = _fp32(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    number = result[()]
    if number < 0 or (strictly_positive and number == 0):
        raise ValueError(f"{name} must be {'positive' if strictly_positive else 'nonnegative'}")
    return number


def _square(value, name, *, nonnegative=False):
    result = _fp32(value, name)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _vector(value, name, *, n=None, nonnegative=False):
    result = _fp32(value, name)
    if result.ndim != 1 or result.size == 0 or (n is not None and result.size != n):
        raise ValueError(f"{name} must be a nonempty vector of the required length")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _indices(value, n, name):
    if isinstance(value, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError(f"{name} must not contain booleans")
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    if result.size == 0:
        return ()
    if result.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer ordinals (not booleans)")
    values = tuple(int(x) for x in result)
    if len(set(values)) != len(values) or any(x < 0 or x >= n for x in values):
        raise ValueError(f"{name} contains duplicates or out-of-range ordinals")
    return values


def _sequence_map(value, n):
    if value is None:
        return tuple(range(n))
    if isinstance(value, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError("sequence_indices must not contain booleans")
    result = np.asarray(value)
    if result.ndim != 1 or result.size != n or result.dtype.kind not in "iu":
        raise ValueError("sequence_indices must contain N integer positions")
    values = tuple(int(x) for x in result)
    if len(set(values)) != n or any(x < 0 for x in values):
        raise ValueError("sequence_indices must be unique nonnegative positions")
    return values


def _finite(result, name):
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite {name}")
    return result


def cosine_similarity(embeddings, *, eps_norm=DEFAULT_EPS_NORM):
    """Full projector-space cosine in FP32; zero rows stay zero. No M clipping."""
    x = _fp32(embeddings, "embeddings")
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError("embeddings must have nonempty shape [N,D]")
    eps = _scalar(eps_norm, "eps_norm", strictly_positive=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            norms = np.sqrt(np.sum(x * x, axis=1, dtype=np.float32))
            unit = x / np.maximum(norms[:, None], eps)
            result = unit @ unit.T
    except FloatingPointError as exc:
        raise ValueError("float32 cosine overflow/invalid operation") from exc
    return _finite(result, "cosine similarity")


def coverage_utilities(similarity):
    """A_ij=max(0,M_ij-mean_k M_ik), mean over the ORIGINAL full N."""
    matrix = _square(similarity, "similarity")
    try:
        with np.errstate(over="raise", invalid="raise"):
            means = np.mean(matrix, axis=1, keepdims=True, dtype=np.float32)
            result = np.maximum(matrix - means, np.float32(0))
    except FloatingPointError as exc:
        raise ValueError("float32 row calibration overflow/invalid operation") from exc
    return _finite(result, "coverage utilities")


def normalize_relevance(relevance):
    """Min-max over all N FP32 relevance scores; exact equal range -> zeros."""
    r = _vector(relevance, "relevance")
    lo, hi = np.min(r), np.max(r)
    if hi == lo:
        return np.zeros_like(r)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = (r - lo) / (hi - lo)
    except FloatingPointError as exc:
        raise ValueError("float32 relevance normalization overflow/invalid operation") from exc
    return _finite(result, "normalized relevance")


def demand_weights(relevance, *, tau=DEFAULT_TAU):
    """External raw relevance -> min-max -> exp(tau*r_norm); default tau0.3."""
    temperature = _scalar(tau, "tau")
    normalized = normalize_relevance(relevance)
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.exp(temperature * normalized)
    except FloatingPointError as exc:
        raise ValueError("float32 demand weight overflow/invalid operation") from exc
    return _finite(result, "demand weights")


def _state(matrix, selected):
    if not selected:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return np.max(matrix[:, selected], axis=1)


def coverage_state(utilities, selected_indices=()):
    """Recompute current c from ALL selected columns, including arbitrary seeds."""
    matrix = _square(utilities, "utilities", nonnegative=True)
    selected = _indices(selected_indices, matrix.shape[0], "selected_indices")
    return _state(matrix, selected)


def _objective(weights, state):
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.sum(weights * state, dtype=np.float32)
    except FloatingPointError as exc:
        raise ValueError("float32 objective overflow/invalid operation") from exc
    return float(_finite(result, "coverage objective"))


def coverage_value(utilities, weights, selected_indices=()):
    """F(S), with F(empty)=0. Weights may be any finite nonnegative vector."""
    matrix = _square(utilities, "utilities", nonnegative=True)
    w = _vector(weights, "weights", n=matrix.shape[0], nonnegative=True)
    selected = _indices(selected_indices, matrix.shape[0], "selected_indices")
    return _objective(w, _state(matrix, selected))


def _gains(matrix, weights, state):
    try:
        with np.errstate(over="raise", invalid="raise"):
            residual = np.maximum(matrix - state[:, None], np.float32(0))
            residual *= weights[:, None]
            result = np.sum(residual, axis=0, dtype=np.float32)
    except FloatingPointError as exc:
        raise ValueError("float32 marginal gain overflow/invalid operation") from exc
    return _finite(result, "marginal gains")


def marginal_gain(utilities, weights, selected_indices, candidate_index):
    """Gain of one candidate; already selected candidates have gain zero."""
    matrix = _square(utilities, "utilities", nonnegative=True)
    w = _vector(weights, "weights", n=matrix.shape[0], nonnegative=True)
    selected = _indices(selected_indices, matrix.shape[0], "selected_indices")
    candidate = _indices([candidate_index], matrix.shape[0], "candidate_index")[0]
    return float(_gains(matrix, w, _state(matrix, selected))[candidate])


@dataclass(frozen=True)
class GreedyResult:
    seed_indices: tuple[int, ...]
    selection_order: tuple[int, ...]  # seed order followed by greedy additions
    added_indices: tuple[int, ...]
    selection_order_sequence_indices: tuple[int, ...]
    sorted_visual_indices: tuple[int, ...]  # sorted by mapped sequence position
    sorted_sequence_indices: tuple[int, ...]
    marginal_gains: tuple[float, ...]  # additions ONLY, no fabricated seed gains
    objective_values: tuple[float, ...]  # initial seed F, then F after each addition
    coverage: np.ndarray  # independent read-only final c, never written to disk


def global_greedy(utilities, weights, k=64, *, seed_indices=(), sequence_indices=None):
    """Cached-c global greedy; k INCLUDES seeds, exact ties use visual ordinal.

    No spatial quotas or masks. Zero-gain candidates still fill the requested k.
    Neither utilities/weights nor caller-owned seed/index data are modified.
    """
    matrix = _square(utilities, "utilities", nonnegative=True)
    n = matrix.shape[0]
    w = _vector(weights, "weights", n=n, nonnegative=True)
    seed = _indices(seed_indices, n, "seed_indices")
    positions = _sequence_map(sequence_indices, n)
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer total budget")
    try:
        budget = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be an integer total budget") from exc
    if budget < len(seed) or budget > n or budget < 0:
        raise ValueError("require 0 <= seed count <= total k <= N")
    state = _state(matrix, seed)
    selected = list(seed)
    used = np.zeros(n, dtype=bool)
    used[list(seed)] = True
    gains, values, added = [], [_objective(w, state)], []
    while len(selected) < budget:
        scores = _gains(matrix, w, state)
        scores[used] = -np.inf  # internal sentinel only, never an input fallback
        chosen = int(np.argmax(scores))  # first maximum = lowest native ordinal
        gains.append(float(scores[chosen]))
        selected.append(chosen)
        added.append(chosen)
        used[chosen] = True
        state = np.maximum(state, matrix[:, chosen])
        values.append(_objective(w, state))
    ordered = tuple(sorted(selected, key=positions.__getitem__))
    state = state.copy()
    state.flags.writeable = False
    return GreedyResult(
        seed, tuple(selected), tuple(added), tuple(positions[i] for i in selected),
        ordered, tuple(positions[i] for i in ordered), tuple(gains), tuple(values), state,
    )


def select_from_embeddings(embeddings, relevance, k=64, *, seed_indices=(),
                           sequence_indices=None, tau=DEFAULT_TAU,
                           eps_norm=DEFAULT_EPS_NORM):
    """Convenience A pipeline; relevance is EXTERNAL, never an attention probe."""
    utilities = coverage_utilities(cosine_similarity(embeddings, eps_norm=eps_norm))
    weights = demand_weights(relevance, tau=tau)
    return global_greedy(utilities, weights, k, seed_indices=seed_indices,
                         sequence_indices=sequence_indices)