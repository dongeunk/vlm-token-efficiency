"""Study4 fixed A-F CPU adapters over the UNCHANGED P1-A Coverage engine.

All methods share one prepared projector M/A, external P1-B raw query r and w.
Vision E accepts external scalar scores only; no Vision attention extraction.
F reuses the engine's pinned _gains primitive; no second Coverage formula.
"""
from dataclasses import dataclass
import numpy as np
import coverpruner_coverage as engine
from coverpruner_spatial_layout import SpatialLayout, positive_int, readonly

METHODS = ('A', 'B', 'C', 'D', 'E', 'F')


def _scores(values, n, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] != n or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must contain N finite real scores')
    try:
        with np.errstate(over='raise', invalid='raise'):
            result = raw.astype(np.float32, copy=False)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f'{name} not representable in float32') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{name} not finite float32')
    return result


def _owners(cell_ids, n, num_cells):
    num_cells = positive_int(num_cells, 'num_cells')
    if isinstance(cell_ids, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in cell_ids):
        raise ValueError('Boolean owners prohibited')
    ids = np.asarray(cell_ids)
    if ids.shape != (n,) or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= num_cells):
        raise ValueError('Invalid cell owner vector')
    if np.unique(ids).size != num_cells:
        raise ValueError('Empty cell; no fallback or candidate duplication')
    return ids.astype(np.int64, copy=False), num_cells


def center_seeds(centers_xy, cell_ids, original_size, *, grid_shape=(5, 5)):
    """B: FP64 squared ORIGINAL PIXEL distance to the geometric cell center."""
    rows, cols = (positive_int(x, 'grid_shape') for x in grid_shape)
    raw = np.asarray(centers_xy)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('Centers must be real numeric values')
    xy = raw.astype(np.float64, copy=False)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or ((xy < 0) | (xy > 1)).any():
        raise ValueError('Invalid normalized nominal centers')
    ids, count = _owners(cell_ids, len(xy), rows*cols)
    width, height = (positive_int(x, 'original_size') for x in original_size)
    seeds = []
    try:
        with np.errstate(over='raise', invalid='raise'):
            for cell in range(count):
                r, c = divmod(cell, cols)
                members = np.flatnonzero(ids == cell)  # canonical native ordinal
                delta = (xy[members] - [(c+.5)/cols, (r+.5)/rows]) * [width, height]
                distance = np.sum(delta*delta, axis=1, dtype=np.float64)
                if not np.isfinite(distance).all():
                    raise ValueError('Nonfinite center distance')
                seeds.append(int(members[np.argmin(distance)]))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('Pixel distance overflow') from exc
    return tuple(seeds)


def medoid_seeds(raw_cosine, cell_ids, *, num_cells=25):
    """C: within-cell column mean of raw M including self, NOT centered A/w."""
    matrix = np.asarray(raw_cosine)
    if matrix.dtype.kind not in 'fiu' or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('Invalid raw cosine matrix')
    ids, count = _owners(cell_ids, matrix.shape[0], num_cells)
    seeds = []
    for cell in range(count):
        members = np.flatnonzero(ids == cell)
        try:
            with np.errstate(over='raise', invalid='raise'):
                scores = np.mean(matrix[np.ix_(members, members)], axis=0, dtype=np.float32)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('Medoid mean overflow') from exc
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite medoid score')
        seeds.append(int(members[np.argmax(scores)]))
    return tuple(seeds)


def maximum_score_seeds(scores, cell_ids, *, num_cells=25):
    """D/E: independent per-cell raw-score maximum, exact ties lowest ordinal."""
    n = len(cell_ids)
    values = _scores(scores, n, 'attention scores')
    ids, count = _owners(cell_ids, n, num_cells)
    return tuple(int(members[np.argmax(values[members])]) for members in
                 (np.flatnonzero(ids == cell) for cell in range(count)))


@dataclass(frozen=True)
class ConstrainedSeedResult:
    indices: tuple[int, ...]
    marginal_gains: tuple[float, ...]
    objective_values: tuple[float, ...]
    coverage: np.ndarray


def constrained_coverage_seeds(utilities, weights, cell_ids, *, num_cells=25):
    """F stage1: best marginal over ALL open cells, then close the chosen cell.

    Uses the same P1-A _gains used internally by global_greedy. This pinned
    private primitive is regression/oracle tested; the engine is not modified.
    Coverage is rebuilt from all seeds via public coverage_state each step.
    """
    # Validate/cast using the engine's public zero-budget entry point.
    initial = engine.global_greedy(utilities, weights, 0)
    matrix, w = np.asarray(utilities, dtype=np.float32), np.asarray(weights, dtype=np.float32)
    ids, count = _owners(cell_ids, matrix.shape[0], num_cells)
    state = initial.coverage
    chosen, gains, values = [], [], [initial.objective_values[0]]
    closed = np.zeros(count, dtype=bool)
    for _ in range(count):
        scores = engine._gains(matrix, w, state)
        scores[closed[ids]] = -np.inf
        token = int(np.argmax(scores))
        if closed[ids[token]]:
            raise RuntimeError('No eligible cell candidate')
        gains.append(float(scores[token]))
        chosen.append(token)
        closed[ids[token]] = True
        state = engine.coverage_state(matrix, chosen)
        values.append(engine.coverage_value(matrix, w, chosen))
    return ConstrainedSeedResult(tuple(chosen), tuple(gains), tuple(values), readonly(state))


@dataclass(frozen=True)
class CoverageProblem:
    raw_cosine: np.ndarray
    utilities: np.ndarray
    query_relevance: np.ndarray
    weights: np.ndarray
    sequence_indices: tuple[int, ...]
    layout: SpatialLayout | None


def prepare_problem(embeddings, query_relevance, *, layout=None, sequence_indices=None):
    """External raw P1-B r (synthetic in this phase) -> the unchanged w API.

    None sequence_indices means VISUAL ordinals, not merged prompt positions.
    A future Decoder caller must provide its real original sequence mapping.
    A supports layout=None and never requires spatial ownership.
    """
    matrix = engine.cosine_similarity(embeddings)
    utilities = engine.coverage_utilities(matrix)
    weights = engine.demand_weights(query_relevance, tau=.3)
    n = matrix.shape[0]
    validated = engine.global_greedy(utilities, weights, 0, sequence_indices=sequence_indices)
    del validated
    if layout is not None and (not isinstance(layout, SpatialLayout) or layout.n != n):
        raise ValueError('Layout/candidate count mismatch')
    positions = tuple(range(n)) if sequence_indices is None else tuple(int(x) for x in sequence_indices)
    return CoverageProblem(readonly(matrix), readonly(utilities), readonly(_scores(query_relevance,n,'query relevance')),
                           readonly(weights), positions, layout)


@dataclass(frozen=True)
class SpatialSelectionResult:
    method: str
    greedy: engine.GreedyResult
    seed_indices: tuple[int, ...]
    seed_marginal_gains: tuple[float, ...]
    seed_objective_values: tuple[float, ...]
    seed_cell_counts: tuple[int, ...]
    final_cell_counts: tuple[int, ...]
    vision_score_source: str | None


def select_method(method, problem, *, vision_scores=None):
    if method not in METHODS:
        raise ValueError('Method ID must be fixed A/B/C/D/E/F')
    seed, seed_gains, seed_values = (), (), ()
    layout = problem.layout
    if method != 'A':
        if layout is None or layout.grid_shape != (5,5):
            raise ValueError('B-F require the explicit P0-verified5x5 layout')
        ids = layout.cell_ids
        _owners(ids, len(problem.weights), 25)
        if method == 'B':
            seed = center_seeds(layout.centers_xy, ids, layout.original_size)
        elif method == 'C':
            seed = medoid_seeds(problem.raw_cosine, ids)
        elif method == 'D':
            seed = maximum_score_seeds(problem.query_relevance, ids)
        elif method == 'E':
            if vision_scores is None:
                raise ValueError('E requires external Vision scores; no extractor implemented')
            seed = maximum_score_seeds(vision_scores, ids)
        else:
            stage = constrained_coverage_seeds(problem.utilities, problem.weights, ids)
            seed, seed_gains, seed_values = stage.indices, stage.marginal_gains, stage.objective_values
        if len(seed) != 25 or len(set(seed)) != 25 or not np.all(np.bincount(ids[list(seed)],minlength=25)==1):
            raise RuntimeError('Seed25 one-per-cell invariant failed')
    result = engine.global_greedy(problem.utilities, problem.weights, 64,
                                  seed_indices=seed, sequence_indices=problem.sequence_indices)
    if len(result.selection_order)!=64 or len(set(result.selection_order))!=64:
        raise RuntimeError('Final unique K64 invariant failed')
    if method!='A' and (result.selection_order[:25]!=seed or len(result.added_indices)!=39):
        raise RuntimeError('Seed25+39 continuation invariant failed')
    if seed and result.objective_values[0] != engine.coverage_value(problem.utilities, problem.weights, seed):
        raise RuntimeError('Initial seed coverage was reset')
    seed_counts = tuple(int(x) for x in np.bincount(layout.cell_ids[list(seed)],minlength=25)) if layout and seed else ()
    final_counts = tuple(int(x) for x in np.bincount(layout.cell_ids[list(result.selection_order)],minlength=25)) if layout else ()
    return SpatialSelectionResult(method,result,seed,seed_gains,seed_values,seed_counts,final_counts,
                                   'external_injected_only' if method=='E' else None)


def select_all(problem, *, vision_scores):
    """Share the SAME immutable M/A/r/w and pool across every fixed method."""
    _scores(vision_scores, len(problem.weights), 'vision scores')
    return {method: select_method(method, problem, vision_scores=vision_scores) for method in METHODS}