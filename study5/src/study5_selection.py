"""A/D Study4 baselines and the matched Study5 seed/completion ablation."""
from dataclasses import dataclass
import numpy as np

import coverpruner_coverage as coverage
from coverpruner_spatial_selection import select_method as study4_select

METHODS = ("A", "D", "G", "H", "I")
BUDGET = 64
SEEDS = 25


@dataclass(frozen=True)
class Selection:
    method: str
    seed_indices: tuple[int, ...]
    selection_order: tuple[int, ...]
    sorted_visual_indices: tuple[int, ...]
    sorted_sequence_indices: tuple[int, ...]
    seed_cell_counts: tuple[int, ...]
    final_cell_counts: tuple[int, ...]
    initial_coverage: float
    final_coverage: float
    greedy_objectives: tuple[float, ...]


def select_all(problem):
    """Share one prepared Study4 problem; diagnostics never influence selection."""
    a = study4_select("A", problem).greedy
    d = study4_select("D", problem).greedy
    ranking = tuple(int(j) for j in np.argsort(-problem.query_relevance, kind="stable"))
    g_seed = ranking[:SEEDS]
    g = coverage.global_greedy(problem.utilities, problem.weights, BUDGET,
                               seed_indices=g_seed, sequence_indices=problem.sequence_indices)
    reserved = set(d.seed_indices)
    i_order = d.seed_indices + tuple(j for j in ranking if j not in reserved)[:BUDGET-SEEDS]
    plans = (
        ("A", (), a.selection_order, a.objective_values),
        ("D", d.seed_indices, d.selection_order, d.objective_values),
        ("G", g_seed, g.selection_order, g.objective_values),
        ("H", (), ranking[:BUDGET], ()),
        ("I", d.seed_indices, i_order, ()),
    )
    owners = problem.layout.cell_ids
    positions = problem.sequence_indices
    def counts(indices):
        return tuple(int(v) for v in np.bincount(owners[list(indices)], minlength=25))
    results = {}
    for method, seed, order, trace in plans:
        if len(order)!=BUDGET or len(set(order))!=BUDGET or order[:len(seed)]!=seed:
            raise RuntimeError(f"{method}: expected a unique K64 with its seed prefix")
        sorted_ids = tuple(sorted(order,key=positions.__getitem__))
        initial = coverage.coverage_value(problem.utilities, problem.weights, seed)
        final = coverage.coverage_value(problem.utilities, problem.weights, order)
        if trace and (trace[0]!=initial or trace[-1]!=final):
            raise RuntimeError(f"{method}: coverage initialization or completion differs")
        results[method] = Selection(method,seed,order,sorted_ids,tuple(positions[j] for j in sorted_ids),
                                    counts(seed),counts(order),initial,final,trace)
    return results

