# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# Licensed under the Apache License, Version 2.0 (the License).
# You may not use this file except in compliance with the License.
# A copy is in the repository LICENSE and at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed to in writing, distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for permissions and limitations.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Compare Global Coverage and Random across four budgets and two spatial grids.

Uses cached SmolVLM files and five images. No generation is performed.
Coverage includes local/global views; spatial statistics include local only.
Sample standard deviation is used. CV is NaN if spatial mean is zero.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import statistics
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from examples.diagnose_global_coverage import (
    FILENAMES, MODEL_ID, RANDOM_SEEDS, coverage_affinity, coverage_score,
    local_selection_lattice, near_equal_edges, projected_grid_shape,
)
from src.pruning.coverage import global_coverage_select

BUDGETS = (256, 128, 64, 32)
GRID_SIDES = (4, 8)


def region_counts(lattice: torch.Tensor, grid_side: int) -> torch.Tensor:
    """Generalize the existing near-equal partition to grid_side x grid_side."""
    if grid_side < 1:
        raise ValueError("grid_side must be positive")
    row_edges = near_equal_edges(lattice.shape[0], grid_side)
    col_edges = near_equal_edges(lattice.shape[1], grid_side)
    counts = torch.zeros((grid_side, grid_side), dtype=torch.long)
    for row in range(grid_side):
        for col in range(grid_side):
            counts[row, col] = lattice[
                row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]
            ].sum()
    if counts.sum().item() != lattice.sum().item():
        raise RuntimeError("Spatial partition lost selected local tokens")
    return counts


@dataclass
class SpatialStats:
    zero: int
    minimum: int
    maximum: int
    mean: float
    std: float
    cv: float


def spatial_statistics(counts: torch.Tensor) -> SpatialStats:
    """Include all regions, including zeros; use the G-1 sample denominator."""
    values = counts.flatten().tolist()
    if len(values) < 2:
        raise ValueError("Sample spatial statistics require at least two regions")
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return SpatialStats(
        values.count(0), min(values), max(values), float(mean), std,
        std / mean if mean else float("nan"),
    )


@dataclass
class SelectionResult:
    label: str
    score: float
    local: int
    global_count: int
    spatial: dict[int, SpatialStats]


@dataclass
class BudgetResult:
    k: int
    selections: list[SelectionResult]
    random_mean: float
    random_std: float
    improvement: float
    improvement_pct: float


@torch.inference_mode()
def evaluate_budget(embeddings, affinity, rows, cols, token_grid, k, permutations):
    """Choose once per K/seed, then evaluate the same selection on both grids."""
    local_token_count = rows * cols * token_grid[0] * token_grid[1]
    indices_by_label = [("GC", global_coverage_select(embeddings, k))]
    for seed in RANDOM_SEEDS:
        # Same CPU randperm(seed) and final sort as run_random_pruning.py.
        indices = permutations[seed][:k].sort().values.to(embeddings.device)
        indices_by_label.append((f"Random {seed}", indices))
    selections = []
    for label, indices in indices_by_label:
        if indices.numel() != k or indices.unique().numel() != k:
            raise RuntimeError(f"{label} did not select exactly K unique tokens")
        local = (indices < local_token_count).sum().item()
        lattice = local_selection_lattice(indices, rows, cols, token_grid)
        if lattice.sum().item() != local:
            raise RuntimeError("Local/global reconstruction lost tokens")
        spatial = {}
        for side in GRID_SIDES:
            stats = spatial_statistics(region_counts(lattice, side))
            if stats.zero < max(0, side**2 - local):
                raise RuntimeError("Zero-region count violates the token-count lower bound")
            spatial[side] = stats
        selections.append(SelectionResult(
            label, coverage_score(affinity, indices), local, k - local, spatial,
        ))
    random_scores = [s.score for s in selections[1:]]
    random_mean = statistics.mean(random_scores)
    improvement = selections[0].score - random_mean
    return BudgetResult(
        k, selections, random_mean, statistics.stdev(random_scores), improvement,
        improvement / random_mean * 100 if random_mean else float("nan"),
    )


def print_table(headers, rows):
    data = [headers] + [tuple(str(value) for value in row) for row in rows]
    widths = [max(len(row[col]) for row in data) for col in range(len(headers))]
    for index, row in enumerate(data):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))


def print_budget(result):
    print(f"\nK={result.k}: GC F(S)={result.selections[0].score:.6f}")
    for selection in result.selections[1:]:
        print(f"{selection.label} F(S)={selection.score:.6f}")
    print(f"Random F mean +/- sample std: {result.random_mean:.6f} +/- {result.random_std:.6f}")
    print(f"F improvement: {result.improvement:+.6f} ({result.improvement_pct:+.4f}%)")
    for side in GRID_SIDES:
        feasible = side**2 <= result.k
        print(f"Grid {side}x{side}, G={side**2}, G/K={side**2 / result.k:.4f}, "
              f"minimum-one floor feasible (G<=K): {feasible}")
        rows = []
        for selection in result.selections:
            s = selection.spatial[side]
            rows.append((selection.label, f"{selection.local}/{selection.global_count}",
                         s.zero, s.minimum, s.maximum, f"{s.mean:.6f}",
                         f"{s.std:.6f}", f"{s.cv:.6f}"))
        print_table(("Selector", "Selected L/G", "Zero", "Min", "Max", "Mean", "Sample std", "CV"), rows)
    sys.stdout.flush()


def summary_row(filename, result, side):
    gc = result.selections[0]
    random = result.selections[1:]
    zero_mean = statistics.mean(s.spatial[side].zero for s in random)
    # Average per-seed CVs, not the CV of an averaged region matrix.
    cv_mean = statistics.mean(s.spatial[side].cv for s in random)
    return (
        filename, result.k, f"{side}x{side}", f"{side**2 / result.k:.4f}",
        f"{gc.score:.6f}", f"{result.random_mean:.6f}", f"{result.improvement_pct:+.4f}",
        gc.spatial[side].zero, f"{zero_mean:.4f}", f"{gc.spatial[side].cv:.6f}",
        f"{cv_mean:.6f}", str(side**2 <= result.k),
    )


@torch.inference_mode()
def diagnose_image(processor, model, path, device):
    with Image.open(path) as source:
        original_size = source.size
        image = source.convert("RGB")
    image_inputs = processor.image_processor(
        images=[image], return_row_col_info=True, return_tensors="pt"
    ).to(device)
    rows, cols = int(image_inputs["rows"][0][0]), int(image_inputs["cols"][0][0])
    projected = model.get_image_features(
        pixel_values=image_inputs["pixel_values"],
        pixel_attention_mask=image_inputs.get("pixel_attention_mask"), return_dict=True,
    ).pooler_output
    views, tokens_per_view, _ = projected.shape
    token_grid = projected_grid_shape(model.config, image_inputs["pixel_values"].shape, tokens_per_view)
    if views != rows * cols + 1 or views != image_inputs["pixel_values"].shape[1]:
        raise RuntimeError("Crop metadata disagrees with actual view count")
    embeddings = projected.flatten(0, 1)
    n = embeddings.shape[0]
    if max(BUDGETS) > n:
        raise RuntimeError(f"{path.name}: N={n} cannot support K={max(BUDGETS)}")
    affinity = coverage_affinity(embeddings)
    permutations = {
        seed: torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))
        for seed in RANDOM_SEEDS
    }
    print(f"\nImage: {path.name}; size(WxH): {original_size[0]}x{original_size[1]}")
    print(f"Crop: {rows}x{cols}; views: {views}={rows * cols}+1; tokens/view: {tokens_per_view} (8x8 verified)")
    print(f"N: {n}; local/global tokens: {rows * cols * tokens_per_view}/{tokens_per_view}; "
          f"lattice: {rows * token_grid[0]}x{cols * token_grid[1]}", flush=True)
    summary = []
    for k in BUDGETS:
        result = evaluate_budget(embeddings, affinity, rows, cols, token_grid, k, permutations)
        print_budget(result)
        summary.extend(summary_row(path.name, result, side) for side in GRID_SIDES)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "images")
    args = parser.parse_args()
    paths = [args.image_dir / name for name in FILENAMES]
    for path in paths:
        if not path.is_file():
            parser.error(f"Image not found: {path}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    print(f"Device: {device}; model/processor loaded once; budgets={BUDGETS}; grids={GRID_SIDES}")
    print("Spatial statistics use all G local regions, including zeros. CV=NaN if mean=0.")
    print("Floor feasibility is the budget condition G<=K, not a guarantee of occupied regions.", flush=True)
    summary = []
    for path in paths:
        summary.extend(diagnose_image(processor, model, path, device))
    print("\nSummary: Random zero mean and CV mean are averages over seeds 0..4.")
    print_table(("image", "K", "grid", "G/K", "GC F(S)", "Random F mean", "F improvement %",
                 "GC zero regions", "Random zero mean", "GC CV", "Random CV mean", "floor feasible(G<=K)"), summary)


if __name__ == "__main__":
    main()
