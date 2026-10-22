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

"""Plot saved POPE spatial-allocation diagnostics without model inference.

Usage:
    python examples/plot_pope_spatial_heterogeneity.py --analysis analysis.json --output-dir figures

Input schema is the analysis JSON containing rows, groups, method_order and
metadata.  Every row describes one image/K setting, its raw/normalized local
region complexity and the five saved methods' local region-count matrices.
The original photograph is displayed unchanged, with no object/token overlays.
Region heatmaps occupy independent local-lattice coordinate axes; global-view
tokens are excluded and reported only as counts.  Twelve failure-case panels
and three aggregate PNG figures are written to fresh output paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("random", "pure_gc", "ours", "a_local_only", "b_laplacian_quota")
METHOD_LABELS = {
    "random": "Random",
    "pure_gc": "Pure Global Coverage",
    "ours": "Ours: spatial floor + GC",
    "a_local_only": "A: local-only floor + GC",
    "b_laplacian_quota": "B: Laplacian quota + GC",
}
METHOD_COLORS = {
    "random": "#87939f", "pure_gc": "#3579ad", "ours": "#d47c27",
    "a_local_only": "#498567", "b_laplacian_quota": "#8c66a1",
}
FAILURE_QIDS = (249, 527, 388, 1603, 2997, 1799)
BUDGETS = (64, 128)
GROUPS = ("Low", "Medium", "High")


def _plot_libraries():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    return plt, np, Image


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_analysis(analysis, np):
    if not isinstance(analysis, dict) or not isinstance(analysis.get("rows"), list):
        raise ValueError("analysis must contain a rows list")
    rows = analysis["rows"]
    if not rows:
        raise ValueError("analysis rows cannot be empty")
    order = tuple(analysis.get("method_order", METHODS))
    if len(order) != 5 or set(order) != set(METHODS):
        raise ValueError("method_order must contain the five expected methods exactly once")
    indexed = {}
    for row in rows:
        qid = _nonnegative_integer(row["question_id"], "question_id")
        k = _nonnegative_integer(row["k"], "k")
        if k not in BUDGETS or (qid, k) in indexed:
            raise ValueError("Rows must have unique (question_id, K) keys and K=64/128")
        grid = tuple(row["region_grid"])
        token_grid = tuple(row["token_grid"])
        if len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid):
            raise ValueError("region_grid must contain two positive integers")
        if len(token_grid) != 2 or any(type(v) is not int or v < 1 for v in token_grid):
            raise ValueError("token_grid must contain two positive integers")
        cr = _nonnegative_integer(row["crop_rows"], "crop_rows")
        cc = _nonnegative_integer(row["crop_cols"], "crop_cols")
        if cr < 1 or cc < 1:
            raise ValueError("Local crop rows and columns must be positive")
        local_population = cr * cc * token_grid[0] * token_grid[1]
        global_population = token_grid[0] * token_grid[1]
        if row["n"] != local_population + global_population:
            raise ValueError("Saved N does not match the local/global population geometry")
        if row["group"] not in GROUPS:
            raise ValueError("Unknown heterogeneity group")
        for field in ("raw_complexity", "normalized_complexity"):
            values = np.asarray(row["metrics"][field], dtype=np.float64)
            if values.shape != grid or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{field} must be a finite nonnegative region matrix")
            if field == "normalized_complexity" and (values > 1 + 1e-6).any():
                raise ValueError("Normalized complexity exceeds 1")
        h = row["metrics"]["heterogeneity_raw"]
        if not np.isfinite(h) or h < 0:
            raise ValueError("heterogeneity_raw must be finite and nonnegative")
        if set(row["methods"]) != set(METHODS):
            raise ValueError("Every row must contain exactly five method records")
        for method in METHODS:
            saved = row["methods"][method]
            values = np.asarray(saved["region_counts"])
            if (values.shape != grid or not np.issubdtype(values.dtype, np.integer)
                    or (values < 0).any()):
                raise ValueError("Method region_counts must be a nonnegative integer matrix")
            local = _nonnegative_integer(saved["selected_local"], "selected_local")
            glob = _nonnegative_integer(saved["selected_global"], "selected_global")
            if int(values.sum()) != local or local + glob != k:
                raise ValueError("Region counts must sum to local selection only; local+global must equal K")
            if local > local_population or glob > global_population:
                raise ValueError("Selected local/global count exceeds its population")
            if saved["zero_regions"] != int((values == 0).sum()):
                raise ValueError("Saved zero-region count disagrees with region matrix")
            if type(saved["semantic_correct"]) is not bool:
                raise ValueError("semantic_correct must be a boolean")
        indexed[qid, k] = row
    missing = [(qid, k) for qid in FAILURE_QIDS for k in BUDGETS if (qid, k) not in indexed]
    if missing:
        raise ValueError(f"Failure-case panel rows are missing: {missing}")

    grouped = analysis.get("groups")
    if not isinstance(grouped, list):
        raise ValueError("analysis must contain a groups list")
    group_index = {}
    for group in grouped:
        key = group["k"], group["group"], group["method"]
        if key in group_index or key[0] not in BUDGETS or key[1] not in GROUPS or key[2] not in METHODS:
            raise ValueError("Unexpected or duplicate aggregate group key")
        matching = [row for row in rows if row["k"] == key[0] and row["group"] == key[1]]
        n = len(matching)
        correct = sum(row["methods"][key[2]]["semantic_correct"] for row in matching)
        expected = correct / n if n else 0.0
        if (group["n"] != n or group["semantic_correct"] != correct
                or abs(float(group["semantic_accuracy"]) - expected) > 1e-12):
            raise ValueError("Aggregate accuracy differs from the saved per-image decisions")
        group_index[key] = group
    if len(group_index) != len(BUDGETS) * len(GROUPS) * len(METHODS):
        raise ValueError("Expected all 30 K/group/method aggregate records")
    return rows, indexed, group_index, order


def _region_axes(ax, shape):
    height, width = shape
    ax.set_xticks(range(width), [str(i) for i in range(width)])
    ax.set_yticks(range(height), [str(i) for i in range(height)])
    ax.set_xlabel("Local region column", fontsize=8)
    ax.set_ylabel("Local region row", fontsize=8)
    ax.tick_params(labelsize=7, length=0)
    ax.set_xticks([i - 0.5 for i in range(width + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(height + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _annotate_counts(ax, matrix, maximum):
    font = 9 if matrix.shape[0] <= 4 else 7
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = int(matrix[y, x])
            color = "white" if maximum > 0 and value / maximum > 0.58 else "#18242e"
            ax.text(x, y, str(value), ha="center", va="center", fontsize=font,
                    color=color, fontweight="bold" if value == 0 else "normal")


def _save(fig, path, plt):
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _failure_panel(row, path, order, plt, np, Image):
    image_path = Path(row["image_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    grid = tuple(row["region_grid"])
    th, tw = row["token_grid"]
    lattice = row["crop_rows"] * th, row["crop_cols"] * tw
    raw = np.asarray(row["metrics"]["raw_complexity"], dtype=np.float64)
    normalized = np.asarray(row["metrics"]["normalized_complexity"], dtype=np.float64)
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 9.8), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, hspace=0.12)
    flat = list(axes.flat)
    flat[0].imshow(image)
    flat[0].set_title(f"Original image (no overlay)\n{image.width} x {image.height} px", fontsize=10)
    flat[0].set_axis_off()

    for ax, values, title, maximum in ((flat[1], raw, "Raw Laplacian variance", None),
                                       (flat[2], normalized, "Min-max normalized complexity", 1.0)):
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=maximum, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        _region_axes(ax, grid)
        fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02).ax.tick_params(labelsize=7)
    count_max = max(int(np.asarray(row["methods"][method]["region_counts"]).max()) for method in order)
    for ax, method in zip(flat[3:], order):
        saved = row["methods"][method]
        values = np.asarray(saved["region_counts"])
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(1, count_max), interpolation="nearest")
        outcome = "correct" if saved["semantic_correct"] else "incorrect/invalid"
        prediction = saved.get("semantic_parsed_answer") or "invalid"
        ax.set_title(f"{METHOD_LABELS[method]}\nL/G={saved['selected_local']}/{saved['selected_global']}; "
                     f"zero={saved['zero_regions']}; {prediction}: {outcome}", fontsize=9)
        _region_axes(ax, grid)
        _annotate_counts(ax, values, max(1, count_max))
        fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02).ax.tick_params(labelsize=7)
    # Bottom-row labels already identify these shared region coordinates.
    for ax in flat[:4]:
        ax.set_xlabel("")
    fig.suptitle(f"QID {row['question_id']} | {row['image']} | GT={row['label']} | K={row['k']}\n"
                 f"Crop layout {row['crop_rows']} x {row['crop_cols']}; local lattice {lattice[0]} x {lattice[1]}; "
                 f"regions {grid[0]} x {grid[1]}; H(raw)={row['metrics']['heterogeneity_raw']:.4f} ({row['group']})",
                 fontsize=12)
    fig.supxlabel("Independent local-lattice heatmaps; global-view tokens excluded. "
                  "Count colors share one scale within this figure. Image/region mapping is a normalized-coordinate proxy.",
                  fontsize=8)
    _save(fig, path, plt)


def _heterogeneity_scatter(rows, path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharey=True, layout="constrained")
    colors = {-1: "#aa4242", 0: "#80909e", 1: "#3c8660"}
    for ax, k in zip(axes, BUDGETS):
        selected = [row for row in rows if row["k"] == k]
        for delta in (-1, 0, 1):
            subset = [row for row in selected if int(row["methods"]["ours"]["semantic_correct"])
                      - int(row["methods"]["pure_gc"]["semantic_correct"]) == delta]
            ax.scatter([r["metrics"]["heterogeneity_raw"] for r in subset], [delta] * len(subset),
                       color=colors[delta], s=46, alpha=0.78, edgecolors="white", linewidth=0.5,
                       label={-1: "Ours wrong / GC correct", 0: "Same correctness", 1: "Ours correct / GC wrong"}[delta])
        ax.axhline(0, color="#bbc2c8", linewidth=0.8, zorder=0)
        ax.set_title(f"K={k}; n={len(selected)}")
        ax.set_xlabel("Raw-complexity heterogeneity H")
        ax.set_yticks([-1, 0, 1], ["-1", "0", "+1"])
        ax.set_ylim(-1.35, 1.35)
        ax.grid(axis="x", color="#e4e7eb", linewidth=0.6)
    axes[0].set_ylabel("Ours correct - Pure GC correct (semantic)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Per-question correctness difference versus spatial heterogeneity", fontsize=12)
    _save(fig, path, plt)


def _group_accuracy(group_index, path, order, plt, np):
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.1), sharey=True, layout="constrained")
    x = np.arange(len(GROUPS), dtype=np.float64)
    width = 0.15
    for ax, k in zip(axes, BUDGETS):
        for position, method in enumerate(order):
            records = [group_index[k, group, method] for group in GROUPS]
            accuracy = [100 * entry["semantic_accuracy"] for entry in records]
            offset = (position - (len(order) - 1) / 2) * width
            bars = ax.bar(x + offset, accuracy, width=width, color=METHOD_COLORS[method],
                          label=METHOD_LABELS[method], zorder=3)
            ax.bar_label(bars, labels=[f"{entry['semantic_correct']}/{entry['n']}" for entry in records],
                         padding=2, fontsize=6, rotation=90)
        group_counts = [group_index[k, group, order[0]]["n"] for group in GROUPS]
        ax.set_xticks(x, [f"{group}\n(n={n})" for group, n in zip(GROUPS, group_counts)])
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(f"K={k}")
        ax.set_xlabel("Within-K raw-H rank tertile")
        ax.grid(axis="y", color="#e4e7eb", zorder=0)
    axes[0].set_ylabel("Semantic accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Accuracy by heterogeneity group (10/10/10 rank split; no correctness-based grouping)", fontsize=11)
    _save(fig, path, plt)


def _gc_zero_scatter(rows, path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), layout="constrained")
    group_colors = {"Low": "#377fa5", "Medium": "#b18137", "High": "#92579c"}
    for ax, k in zip(axes, BUDGETS):
        subset = [row for row in rows if row["k"] == k]
        for group in GROUPS:
            selected = [row for row in subset if row["group"] == group]
            ax.scatter([r["metrics"]["heterogeneity_raw"] for r in selected],
                       [r["methods"]["pure_gc"]["zero_regions"] for r in selected],
                       color=group_colors[group], s=45, alpha=0.8, edgecolors="white", linewidth=0.5,
                       label=group)
        ax.set_title(f"Pure GC | K={k}; {subset[0]['region_grid'][0]} x {subset[0]['region_grid'][1]} regions")
        ax.set_xlabel("Raw-complexity heterogeneity H")
        ax.set_ylabel("Zero-token local regions")
        ax.set_ylim(bottom=-0.35)
        ax.grid(color="#e4e7eb", linewidth=0.6, zorder=0)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Pure Global Coverage: local spatial gaps versus heterogeneity", fontsize=12)
    _save(fig, path, plt)


def plot_analysis(analysis, output_dir):
    """Create 12 image/K panels and three aggregate figures from saved results.

    Returns the 15 created Path objects.  Every target path is checked before
    writing; existing figures are never overwritten.  H/group definitions are
    consumed from the analysis JSON, not recomputed or tuned using outcomes.
    """
    plt, np, Image = _plot_libraries()
    rows, indexed, group_index, order = _validate_analysis(analysis, np)
    output_dir = Path(output_dir).resolve()
    paths = [output_dir / "panels" / f"qid_{qid}_k_{k}.png" for qid in FAILURE_QIDS for k in BUDGETS]
    paths.extend(output_dir / name for name in ("heterogeneity_vs_ours_minus_gc.png",
                                               "accuracy_by_heterogeneity_group.png",
                                               "gc_zero_regions_vs_heterogeneity.png"))
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Existing figures are preserved: {existing}")
    for qid in FAILURE_QIDS:
        for k in BUDGETS:
            if not Path(indexed[qid, k]["image_path"]).is_file():
                raise FileNotFoundError(indexed[qid, k]["image_path"])
    (output_dir / "panels").mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"}):
        for path, (qid, k) in zip(paths[:12], ((qid, k) for qid in FAILURE_QIDS for k in BUDGETS)):
            _failure_panel(indexed[qid, k], path, order, plt, np, Image)
        _heterogeneity_scatter(rows, paths[12], plt)
        _group_accuracy(group_index, paths[13], order, plt, np)
        _gc_zero_scatter(rows, paths[14], plt)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    paths = plot_analysis(analysis, args.output_dir)
    print(json.dumps({"analysis": str(args.analysis.resolve()), "figure_count": len(paths),
                      "figures": [str(path) for path in paths], "model_inference_executed": False,
                      "image_token_overlays": False, "global_tokens_in_local_heatmaps": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
