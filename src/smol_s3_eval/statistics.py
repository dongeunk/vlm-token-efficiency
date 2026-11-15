# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# See LICENSE for terms. Publication adds comments only; experimental body unchanged.

"""Paired image-cluster bootstrap: sum confusion counts, recompute micro metrics."""
from __future__ import annotations

import numpy as np

from .lock import METHODS
from .scoring import COUNT_FIELDS

METRICS = ("pope_f1", "pope_accuracy", "invalid_rate", "strict_accuracy")


def metrics_from_counts(counts):
    c = np.asarray(counts, dtype=np.float64)
    if c.shape[-1] != len(COUNT_FIELDS) or not np.isfinite(c).all() or (c < 0).any() or (c != np.floor(c)).any():
        raise ValueError("nonnegative finite integer confusion counts required")
    tp, fp, fn, tn, iy, ino, strict, strict_invalid = np.moveaxis(c, -1, 0)
    n = tp+fp+fn+tn+iy+ino
    if (n <= 0).any() or (strict+strict_invalid > n).any():
        raise ValueError("empty or inconsistent strict count support")
    den = 2*tp+fp+fn+iy
    f1 = np.divide(2*tp, den, out=np.zeros_like(den), where=den != 0)
    return np.stack((f1, (tp+tn)/n, (iy+ino)/n, strict/n), axis=-1)


def paired_bootstrap(per_image, resamples=10000, seed=42, draws=None):
    """All methods reuse every sampled image cluster and all6 questions.

    draws is for independent scalar-oracle tests; production uses PCG64(42).
    Output rates/CI are fractions; multiply by100 for percentage-point display.
    """
    rows = sorted(per_image, key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)) or any(r["questions"] != 6 for r in rows):
        raise ValueError("distinct complete image clusters required")
    if type(seed) is not int or type(resamples) is not int or resamples < 1:
        raise ValueError("integer seed/resamples required")
    counts = np.asarray([[r["counts"][m] for m in METHODS] for r in rows], dtype=np.float64)
    if counts.shape != (len(rows), len(METHODS), len(COUNT_FIELDS)):
        raise ValueError("count matrix shape invalid")
    metrics_from_counts(counts)
    if not np.all(counts[:, :, :6].sum(axis=-1) == 6):
        raise ValueError("cluster support must be six for every method")
    if not np.all(counts[:,:,[0,2,4]].sum(axis=-1) == 3) or not np.all(counts[:,:,[1,3,5]].sum(axis=-1) == 3):
        raise ValueError("paired image label support must remain yes3/no3")
    if draws is None:
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(rows), size=(resamples, len(rows)))
    else:
        indices = np.asarray(draws)
        if (indices.ndim != 2 or indices.shape[1] != len(rows) or indices.shape[0] == 0
                or indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(rows)).any()):
            raise ValueError("invalid oracle resampling indices")
    point = metrics_from_counts(counts.sum(axis=0))
    # Chunking bounds memory, especially for 10,000 x 200 x 4 x 8.
    samples = np.empty((len(indices), len(METHODS), len(METRICS)), dtype=np.float64)
    for start in range(0, len(indices), 128):
        end = min(start+128, len(indices))
        samples[start:end] = metrics_from_counts(counts[indices[start:end]].sum(axis=1))
    comparisons = {}
    for a, b in (("s2_c", "s2_l"), ("s2_c", "s2_u"), ("s2_l", "s2_u"),
                 ("s2_c", "full"), ("s2_l", "full"), ("s2_u", "full")):
        ia, ib = METHODS.index(a), METHODS.index(b)
        diff = samples[:, ia, :] - samples[:, ib, :]
        ci = np.quantile(diff, [0.025, 0.975], axis=0, method="linear")
        comparisons[f"{a}-{b}"] = {metric: dict(estimate=float(point[ia,j]-point[ib,j]),
                                               ci95=[float(ci[0,j]),float(ci[1,j])],
                                               primary=(a,b,metric) == ("s2_c","s2_l","pope_f1"))
                                       for j, metric in enumerate(METRICS)}
    return dict(images=len(rows), questions_per_image=6, resamples=len(indices), seed=seed,
                rng="PCG64" if draws is None else "explicit oracle draws", image_ids=ids,
                confidence=0.95, interval="percentile, linear quantiles", units="fractions (x100 = percentage points)",
                count_fields=list(COUNT_FIELDS), primary="s2_c-s2_l / pope_f1", comparisons=comparisons,
                inferential_scope="paired six-question image clusters of locked subset; no full-POPE claim")
