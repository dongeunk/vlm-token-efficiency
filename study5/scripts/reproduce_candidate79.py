"""Reproduce Candidate79 scores and bootstrap CIs from anonymous image counts."""
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
METHODS = "ADGHI"
COUNTS = ("tp", "fp", "tn", "fn", "invalid_yes", "invalid_no")
SEED = 20261126
REPETITIONS = 10_000


def measure(vectors):
    tp, fp, tn, fn, invalid_yes, invalid_no = np.moveaxis(vectors, -1, 0)

    def rate(top, bottom):
        return np.divide(top, bottom, out=np.zeros_like(top, dtype=float), where=bottom != 0)

    return {
        "accuracy": rate(tp + tn, vectors.sum(axis=-1)),
        "f1": rate(2 * tp, 2 * tp + fp + fn + invalid_yes),
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn + invalid_yes),
        "no_recall": rate(tn, tn + fp + invalid_no),
    }


def verify():
    counts = np.full((79, 5, 6), -1, dtype=np.int64)
    with (RESULTS / "image_cluster_counts.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            cluster = int(row["cluster"]) - 1
            method = METHODS.index(row["method"])
            if not 0 <= cluster < 79 or counts[cluster, method, 0] != -1:
                raise ValueError("Invalid or duplicate image cluster")
            counts[cluster, method] = [int(row[key]) for key in COUNTS]
    if (counts < 0).any() or not (counts.sum(axis=-1) == 6).all():
        raise ValueError("Missing or invalid image counts")
    if not ((counts[:, :, 0] + counts[:, :, 3] + counts[:, :, 4]) == 3).all():
        raise ValueError("Unexpected label balance")

    packet = json.loads((RESULTS / "candidate_summary.json").read_text(encoding="utf-8"))
    summary = packet["payload"]
    fingerprints = json.loads((RESULTS / "PROVENANCE.json").read_text(encoding="utf-8"))
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != packet["sha256"] or (
        fingerprints["public_summary_payload_sha256"] != packet["sha256"]
    ):
        raise ValueError("Summary checksum mismatch")
    if (summary["independent_test"], summary["bootstrap_seed"], summary["repetitions"]) != (
        False, SEED, REPETITIONS
    ):
        raise ValueError("Unexpected analysis settings")

    totals = counts.sum(axis=0)
    scores = measure(totals)
    with (RESULTS / "performance.csv").open(encoding="utf-8", newline="") as stream:
        perf = {row.pop("method"): {k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)}
    for i, method in enumerate(METHODS):
        expected = summary["strict"][method]
        calculated = {k: int(totals[i, j]) for j, k in enumerate(COUNTS)}
        calculated.update({k: float(v[i]) for k, v in scores.items()})
        # The displayed per-method F1 uses the harmonic mean in the original scorer.
        # Bootstrap contrasts use the equivalent count formula in the original analysis.
        precision, recall = calculated["precision"], calculated["recall"]
        calculated["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        calculated.update(n=int(totals[i].sum()), invalid_rate=float(totals[i, 4:].sum() / totals[i].sum()))
        if calculated != expected or expected != perf[method] or expected != summary["official"][method]:
            raise ValueError(f"Metric mismatch for {method}")

    rng = random.Random(SEED)
    draws = np.array([
        np.bincount([rng.randrange(79) for _ in range(79)], minlength=79)
        for _ in range(REPETITIONS)
    ])
    scores_sample = measure((draws @ counts.reshape(79, -1)).reshape(REPETITIONS, 5, 6))
    checks = {}
    with (RESULTS / "cluster_bootstrap.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["contrast"], row["metric"])
            if key in checks:
                raise ValueError("Duplicate bootstrap comparison")
            checks[key] = row
    for contrast, entry in summary["comparisons"].items():
        left, right = (METHODS.index(arm) for arm in contrast.split("-"))
        for metric, expected in entry["metrics"].items():
            result = scores[metric][left] - scores[metric][right]
            interval = np.quantile(scores_sample[metric][:, left] - scores_sample[metric][:, right],
                                   [.025, .975], method="linear")
            row = checks.pop((contrast, metric))
            if result != expected["difference"] or interval.tolist() != expected["ci95"] or (
                [result, *interval] != [float(row[key]) for key in ("difference", "lower95", "upper95")]
            ):
                raise ValueError(f"Bootstrap mismatch for {contrast} {metric}")
            for key in ("correct_to_wrong", "wrong_to_correct"):
                if int(row[key]) != entry[key]:
                    raise ValueError(f"Exported transition total mismatch for {contrast}")
    if checks:
        raise ValueError("Unexpected bootstrap rows")
    print("PASS: five methods, 79 image clusters, 10,000 draws, 35 paired estimates and intervals.")
    print("Paired transition totals match across published exports.")


if __name__ == "__main__":
    verify()
