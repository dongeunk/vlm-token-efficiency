# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# See LICENSE for terms. Publication adds comments only; experimental body unchanged.

"""Frozen parser adapters, all-item metrics and complete paired record validation."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
import statistics

from src.evaluation.pope import compute_metrics, parse_yes_no
from src.evaluation.semantic_yes_no import explain_semantic_yes_no, PARSER_VERSION
from .contracts import validate_runtime
from .lock import METHODS, REVISION

COUNT_FIELDS = ("tp", "fp", "fn", "tn", "invalid_yes", "invalid_no", "strict_correct", "strict_invalid")


class IncompleteEvaluation(ValueError):
    """An invalid run must not become a selectively complete evaluation."""


def parse_response(raw_text):
    """Only raw text enters either parser; labels/questions never do."""
    decision = explain_semantic_yes_no(raw_text)
    return dict(semantic_answer=decision.answer, strict_answer=parse_yes_no(raw_text),
                semantic_rule_id=decision.rule_id, semantic_evidence=decision.evidence,
                semantic_reason=decision.reason, parser_version=PARSER_VERSION)


def metric_bundle(rows):
    """Semantic POPE-style counts; this is not the original permissive POPE parser."""
    rows = list(rows)
    semantic = compute_metrics([dict(label=r["label"], parsed_answer=r["semantic_answer"]) for r in rows])
    strict = compute_metrics([dict(label=r["label"], parsed_answer=r["strict_answer"]) for r in rows])
    return dict(n=semantic["n"], pope_accuracy=semantic["accuracy"], semantic_accuracy=semantic["accuracy"],
                pope_precision=semantic["precision"], pope_recall=semantic["recall"], pope_f1=semantic["f1"],
                strict_accuracy=strict["accuracy"], invalid_count=semantic["invalid_yes"]+semantic["invalid_no"],
                invalid_rate=semantic["invalid_ratio"], strict_invalid_count=strict["invalid_yes"]+strict["invalid_no"],
                semantic_confusion=semantic, strict_confusion=strict, original_official_parser=False,
                primary_parser=PARSER_VERSION)


def count_vector(rows):
    rows = list(rows)
    m = metric_bundle(rows)
    c = m["semantic_confusion"]
    return [c[k] for k in COUNT_FIELDS[:6]] + [sum(r["strict_answer"] == r["label"] for r in rows), m["strict_invalid_count"]]


def token_overlap(a, b, n_local):
    a, b = set(a), set(b)
    intersection, union = a & b, a | b
    al, bl = {x for x in a if x < n_local}, {x for x in b if x < n_local}
    ag, bg = a-al, b-bl
    return dict(intersection=len(intersection), union=len(union), jaccard=len(intersection)/len(union) if union else 0.0,
                local_intersection=len(al & bl), global_intersection=len(ag & bg),
                same_local=(al == bl), same_global=(ag == bg))


def measurement_summary(values):
    values = sorted(values)
    if not values:
        raise ValueError("empty measurements")
    position = (len(values)-1)*0.95
    low = math.floor(position)
    high = math.ceil(position)
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p95=values[low]+(values[high]-values[low])*(position-low), minimum=values[0], maximum=values[-1])


def score_complete(records, questions, protocol_sha256, selection_lock_sha256):
    """Fail on missing/duplicate/foreign/error runs. Never complete-case filter.

    Empty successful text is a parser-invalid prediction, not a runtime error.
    Raw output IDs and all runtime evidence remain in the caller's JSONL.
    """
    questions = list(questions)
    question_map = {(q["image_id"], q["question_id"]): q for q in questions}
    if len(question_map) != len(questions) or len({q["question_id"] for q in questions}) != len(questions) or not questions:
        raise IncompleteEvaluation("empty/duplicate manifest")
    splits = {q["split"] for q in questions}
    if len(splits) != 1:
        raise IncompleteEvaluation("pilot and main must be scored separately")
    split = next(iter(splits))
    image_questions = defaultdict(list)
    for q in questions:
        image_questions[q["image_id"]].append(q)
    for group in image_questions.values():
        if len(group) != 6 or sum(q["label"] == "yes" for q in group) != 3 or sum(q["label"] == "no" for q in group) != 3:
            raise IncompleteEvaluation("each image must have all six official questions (3yes/3no)")
    expected = {(i, q, m) for i, q in question_map for m in METHODS}
    indexed, scored = {}, {}
    for row in records:
        try:
            if type(row["image_id"]) is not int or type(row["question_id"]) is not int:
                raise IncompleteEvaluation("image/question IDs must be integers")
            key = row["image_id"], row["question_id"], row["method"]
            if key not in expected or key in indexed:
                raise IncompleteEvaluation("foreign/duplicate run key")
            q = question_map[key[:2]]
            if row["split"] != split or row["status"] != "ok":
                raise IncompleteEvaluation("wrong split or unsuccessful runtime (cannot silently drop)")
            if row["protocol_sha256"] != protocol_sha256 or row["selection_lock_sha256"] != selection_lock_sha256:
                raise IncompleteEvaluation("configuration/cohort lock mismatch")
            if (row["model_revision"] != REVISION or row["image_sha256"] != q["image_sha256"]
                    or row["prompt_sha256"] != q["prompt_sha256"] or row["label"] != q["label"]
                    or row["source_record_sha256"] != q["source_record_sha256"]):
                raise IncompleteEvaluation("checkpoint, input or ground truth mismatch")
            if row["schema_version"] != "smol_s3_run_v1" or not isinstance(row["raw_text"], str):
                raise IncompleteEvaluation("invalid record schema/raw text type")
            for field in ("input_ids_sha256", "pixel_values_sha256"):
                digest=row[field]
                if not isinstance(digest,str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise IncompleteEvaluation("input tensor SHA must be lower-case SHA256")
            diagnostics = validate_runtime(row)
            parsed = parse_response(row["raw_text"])
            scored[key] = dict(split=split, image_id=key[0], question_id=key[1], method=key[2], label=q["label"],
                               raw_text=row["raw_text"], **parsed, generation=diagnostics,
                               correct=parsed["semantic_answer"] == q["label"],
                               strict_correct=parsed["strict_answer"] == q["label"])
            indexed[key] = row
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IncompleteEvaluation):
                raise
            raise IncompleteEvaluation(f"malformed run: {error}") from error
    if set(indexed) != expected:
        raise IncompleteEvaluation(f"incomplete paired run: missing {len(expected-set(indexed))} records")
    overlaps, paired_questions = [], []
    for key, q in sorted(question_map.items()):
        runs = {m: indexed[(*key, m)] for m in METHODS}
        full_seq = runs["full"]["sequence"]
        for m in METHODS[1:]:
            for field in ("input_visual_positions", "nonvisual_positions", "original_sequence_length", "rows", "cols"):
                if runs[m]["sequence"][field] != full_seq[field]:
                    raise IncompleteEvaluation("paired methods did not receive identical original sequences/layouts")
            for field in ("input_ids_sha256", "pixel_values_sha256"):
                if runs[m][field] != runs["full"][field]:
                    raise IncompleteEvaluation("paired token IDs or preprocessed pixels differ")
        global_sets = [{x for x in runs[m]["sequence"]["selected_visual_indices"] if x >= full_seq["local_tokens"]} for m in METHODS[1:]]
        if any(s != global_sets[0] for s in global_sets):
            raise IncompleteEvaluation("global16 selection differs across L/U/C")
        paired_questions.append(dict(split=split, image_id=key[0], question_id=key[1], label=q["label"],
                                     methods={m: scored[(*key, m)] for m in METHODS}))
        for a, b in combinations(METHODS, 2):
            overlaps.append(dict(split=split, image_id=key[0], question_id=key[1], method_a=a, method_b=b,
                                 **token_overlap(runs[a]["sequence"]["selected_visual_indices"],
                                                 runs[b]["sequence"]["selected_visual_indices"], full_seq["local_tokens"])))
    summary = {}
    for m in METHODS:
        rows = [scored[(*key, m)] for key in sorted(question_map)]
        summary[m] = metric_bundle(rows)
        summary[m]["eos_count"] = sum(r["generation"]["eos_generated"] for r in rows)
        summary[m]["possible_truncation_count"] = sum(r["generation"]["possible_truncation"] for r in rows)
        summary[m]["measurement"] = {field: measurement_summary([indexed[(*key,m)]["measurement"][field] for key in sorted(question_map)])
                                      for field in ("ttft_e2e_ms", "ttft_decoder_ms", "selection_overhead_ms", "peak_allocated_mib", "peak_reserved_mib")}
        summary[m]["mean_region_budgets"] = ([statistics.fmean(indexed[(*key,m)]["local_budgets"][i] for key in sorted(question_map)) for i in range(16)]
                                                if m != "full" else None)
    per_image = []
    for image_id in sorted(image_questions):
        per_image.append(dict(image_id=image_id, questions=6,
                              counts={m: count_vector([scored[(image_id,q["question_id"],m)] for q in image_questions[image_id]])
                                      for m in METHODS}))
    return dict(complete=True, split=split, images=len(image_questions), questions=len(question_map),
                records=len(indexed), count_fields=list(COUNT_FIELDS), summary=summary,
                per_question=paired_questions, per_image=per_image, token_overlap=overlaps,
                scope="locked POPE Random subset, frozen semantic parser; not official full POPE benchmark")
