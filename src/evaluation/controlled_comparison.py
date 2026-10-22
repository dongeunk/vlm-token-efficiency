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

"""CPU-only scoring of saved, matched POPE pruning comparison records.

No inference, file writes, image loading, Torch, or network access occurs here.
Both parsers are reused unchanged.  All raw responses are parsed before labels
are joined for scoring; stored decisions are checked rather than trusted.
An explicitly scoped smoke subset is never reported as the 30-question pilot.
"""

from collections import Counter
import re

from src.evaluation.pope import compute_metrics, parse_yes_no
from src.evaluation.semantic_yes_no import PARSER_VERSION, explain_semantic_yes_no


DEFAULT_METHODS = ("random", "pure_gc", "ours", "a_local_only", "b_laplacian_quota")
DEFAULT_PAIRS = (("a_local_only", "b_laplacian_quota"),
                 ("ours", "a_local_only"), ("ours", "pure_gc"))
_IMAGE_NAME = re.compile(r"COCO_val2014_(\d{12})\.jpg\Z", re.ASCII)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(record, context):
    if not isinstance(record, dict):
        raise ValueError(f"{context}: expected a record object")
    required = ("question_id", "image_id", "image", "label")
    if any(key not in record for key in required):
        raise ValueError(f"{context}: missing identity fields")
    question_id = _integer(record["question_id"], f"{context}: question_id")
    image_id = _integer(record["image_id"], f"{context}: image_id")
    image = record["image"]
    match = _IMAGE_NAME.fullmatch(image) if isinstance(image, str) else None
    if match is None or int(match.group(1)) != image_id:
        raise ValueError(f"{context}: image filename/image_id disagree")
    question = record.get("text", record.get("question"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{context}: nonempty text/question is required")
    if "text" in record and "question" in record and record["text"] != record["question"]:
        raise ValueError(f"{context}: text and question disagree")
    if record["label"] not in ("yes", "no"):
        raise ValueError(f"{context}: label must be yes or no")
    return question_id, image_id, image, question, record["label"]


def _indexed_records(records, context):
    records = list(records)
    identities, indexed = [], {}
    for position, record in enumerate(records):
        identity = _identity(record, f"{context}[{position}]")
        if identity[0] in indexed:
            raise ValueError(f"{context}: duplicate question_id {identity[0]}")
        indexed[identity[0]] = record
        identities.append(identity)
    return records, identities, indexed


def _saved_output(record, context):
    if not isinstance(record.get("raw_output"), str):
        raise ValueError(f"{context}: completed raw_output must be a string")
    if record.get("error") is not None:
        raise ValueError(f"{context}: errored generation is not an evaluated response")
    if "generation_success" in record and record["generation_success"] is not True:
        raise ValueError(f"{context}: generation_success must be true")
    if "generated_token_ids" in record:
        tokens = record["generated_token_ids"]
        if not isinstance(tokens, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                                              for i in tokens):
            raise ValueError(f"{context}: generated_token_ids must be a list of nonnegative integers")


def _score_saved(records, context):
    """Complete the raw-only parsing pass before accessing labels for scoring."""
    for position, record in enumerate(records):
        _saved_output(record, f"{context}[{position}]")
    raw = [record["raw_output"] for record in records]
    strict = [parse_yes_no(text) for text in raw]
    semantic = [explain_semantic_yes_no(text) for text in raw]
    result = []
    for position, (record, strict_answer, decision) in enumerate(zip(records, strict, semantic)):
        truth = record["label"]
        strict_correct, semantic_correct = strict_answer == truth, decision.answer == truth
        expected = dict(parsed_answer=strict_answer, correct=strict_correct,
                        strict_parsed_answer=strict_answer, strict_correct=strict_correct,
                        strict_invalid=strict_answer is None,
                        semantic_parsed_answer=decision.answer, semantic_correct=semantic_correct,
                        semantic_invalid=decision.answer is None)
        for key, value in expected.items():
            if key in record and (record[key] != value or
                                  (isinstance(value, bool) and type(record[key]) is not bool)):
                raise ValueError(f"{context}[{position}]: stored {key} disagrees with frozen parser")
        if "semantic_parser_version" in record and record["semantic_parser_version"] != PARSER_VERSION:
            raise ValueError(f"{context}[{position}]: semantic parser version changed")
        result.append(dict(question_id=record["question_id"], image_id=record["image_id"],
                           image=record["image"], question=record.get("text", record.get("question")),
                           label=truth, raw_output=record["raw_output"],
                           generated_token_ids=(list(record["generated_token_ids"])
                                                if "generated_token_ids" in record else None),
                           strict_parsed_answer=strict_answer, strict_correct=strict_correct,
                           strict_invalid=strict_answer is None,
                           semantic_parsed_answer=decision.answer, semantic_correct=semantic_correct,
                           semantic_invalid=decision.answer is None,
                           semantic_rule_id=decision.rule_id, semantic_reason=decision.reason))
    return result


def _metrics(rows, mode):
    answer = mode + "_parsed_answer"
    return compute_metrics([dict(label=row["label"], parsed_answer=row[answer]) for row in rows])


def _metric_summary(rows):
    return dict(question_count=len(rows), question_ids=[row["question_id"] for row in rows],
                strict_metrics=_metrics(rows, "strict"), semantic_metrics=_metrics(rows, "semantic"))


def validate_full_baseline(manifest, full_records, *, expected_count=30,
                           expected_semantic_correct=26):
    """Validate ordered original identities and reproduce the saved FullModel.

    Defaults enforce this experiment's frozen 30-question, 26/30 semantic
    baseline.  Synthetic callers may supply a smaller expected_count and an
    explicit expected_semantic_correct (or None to skip only that count gate).
    All supplied saved parser decisions/correctness fields must still match.
    The returned scored_records are new dictionaries; inputs are not modified.
    """
    expected_count = _integer(expected_count, "expected_count", minimum=1)
    manifest, manifest_ids, _ = _indexed_records(manifest, "manifest")
    full_records, full_ids, _ = _indexed_records(full_records, "full_records")
    if len(manifest) != expected_count or len(full_records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} manifest and FullModel records")
    if manifest_ids != full_ids:
        raise ValueError("FullModel records differ from the original ordered manifest identities")
    for position, (annotation, full) in enumerate(zip(manifest, full_records)):
        if "sample_order" in annotation and full.get("sample_order") != annotation["sample_order"]:
            raise ValueError(f"FullModel sample_order differs at position {position}")
        if "original_visual_tokens" in full and "retained_visual_tokens" in full:
            n = _integer(full["original_visual_tokens"], "original_visual_tokens", minimum=1)
            if full["retained_visual_tokens"] != n:
                raise ValueError("Stored FullModel baseline was pruned")
    rows = _score_saved(full_records, "full_records")
    semantic_correct = sum(row["semantic_correct"] for row in rows)
    if expected_semantic_correct is not None:
        expected_semantic_correct = _integer(expected_semantic_correct, "expected_semantic_correct")
        if semantic_correct != expected_semantic_correct:
            raise ValueError(f"FullModel semantic correct count changed: expected "
                             f"{expected_semantic_correct}/{expected_count}, got "
                             f"{semantic_correct}/{len(rows)}")
    return dict(identity_and_order_verified=True, parser_decisions_recomputed=True,
                semantic_parser_version=PARSER_VERSION,
                strict_correct_count=sum(row["strict_correct"] for row in rows),
                semantic_correct_count=semantic_correct,
                label_counts=dict(Counter(row["label"] for row in rows)),
                scored_records=rows, **_metric_summary(rows))


def _delta(left, right, mode):
    """Compare right against left; gains/losses refer to correctness, not labels."""
    answer, correct = mode + "_parsed_answer", mode + "_correct"
    qids = [row["question_id"] for row in left]
    if qids != [row["question_id"] for row in right]:
        raise ValueError("Pairwise rows must refer to the same ordered questions")
    changed_answers, changed_correct, gained, lost = [], [], [], []
    both_correct, both_incorrect = [], []
    invalid_to_valid, valid_to_invalid = [], []
    for before, after in zip(left, right):
        qid = before["question_id"]
        if before[answer] != after[answer]:
            changed_answers.append(qid)
        if before[correct] != after[correct]:
            changed_correct.append(qid)
        if not before[correct] and after[correct]:
            gained.append(qid)
        elif before[correct] and not after[correct]:
            lost.append(qid)
        elif before[correct]:
            both_correct.append(qid)
        else:
            both_incorrect.append(qid)
        if before[answer] is None and after[answer] is not None:
            invalid_to_valid.append(qid)
        if before[answer] is not None and after[answer] is None:
            valid_to_invalid.append(qid)
    before_correct = sum(row[correct] for row in left)
    after_correct = sum(row[correct] for row in right)
    return dict(n=len(left), left_correct=before_correct, right_correct=after_correct,
                correct_count_difference=after_correct - before_correct,
                accuracy_difference_pp=(100 * (after_correct - before_correct) / len(left)
                                        if left else 0.0),
                changed_answer_question_ids=changed_answers,
                correctness_changed_question_ids=changed_correct,
                gained_correct_question_ids=gained, lost_correct_question_ids=lost,
                both_correct_question_ids=both_correct, both_incorrect_question_ids=both_incorrect,
                invalid_to_valid_question_ids=invalid_to_valid,
                valid_to_invalid_question_ids=valid_to_invalid)


def build_comparison_report(manifest, full_records, predictions, *, methods=DEFAULT_METHODS,
                            budgets=(64, 128), question_ids=None, require_complete=True,
                            pairs=DEFAULT_PAIRS, baseline_expected_count=30,
                            baseline_expected_semantic_correct=26):
    """Score matched saved records, validating exact (qid, method, K) coverage.

    Unified predictions use the existing smoke schema: question_id, image_id,
    image, question (or text), label, method, k and raw_output.  Saved strict/
    semantic decisions are optional but checked when supplied.  Successful
    records only are accepted; error records must be reported as run failures.

    Explicit question_ids define a smoke subset, retaining original manifest
    order.  With five methods and two budgets, two questions require 20 unique
    records; the complete 30-question run requires exactly 300.  Missing records
    raise unless require_complete=False, which creates a prominently incomplete
    report with actual per-group and pairwise denominators.  Unexpected/duplicate
    records always raise.  Pairwise deltas are right minus left; versus_full is
    the method minus FullModel on the identical subset.  Both the full baseline
    and scoped baseline are retained, preventing denominator confusion.
    """
    baseline = validate_full_baseline(
        manifest, full_records, expected_count=baseline_expected_count,
        expected_semantic_correct=baseline_expected_semantic_correct)
    full_scored = baseline["scored_records"]
    original_ids = baseline["question_ids"]
    originals = {row["question_id"]: row for row in full_scored}
    methods = tuple(methods)
    if not methods or any(not isinstance(m, str) or not m for m in methods) or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty unique names")
    budgets = tuple(_integer(k, "budget", minimum=1) for k in budgets)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be nonempty and unique")
    pairs = tuple(tuple(pair) for pair in pairs)
    if any(len(pair) != 2 or pair[0] not in methods or pair[1] not in methods or pair[0] == pair[1]
           for pair in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("pairs must contain unique distinct method pairs from methods")
    if type(require_complete) is not bool:
        raise ValueError("require_complete must be boolean")
    if question_ids is None:
        chosen_ids = list(original_ids)
    else:
        requested = [_integer(qid, "question_id") for qid in question_ids]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("question_ids must be nonempty and unique")
        if not set(requested).issubset(originals):
            raise ValueError("question_ids include a question outside the original manifest")
        wanted = set(requested)
        chosen_ids = [qid for qid in original_ids if qid in wanted]
    chosen = set(chosen_ids)
    expected_keys = {(qid, method, k) for qid in chosen_ids for method in methods for k in budgets}
    indexed, source_rows = {}, []
    for position, record in enumerate(predictions):
        identity = _identity(record, f"predictions[{position}]")
        qid = identity[0]
        if qid not in chosen:
            raise ValueError(f"Prediction question {qid} is outside the declared evaluation scope")
        full = originals[qid]
        expected_identity = (qid, full["image_id"], full["image"], full["question"], full["label"])
        if identity != expected_identity:
            raise ValueError(f"Prediction identity differs from original manifest for question {qid}")
        method = record.get("method")
        k = _integer(record.get("k"), f"predictions[{position}]: k", minimum=1)
        if method not in methods or k not in budgets:
            raise ValueError(f"Unexpected method/budget in prediction {position}")
        key = qid, method, k
        if key in indexed:
            raise ValueError(f"Duplicate prediction key {key}")
        indexed[key] = record
        source_rows.append(record)
    missing = expected_keys.difference(indexed)
    if missing and require_complete:
        raise ValueError(f"Incomplete declared comparison: {len(indexed)}/{len(expected_keys)} records, "
                         f"{len(missing)} missing (qid, method, K) combinations")
    scored = _score_saved(source_rows, "predictions")
    scored_index = {(source["question_id"], source["method"], source["k"]): row
                    for source, row in zip(source_rows, scored)}
    subset = [originals[qid] for qid in chosen_ids]
    groups = []
    for method in methods:
        for k in budgets:
            rows = [scored_index[(qid, method, k)] for qid in chosen_ids if (qid, method, k) in scored_index]
            matched_full = [originals[row["question_id"]] for row in rows]
            groups.append(dict(method=method, k=k, complete_for_scope=len(rows) == len(chosen_ids),
                               missing_question_ids=[qid for qid in chosen_ids if (qid, method, k) not in scored_index],
                               versus_full={mode: _delta(matched_full, rows, mode) for mode in ("strict", "semantic")},
                               **_metric_summary(rows)))
    pairwise = []
    for left_method, right_method in pairs:
        for k in budgets:
            common_ids = [qid for qid in chosen_ids if (qid, left_method, k) in scored_index
                          and (qid, right_method, k) in scored_index]
            left = [scored_index[(qid, left_method, k)] for qid in common_ids]
            right = [scored_index[(qid, right_method, k)] for qid in common_ids]
            pairwise.append(dict(left_method=left_method, right_method=right_method, k=k,
                                 question_ids=common_ids, n=len(common_ids),
                                 complete_for_scope=len(common_ids) == len(chosen_ids),
                                 strict=_delta(left, right, "strict"),
                                 semantic=_delta(left, right, "semantic"),
                                 raw_output_changed_question_ids=[qid for qid, a, b in zip(common_ids, left, right)
                                                                  if a["raw_output"] != b["raw_output"]],
                                 generated_token_ids_changed_question_ids=[qid for qid, a, b in zip(common_ids, left, right)
                                                                           if a["generated_token_ids"] != b["generated_token_ids"]]))
    questions = []
    for qid in chosen_ids:
        full = originals[qid]
        runs = []
        for method in methods:
            for k in budgets:
                row = scored_index.get((qid, method, k))
                if row is None:
                    runs.append(dict(method=method, k=k, status="missing"))
                else:
                    item = dict(row, method=method, k=k, status="completed")
                    for mode in ("strict", "semantic"):
                        item[mode + "_answer_changed_vs_full"] = row[mode + "_parsed_answer"] != full[mode + "_parsed_answer"]
                        item[mode + "_correctness_changed_vs_full"] = row[mode + "_correct"] != full[mode + "_correct"]
                    runs.append(item)
        questions.append(dict(question_id=qid, image_id=full["image_id"], image=full["image"],
                              question=full["question"], label=full["label"], full=dict(full), methods=runs))
    is_subset = len(chosen_ids) != len(original_ids)
    complete = not missing
    label = (f"{len(chosen_ids)}-question smoke subset; not a {len(original_ids)}-question result"
             if is_subset else f"{len(original_ids)}-question controlled comparison")
    if not complete:
        label = "INCOMPLETE: " + label
    scope = dict(label=label, full_manifest_count=len(original_ids),
                 evaluated_question_count=len(chosen_ids), question_ids=chosen_ids,
                 expected_prediction_records=len(expected_keys), actual_prediction_records=len(indexed),
                 complete_for_scope=complete, subset_only=is_subset,
                 full_30_comparison_complete=(len(original_ids) == 30 and not is_subset and complete
                                              and len(expected_keys) == 300
                                              and set(methods) == set(DEFAULT_METHODS)
                                              and set(budgets) == {64, 128}),
                 missing_keys=[dict(question_id=qid, method=method, k=k)
                               for qid, method, k in sorted(missing)])
    # The compatibility name deliberately identifies the frozen original pilot;
    # synthetic callers can inspect full_manifest_count if expected_count differs.
    return dict(scope=scope, methods=list(methods), budgets=list(budgets),
                semantic_parser_version=PARSER_VERSION, report_builder_executed_inference=False,
                all_decisions_completed_before_label_scoring=True,
                full_baseline_30=baseline, full_baseline_subset=_metric_summary(subset),
                groups=groups, pairwise=pairwise, per_question=questions,
                interpretation_notes=[
                    "All invalid answers remain incorrect and a separate confusion column.",
                    "FullModel and each method use the same questions within every comparison.",
                    "Smoke subset results are not the complete 30-question pilot.",
                    "Pairwise deltas are right minus left; versus_full deltas are method minus FullModel.",
                    "Raw answers and generated IDs are retained; no parser or inference is changed.",
                    "Coverage objective values do not establish downstream VQA superiority."])
