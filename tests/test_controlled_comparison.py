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

"""CPU-only tests of saved-record comparison, with no inference or file writes.

Synthetic responses below are test fixtures, not benchmark/model outputs.
An optional read-only regression reproduces the existing original FullModel
pilot when those local result files are available.
"""

import copy
import json
from pathlib import Path
import random
import unittest

from src.evaluation.controlled_comparison import (
    DEFAULT_METHODS,
    build_comparison_report,
    validate_full_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
SAVED_MANIFEST = ROOT / "results/pope_pilot/sample_manifest.jsonl"
SAVED_FULL = ROOT / "results/pope_pilot/full_model_seed42/predictions.jsonl"


def annotation(qid):
    return dict(question_id=qid, image_id=qid, image=f"COCO_val2014_{qid:012d}.jpg",
                text="Is there a saxophone in the image?", label="yes" if qid % 2 else "no",
                sample_order=qid - 1)


def token_fixture(raw):
    # Tiny deterministic IDs are solely for verifying retained metadata/deltas.
    return [1 if raw == "Yes." else 2 if raw == "No." else 3]


def baseline_fixture(count=30):
    manifest = [annotation(qid) for qid in range(1, count + 1)]
    full = []
    for row in manifest:
        qid = row["question_id"]
        raw = "Yes." if row["label"] == "yes" else "No."
        if count == 30:
            if qid in (27, 28):
                raw = "No." if row["label"] == "yes" else "Yes."
            elif qid in (29, 30):
                raw = "Maybe."
        full.append(dict(row, raw_output=raw, generated_token_ids=token_fixture(raw),
                         generation_success=True, original_visual_tokens=832,
                         retained_visual_tokens=832))
    return manifest, full


def prediction_fixture(manifest, qids=None, overrides=None):
    wanted = set(qids) if qids is not None else {row["question_id"] for row in manifest}
    overrides = overrides or {}
    records = []
    for row in manifest:
        qid = row["question_id"]
        if qid not in wanted:
            continue
        for method in DEFAULT_METHODS:
            for k in (64, 128):
                raw = overrides.get((qid, method, k), "Yes." if row["label"] == "yes" else "No.")
                record = dict(row)
                record["question"] = record.pop("text")
                record.update(method=method, k=k, raw_output=raw,
                              generated_token_ids=token_fixture(raw), error=None,
                              generation_success=True)
                records.append(record)
    return records


def group(report, method, k):
    return next(row for row in report["groups"] if row["method"] == method and row["k"] == k)


class ControlledBaselineTest(unittest.TestCase):
    def test_default_synthetic30_reproduces26_correct_with_invalid_penalty(self):
        manifest, full = baseline_fixture()
        result = validate_full_baseline(manifest, full)
        self.assertTrue(result["identity_and_order_verified"])
        self.assertEqual(result["question_count"], 30)
        self.assertEqual(result["semantic_correct_count"], 26)
        self.assertEqual(result["strict_correct_count"], 26)
        metrics = result["semantic_metrics"]
        self.assertEqual(metrics["accuracy"], 26 / 30)
        self.assertEqual(metrics["invalid_ratio"], 2 / 30)
        self.assertEqual(metrics["confusion_matrix"], {
            "yes": {"yes": 13, "no": 1, "invalid": 1},
            "no": {"yes": 1, "no": 13, "invalid": 1},
        })

    def test_baseline_identity_order_duplicate_and_sample_order_rejected(self):
        manifest, full = baseline_fixture()
        variants = []
        variants.append(list(reversed(full)))
        duplicate = copy.deepcopy(full)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        variants.append(duplicate)
        for field, value in (("text", "Is there a teapot in the image?"),
                             ("label", "no"), ("sample_order", 999),
                             ("image_id", 999), ("image", "COCO_val2014_000000000999.jpg")):
            changed = copy.deepcopy(full)
            changed[0][field] = value
            variants.append(changed)
        for index, records in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                validate_full_baseline(manifest, records)
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest[-1] = copy.deepcopy(bad_manifest[0])
        with self.assertRaises(ValueError):
            validate_full_baseline(bad_manifest, full)

    def test_baseline_count_semantic_gate_and_no_pruning_are_enforced(self):
        manifest, full = baseline_fixture()
        with self.assertRaises(ValueError):
            validate_full_baseline(manifest[:-1], full[:-1])
        changed = copy.deepcopy(full)
        changed[26]["raw_output"] = "Yes."
        with self.assertRaisesRegex(ValueError, "semantic correct count changed"):
            validate_full_baseline(manifest, changed)
        changed = copy.deepcopy(full)
        changed[0]["retained_visual_tokens"] = 64
        with self.assertRaisesRegex(ValueError, "was pruned"):
            validate_full_baseline(manifest, changed)

    @unittest.skipUnless(SAVED_MANIFEST.is_file() and SAVED_FULL.is_file(), "Original local FullModel artifacts unavailable")
    def test_existing_original_baseline26_semantic_and18_strict_read_only(self):
        manifest = [json.loads(line) for line in SAVED_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
        full = [json.loads(line) for line in SAVED_FULL.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = validate_full_baseline(manifest, full)
        self.assertEqual(result["semantic_correct_count"], 26)
        self.assertEqual(result["strict_correct_count"], 18)
        self.assertEqual(result["semantic_metrics"]["accuracy"], 26 / 30)
        self.assertEqual(result["semantic_metrics"]["invalid_ratio"], 2 / 30)
        self.assertEqual(result["strict_metrics"]["accuracy"], 18 / 30)
        self.assertEqual(result["strict_metrics"]["invalid_ratio"], 12 / 30)


class ControlledScoringTest(unittest.TestCase):
    def test_both_parsers_invalids_count_as_incorrect_over_all_questions(self):
        manifest, full = baseline_fixture(4)
        overrides = {(qid, method, k): "Maybe." for qid in (1, 2)
                     for method in DEFAULT_METHODS for k in (64, 128)}
        predictions = prediction_fixture(manifest, overrides=overrides)
        report = build_comparison_report(manifest, full, predictions,
            baseline_expected_count=4, baseline_expected_semantic_correct=None)
        for mode in ("strict", "semantic"):
            metrics = group(report, "ours", 64)[mode + "_metrics"]
            self.assertEqual(metrics["n"], 4)
            self.assertEqual(metrics["accuracy"], 1 / 2)
            self.assertEqual(metrics["yes_class_accuracy"], 1 / 2)
            self.assertEqual(metrics["no_class_accuracy"], 1 / 2)
            self.assertEqual(metrics["precision"], 1)
            self.assertEqual(metrics["recall"], 1 / 2)
            self.assertAlmostEqual(metrics["f1"], 2 / 3)
            self.assertEqual(metrics["yes_prediction_ratio"], 1 / 4)
            self.assertEqual(metrics["invalid_ratio"], 1 / 2)
            self.assertEqual(metrics["confusion_matrix"], {
                "yes": {"yes": 1, "no": 0, "invalid": 1},
                "no": {"yes": 0, "no": 1, "invalid": 1}})

    def test_strict_and_frozen_semantic_parsers_are_scored_separately(self):
        manifest, full = baseline_fixture(2)
        answers = {1: "There is a saxophone visible in the image.",
                   2: "There is no saxophone visible in the image."}
        overrides = {(qid, method, k): raw for qid, raw in answers.items()
                     for method in DEFAULT_METHODS for k in (64, 128)}
        report = build_comparison_report(manifest, full, prediction_fixture(manifest, overrides=overrides),
            baseline_expected_count=2, baseline_expected_semantic_correct=None)
        run = group(report, "pure_gc", 128)
        self.assertEqual(run["strict_metrics"]["accuracy"], 0)
        self.assertEqual(run["strict_metrics"]["invalid_ratio"], 1)
        self.assertEqual(run["semantic_metrics"]["accuracy"], 1)
        self.assertEqual(run["semantic_metrics"]["invalid_ratio"], 0)

    def test_saved_parser_answers_flags_and_version_cannot_disagree_with_raw(self):
        manifest, full = baseline_fixture(2)
        predictions = prediction_fixture(manifest)
        wrong_fields = {"parsed_answer": "no", "correct": False, "strict_parsed_answer": "no",
                        "strict_correct": False, "strict_invalid": True,
                        "semantic_parsed_answer": "no", "semantic_correct": False,
                        "semantic_invalid": True, "semantic_parser_version": "changed_parser"}
        for field, value in wrong_fields.items():
            changed = copy.deepcopy(predictions)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "parser"):
                build_comparison_report(manifest, full, changed,
                    baseline_expected_count=2, baseline_expected_semantic_correct=None)
        changed = copy.deepcopy(full)
        changed[0]["correct"] = 1  # bool-equivalent integers must not hide a malformed flag.
        with self.assertRaises(ValueError):
            validate_full_baseline(manifest, changed, expected_count=2, expected_semantic_correct=None)

    def test_error_nontext_and_failed_generation_records_are_not_scored(self):
        manifest, full = baseline_fixture(2)
        predictions = prediction_fixture(manifest)
        variants = (("raw_output", None), ("error", "OOM"), ("generation_success", False),
                    ("generated_token_ids", [True]), ("generated_token_ids", [-1]),
                    ("generated_token_ids", "1,2"))
        for field, value in variants:
            changed = copy.deepcopy(predictions)
            changed[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_comparison_report(manifest, full, changed,
                    baseline_expected_count=2, baseline_expected_semantic_correct=None)

    def test_pairwise_gains_losses_use_matched_questions_and_right_minus_left(self):
        manifest, full = baseline_fixture(4)
        before = {1: "Yes.", 2: "Yes.", 3: "Maybe.", 4: "No."}
        after = {1: "No.", 2: "No.", 3: "Yes.", 4: "Maybe."}
        for record in full:
            record["raw_output"] = before[record["question_id"]]
        overrides = {(qid, method, k): (after[qid] if method == "b_laplacian_quota" else before[qid])
                     for qid in before for method in DEFAULT_METHODS for k in (64, 128)}
        report = build_comparison_report(manifest, full, prediction_fixture(manifest, overrides=overrides),
            baseline_expected_count=4, baseline_expected_semantic_correct=None)
        pair = next(row for row in report["pairwise"]
                    if row["left_method"] == "a_local_only" and row["right_method"] == "b_laplacian_quota" and row["k"] == 64)
        self.assertEqual(pair["question_ids"], [1, 2, 3, 4])
        self.assertEqual(pair["raw_output_changed_question_ids"], [1, 2, 3, 4])
        self.assertEqual(pair["generated_token_ids_changed_question_ids"], [1, 2, 3, 4])
        for mode in ("strict", "semantic"):
            delta = pair[mode]
            self.assertEqual(delta["n"], 4)
            self.assertEqual(delta["left_correct"], 2)
            self.assertEqual(delta["right_correct"], 2)
            self.assertEqual(delta["gained_correct_question_ids"], [2, 3])
            self.assertEqual(delta["lost_correct_question_ids"], [1, 4])
            self.assertEqual(delta["correctness_changed_question_ids"], [1, 2, 3, 4])
            self.assertEqual(delta["invalid_to_valid_question_ids"], [3])
            self.assertEqual(delta["valid_to_invalid_question_ids"], [4])
            self.assertEqual(delta["correct_count_difference"], 0)
            self.assertEqual(delta["accuracy_difference_pp"], 0)
            self.assertEqual(group(report, "b_laplacian_quota", 64)["versus_full"][mode], delta)

    def test_nonzero_pairwise_and_vs_full_differences_have_the_correct_sign(self):
        manifest, full = baseline_fixture(4)  # FullModel is correct on all four.
        before = {1: "No.", 2: "Yes.", 3: "Yes.", 4: "No."}  # A: two correct.
        after = dict(before)
        after[1] = "Yes."  # B: one gain, three correct.
        overrides = {(qid, method, k): (after[qid] if method == "b_laplacian_quota" else before[qid])
                     for qid in before for method in DEFAULT_METHODS for k in (64, 128)}
        report = build_comparison_report(manifest, full, prediction_fixture(manifest, overrides=overrides),
            baseline_expected_count=4, baseline_expected_semantic_correct=None)
        pair = next(row for row in report["pairwise"] if row["left_method"] == "a_local_only"
                    and row["right_method"] == "b_laplacian_quota" and row["k"] == 128)
        for mode in ("strict", "semantic"):
            self.assertEqual(pair[mode]["gained_correct_question_ids"], [1])
            self.assertEqual(pair[mode]["lost_correct_question_ids"], [])
            self.assertEqual(pair[mode]["correct_count_difference"], 1)
            self.assertEqual(pair[mode]["accuracy_difference_pp"], 25)
            self.assertEqual(group(report, "a_local_only", 128)["versus_full"][mode]["correct_count_difference"], -2)
            self.assertEqual(group(report, "b_laplacian_quota", 128)["versus_full"][mode]["accuracy_difference_pp"], -25)


class ControlledCompletenessTest(unittest.TestCase):
    def test_full300_has_all10_groups_and_complete30_denominators(self):
        manifest, full = baseline_fixture()
        report = build_comparison_report(manifest, full, prediction_fixture(manifest))
        scope = report["scope"]
        self.assertEqual(scope["actual_prediction_records"], 300)
        self.assertEqual(scope["expected_prediction_records"], 300)
        self.assertTrue(scope["full_30_comparison_complete"])
        self.assertTrue(scope["complete_for_scope"])
        self.assertFalse(scope["subset_only"])
        self.assertEqual(scope["missing_keys"], [])
        self.assertEqual(len(report["groups"]), 10)
        self.assertEqual(len(report["pairwise"]), 6)
        for run in report["groups"]:
            self.assertEqual(run["question_count"], 30)
            self.assertEqual(run["strict_metrics"]["n"], 30)
            self.assertEqual(run["semantic_metrics"]["n"], 30)
            self.assertEqual(run["versus_full"]["semantic"]["n"], 30)

    def test_explicit_two_question20_subset_does_not_become_full30_result(self):
        manifest, full = baseline_fixture()
        predictions = prediction_fixture(manifest, qids=(1, 2))
        report = build_comparison_report(manifest, full, predictions, question_ids=[2, 1])
        scope = report["scope"]
        self.assertEqual(scope["question_ids"], [1, 2])
        self.assertEqual(scope["actual_prediction_records"], 20)
        self.assertEqual(scope["expected_prediction_records"], 20)
        self.assertTrue(scope["subset_only"])
        self.assertTrue(scope["complete_for_scope"])
        self.assertFalse(scope["full_30_comparison_complete"])
        self.assertIn("not a 30-question result", scope["label"])
        self.assertEqual(report["full_baseline_30"]["semantic_correct_count"], 26)
        self.assertEqual(report["full_baseline_subset"]["semantic_metrics"]["n"], 2)
        for run in report["groups"]:
            self.assertEqual(run["question_ids"], [1, 2])
            self.assertEqual(run["question_count"], 2)

    def test_undeclared20_of300_and_missing_subset_combinations_fail(self):
        manifest, full = baseline_fixture()
        subset = prediction_fixture(manifest, qids=(1, 2))
        with self.assertRaisesRegex(ValueError, "20/300"):
            build_comparison_report(manifest, full, subset)
        with self.assertRaisesRegex(ValueError, "19/20"):
            build_comparison_report(manifest, full, subset[:-1], question_ids=[1, 2])

    def test_partial_report_clearly_labels_missing_and_uses_actual_pair_denominators(self):
        manifest, full = baseline_fixture()
        subset = prediction_fixture(manifest, qids=(1, 2))
        partial = [row for row in subset if not (row["question_id"] == 2
                    and row["method"] == "b_laplacian_quota" and row["k"] == 64)]
        report = build_comparison_report(manifest, full, partial, question_ids=[1, 2], require_complete=False)
        self.assertIn("INCOMPLETE", report["scope"]["label"])
        self.assertFalse(report["scope"]["complete_for_scope"])
        self.assertFalse(report["scope"]["full_30_comparison_complete"])
        self.assertEqual(report["scope"]["actual_prediction_records"], 19)
        run = group(report, "b_laplacian_quota", 64)
        self.assertEqual(run["missing_question_ids"], [2])
        self.assertEqual(run["semantic_metrics"]["n"], 1)
        self.assertEqual(run["versus_full"]["semantic"]["n"], 1)
        pair = next(row for row in report["pairwise"] if row["left_method"] == "a_local_only"
                    and row["right_method"] == "b_laplacian_quota" and row["k"] == 64)
        self.assertEqual(pair["n"], 1)
        self.assertEqual(pair["question_ids"], [1])
        self.assertEqual(pair["semantic"]["n"], 1)

    def test300_records_with_duplicate_key_or_wrong_qid_are_rejected(self):
        manifest, full = baseline_fixture()
        predictions = prediction_fixture(manifest)
        duplicate = copy.deepcopy(predictions)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        self.assertEqual(len(duplicate), 300)
        with self.assertRaisesRegex(ValueError, "Duplicate prediction key"):
            build_comparison_report(manifest, full, duplicate)
        wrong = copy.deepcopy(predictions)
        wrong[-1]["question_id"] = 999
        with self.assertRaisesRegex(ValueError, "outside"):
            build_comparison_report(manifest, full, wrong)

    def test_prediction_identity_unknown_method_budget_and_scope_ids_are_rejected(self):
        manifest, full = baseline_fixture()
        predictions = prediction_fixture(manifest)
        for field, value in (("question", "Is there a teapot?"), ("label", "no"),
                             ("image_id", 99), ("method", "unknown"), ("k", 32)):
            changed = copy.deepcopy(predictions)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_comparison_report(manifest, full, changed)
        for question_ids in ([1, 1], [1, 999], [], [True]):
            with self.subTest(question_ids=question_ids), self.assertRaises(ValueError):
                build_comparison_report(manifest, full, [], question_ids=question_ids)

    def test_unordered_predictions_and_returned_copy_do_not_mutate_input(self):
        manifest, full = baseline_fixture()
        predictions = prediction_fixture(manifest)
        random.Random(42).shuffle(predictions)
        original = copy.deepcopy((manifest, full, predictions))
        report = build_comparison_report(manifest, full, predictions)
        self.assertEqual((manifest, full, predictions), original)
        self.assertEqual(report["scope"]["question_ids"], list(range(1, 31)))
        for run in report["groups"]:
            self.assertEqual(run["question_ids"], list(range(1, 31)))
        report["per_question"][0]["methods"][0]["generated_token_ids"].append(999)
        report["full_baseline_30"]["scored_records"][0]["generated_token_ids"].append(999)
        self.assertEqual((manifest, full, predictions), original)


if __name__ == "__main__":
    unittest.main()
