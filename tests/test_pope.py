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

"""POPE core checks requiring neither model packages nor network access."""

import copy
import io
import itertools
import json
import random
import unittest
from unittest.mock import patch

from src.evaluation.pope import compute_metrics, load_annotations, parse_yes_no, sample_balanced


def annotation(question_id, image_id, label):
    return dict(question_id=question_id, image=f"COCO_val2014_{image_id:012d}.jpg",
                text="Is there a cat in the image?", label=label)


class PopeAnnotationTest(unittest.TestCase):
    def load_lines(self, lines):
        # Keep annotation fixtures in memory; no machine-specific temp paths.
        with patch("src.evaluation.pope.Path.open", return_value=io.StringIO("\n".join(lines))):
            return load_annotations("official_annotations.jsonl")

    def test_jsonl_preserves_original_fields_and_parses_image_id(self):
        record = annotation(1, 123, "yes")
        record["text"] = "  Is there a cat?  "
        record["extra"] = {"source": "official"}
        result = self.load_lines(["", json.dumps(record), " "])
        self.assertEqual(result, [dict(record, image_id=123)])

    def test_empty_malformed_or_non_object_annotations_fail(self):
        for lines in ([], [" "], ["{"], ["[]"], ["null"]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                self.load_lines(lines)

    def test_validation_rejects_invalid_required_fields(self):
        base = annotation(1, 123, "yes")
        invalid = [dict(base, question_id=True), dict(base, question_id=-1),
                   dict(base, question_id="1"), dict(base, question_id=1.0),
                   dict(base, image="COCO_val2014_123.jpg"),
                   dict(base, image="images/COCO_val2014_000000000123.jpg"),
                   dict(base, image="COCO_val2014_000000000123.JPG"), dict(base, image=None),
                   dict(base, image="COCO_val2014_１２３４５６７８９０１２.jpg"),
                   dict(base, text="  "), dict(base, text=None), dict(base, label="Yes"),
                   dict(base, label="unknown"), dict(base, image_id=124), dict(base, image_id=True)]
        for field in base:
            record = dict(base)
            del record[field]
            invalid.append(record)
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.load_lines([json.dumps(record)])

    def test_duplicate_question_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
            self.load_lines([json.dumps(annotation(1, 1, "yes")), json.dumps(annotation(1, 2, "no"))])


class PopeSamplingTest(unittest.TestCase):
    def test_seed42_balanced30_distinct_and_input_order_independent(self):
        records = [annotation(index, index, "yes" if index % 2 else "no") for index in range(1, 81)]
        original = copy.deepcopy(records)
        first = sample_balanced(records)
        shuffled = list(records)
        random.Random(999).shuffle(shuffled)
        second = sample_balanced(shuffled)
        self.assertEqual(first, second)
        self.assertEqual(records, original)
        self.assertEqual(len(first), 30)
        self.assertEqual(sum(record["label"] == "yes" for record in first), 15)
        self.assertEqual(sum(record["label"] == "no" for record in first), 15)
        self.assertEqual(len({record["image"] for record in first}), 30)
        self.assertEqual(len({record["question_id"] for record in first}), 30)
        self.assertNotEqual(first, sample_balanced(records, seed=43))

    def test_matching_preserves_distinct_images_when_greedy_can_block(self):
        # Images 1/2 supply both classes; 3/4 supply only yes. Balanced p=2 can
        # use all four, but assigning yes to images 1/2 first could block that.
        records = [annotation(1, 1, "yes"), annotation(2, 2, "yes"),
                   annotation(3, 3, "yes"), annotation(4, 4, "yes"),
                   annotation(5, 1, "no"), annotation(6, 2, "no")]
        for seed in range(10):
            with self.subTest(seed=seed):
                selected = sample_balanced(records, per_label=2, seed=seed)
                self.assertEqual(len({record["image"] for record in selected}), 4)

    def test_limited_diversity_uses_distinct_questions_without_replacement(self):
        records = [annotation(index, index % 3, "yes" if index % 2 else "no") for index in range(1, 21)]
        selected = sample_balanced(records, per_label=5)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({record["question_id"] for record in selected}), 10)
        self.assertEqual(len({record["image"] for record in selected}), 3)
        self.assertEqual(sum(record["label"] == "yes" for record in selected), 5)

    def test_small_selection_matches_exhaustive_maximum_diversity(self):
        rng = random.Random(47)
        for fixture in range(20):
            records = [annotation(index, rng.randrange(5), "yes" if index < 4 else "no")
                       for index in range(8)]
            yes, no = records[:4], records[4:]
            expected = max(len({record["image"] for record in a + b})
                           for a in itertools.combinations(yes, 2)
                           for b in itertools.combinations(no, 2))
            selected = sample_balanced(records, per_label=2, seed=fixture)
            self.assertEqual(len({record["image"] for record in selected}), expected)

    def test_invalid_sampling_arguments_and_shortage(self):
        records = [annotation(1, 1, "yes"), annotation(2, 2, "no")]
        self.assertEqual(sample_balanced(records, per_label=0), [])
        with self.assertRaises(ValueError):
            sample_balanced(records, per_label=2)
        with self.assertRaises(ValueError):
            sample_balanced(records, per_label=-1)
        for value in (True, 1.5, "15"):
            with self.subTest(per_label=value), self.assertRaises(TypeError):
                sample_balanced(records, per_label=value)
        for value in (True, 42.0, "42"):
            with self.subTest(seed=value), self.assertRaises(TypeError):
                sample_balanced(records, per_label=1, seed=value)


class PopeParserTest(unittest.TestCase):
    def test_clear_answers_and_explanations(self):
        accepted = {"Yes": "yes", " NO. ": "no", "yEs, there is a cat.": "yes",
                    "No, there are no cats.": "no", "Yes! A cat is visible.": "yes",
                    "No; the image shows a dog.": "no", "Yes: nobody is obscuring it.": "yes",
                    "No. Yesterday was sunny.": "no", "Yes\nA cat is visible.": "yes"}
        for text, expected in accepted.items():
            with self.subTest(text=text):
                self.assertEqual(parse_yes_no(text), expected)

    def test_invalid_leading_word_boundaries_and_prefixes(self):
        invalid = ("", "  ", None, 1, "Yesterday", "Nobody", "yesman", "not", "yes_indeed",
                   '"Yes"', "Answer: yes", "```yes```", "The answer is yes.", "Yes-man", "No-confidence", "Yeſ")
        for text in invalid:
            with self.subTest(text=text):
                self.assertIsNone(parse_yes_no(text))

    def test_both_standalone_answer_words_are_ambiguous(self):
        for text in ("Yes and no.", "No, yes.", "Yes/no", "Yes, there are no cats.",
                     "Yes. 'NO' is also possible.", "No—yes"):
            with self.subTest(text=text):
                self.assertIsNone(parse_yes_no(text))

    def test_question_echoes_and_uncertainty_are_rejected(self):
        invalid = ("Is there a cat? Yes.", "Yes. Is there a cat?", "Yes?", "No. Question: Is a cat present.",
                   "Yes. Is there a cat in this image.", "Yes, maybe.", "No, I am not sure.",
                   "Yes, I cannot determine it.", "No, unable to tell.", "Yes, I think so.",
                   "Yes, probably.", "No, perhaps.", "Yes, it appears to be a cat.",
                   "Yes, it might be a cat.", "No, I don't know.", "Yes, I can’t confirm.",
                   "No, not enough information.", "Yes, cannot be determined.",
                   "Yes, there could be a cat.", "No, I am not entirely sure.", "Yes, I cannot be sure.")
        for text in invalid:
            with self.subTest(text=text):
                self.assertIsNone(parse_yes_no(text))

    def test_raw_output_is_preserved(self):
        raw = "  YES, there is a cat.\n"
        record = {"raw_answer": raw, "label": "yes"}
        record["parsed_answer"] = parse_yes_no(record["raw_answer"])
        before = copy.deepcopy(record)
        compute_metrics([record])
        self.assertEqual(record, before)
        self.assertEqual(record["raw_answer"], raw)


class PopeMetricsTest(unittest.TestCase):
    def test_independently_expected_confusion_and_all_item_denominators(self):
        records = [dict(label=label, parsed_answer=answer) for label, answers in
                   (("yes", ("yes", "yes", "no", None)), ("no", ("yes", "no", None, None)))
                   for answer in answers]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["n"], 8)
        self.assertEqual(metrics["confusion_matrix"], {"yes": {"yes": 2, "no": 1, "invalid": 1},
                                                        "no": {"yes": 1, "no": 1, "invalid": 2}})
        self.assertEqual(metrics["support"], {"yes": 4, "no": 4})
        self.assertEqual(metrics["accuracy"], 3 / 8)
        self.assertEqual(metrics["yes_class_accuracy"], 1 / 2)
        self.assertEqual(metrics["no_class_accuracy"], 1 / 4)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)
        self.assertEqual(metrics["yes_prediction_ratio"], 3 / 8)
        self.assertEqual(metrics["invalid_ratio"], 3 / 8)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (2, 1, 1, 1))

    def test_all_invalid_answers_remain_invalid_and_incorrect(self):
        metrics = compute_metrics([dict(label=label, parsed_answer=None) for label in ("yes", "no")])
        for key in ("accuracy", "yes_class_accuracy", "no_class_accuracy", "precision", "recall", "f1", "yes_prediction_ratio"):
            self.assertEqual(metrics[key], 0.0)
        self.assertEqual(metrics["invalid_ratio"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["no"]["invalid"], 1)
        self.assertEqual(metrics["tn"], 0)

    def test_empty_and_absent_classes_have_defined_zero_ratios(self):
        empty = compute_metrics([])
        self.assertEqual(empty["n"], 0)
        for key in ("accuracy", "yes_class_accuracy", "no_class_accuracy", "precision", "recall", "f1", "yes_prediction_ratio", "invalid_ratio"):
            self.assertEqual(empty[key], 0.0)
        all_no = compute_metrics([dict(label="no", parsed_answer="no")])
        self.assertEqual(all_no["accuracy"], 1)
        self.assertEqual(all_no["no_class_accuracy"], 1)
        self.assertEqual(all_no["yes_class_accuracy"], 0)
        self.assertEqual(all_no["f1"], 0)

    def test_invalid_metric_records_fail(self):
        for record in ({}, {"label": "yes"}, {"label": "YES", "parsed_answer": "yes"},
                       {"label": "no", "parsed_answer": "invalid"}, {"label": "no", "parsed_answer": False}, None):
            with self.subTest(record=record), self.assertRaises(ValueError):
                compute_metrics([record])


if __name__ == "__main__":
    unittest.main()
