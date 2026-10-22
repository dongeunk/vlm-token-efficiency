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

"""Synthetic, raw-only semantic parser tests; no pilot labels or model calls.

Object names deliberately differ across fixtures. These tests validate general
grammar and conservative abstention, not a list of question IDs or benchmark
answers. No network, torch, Transformers, or image files are needed.
"""

from dataclasses import FrozenInstanceError, is_dataclass
import inspect
import unittest

from src.evaluation.semantic_yes_no import explain_semantic_yes_no, parse_semantic_yes_no


class SemanticYesNoTest(unittest.TestCase):
    def assert_answers(self, fixtures, expected):
        for raw in fixtures:
            with self.subTest(raw=raw):
                self.assertEqual(parse_semantic_yes_no(raw), expected)
                explanation = explain_semantic_yes_no(raw)
                self.assertEqual(explanation.answer, expected)
                self.assertIsInstance(explanation.rule_id, str)
                self.assertTrue(explanation.rule_id)

    def test_clear_leading_yes_and_no(self):
        self.assert_answers(("Yes", "YES.", "  Yes, there is a teapot in the image.\n",
                             "Yes. There is an umbrella visible in the image."), "yes")
        self.assert_answers(("No", "no.", "No, there is no zeppelin in the image.",
                             "No. There are no skateboards visible in the picture."), "no")

    def test_complete_visual_absence_assertions_generalize_across_objects(self):
        for noun in ("zeppelin", "saxophone", "fire hydrant", "teapot", "traffic light"):
            self.assert_answers((f"There is no {noun} present in the image.",
                                 f"There is no {noun} visible in the image.",
                                 f"  There is no {noun} in the picture.\n"), "no")
        self.assert_answers(("There are no teapots in the picture.",
                             "There are no skateboards visible in the image.",
                             "There are no bicycles present in the picture."), "no")

    def test_complete_visual_presence_assertions_generalize_across_objects(self):
        self.assert_answers(("There is a zeppelin in the image.",
                             "There is a saxophone visible in the image.",
                             "There is a fire hydrant present in the picture.",
                             "There is an umbrella in the image.",
                             "There is an elephant visible in the picture."), "yes")

    def test_no_mention_no_evidence_and_negation_scope_are_not_absence(self):
        self.assert_answers((
            "There is no mention of a saxophone in the image.",
            "There is no mention of a teapot in the provided facts, so we cannot determine whether one is present.",
            "There is no evidence of a teapot in the image.",
            "There is no doubt that a teapot is in the image.",
            "There is not only a teapot in the image.",
            "There is no reason to think a zeppelin is present in the image.",
            "There is no guarantee of a teapot in the image.",
            "There is no absence of a teapot in the image.",
            "There is no lack of a teapot in the image.",
            "I cannot rule out a saxophone in the image.",
            "No, I cannot rule out a teapot in the image.",
        ), None)

    def test_uncertainty_cannot_be_overridden_by_leading_answer(self):
        self.assert_answers((
            "Not sure whether there is a teapot in the image.",
            "There is no zeppelin present in the image, but I am not sure.",
            "There is a teapot in the image, perhaps.",
            "Yes, I am not sure.", "No, I cannot determine that.",
            "Yes, I am unable to tell.", "No, there might be a teapot.",
            "Yes, I think there is a fire hydrant in the image.",
            "No, probably.", "Yes, maybe.",
            "There is no teapot visible in the image, although the image is unclear.",
        ), None)

    def test_cross_polarity_contradictions_and_multiple_objects_abstain(self):
        self.assert_answers((
            "Yes, there is no teapot in the image.",
            "Yes. There are no skateboards visible in the image.",
            "No, there is a saxophone in the image.",
            "No. There is an umbrella visible in the picture.",
            "Yes and no.", "No, yes.",
            "There is no teapot in the image, but there is an umbrella in the image.",
            "There is a saxophone in the image. There is no saxophone in the image.",
            "There is a teapot and no umbrella in the image.",
        ), None)

    def test_question_echoes_quotes_and_nonvisual_statements_abstain(self):
        self.assert_answers((
            "Is there a teapot in the image? No.",
            "Question: Is there a teapot in the image? Answer: no.",
            "Yes. Is there a teapot in the image?",
            '"There is no teapot present in the image."',
            "'There is a saxophone in the image.'",
            '"No," says the caption.',
            "There is no teapot in the image?",
            "There is no teapot in the kitchen.",
            "A teapot is mentioned in the question.",
            "No evidence proves that no teapot is present.",
        ), None)

    def test_incomplete_clauses_do_not_turn_into_absence(self):
        self.assert_answers((
            "There is no teapot", "There is no teapot present in the", "There is a teapot in the",
            "There are no", "There is no mention of a", "There is no teapot present in the image, but",
            "There is no teapot visible in the image, so we cannot determine",
            "There is a saxophone in the picture, and",
        ), None)

    def test_api_uses_only_raw_text_and_ignores_counterfactual_context(self):
        for function in (parse_semantic_yes_no, explain_semantic_yes_no):
            self.assertEqual(tuple(inspect.signature(function).parameters), ("text",))
        raw = "There is no saxophone visible in the image."
        contexts = (
            {"question_id": 1, "text": "Is there a saxophone?", "label": "yes", "raw_output": raw},
            {"question_id": 999, "text": "Is there a teapot?", "label": "no", "raw_output": raw},
        )
        self.assertEqual([parse_semantic_yes_no(record["raw_output"]) for record in contexts], ["no", "no"])
        with self.assertRaises(TypeError):
            parse_semantic_yes_no(raw, label="no")

    def test_explanation_is_frozen_auditable_and_raw_text_is_not_modified(self):
        raw = "  There is no fire hydrant visible in the image.\n"
        record = {"raw_output": raw}
        explanation = explain_semantic_yes_no(record["raw_output"])
        self.assertTrue(is_dataclass(explanation))
        self.assertEqual(explanation.answer, "no")
        for name in ("rule_id", "evidence", "reason"):
            self.assertTrue(hasattr(explanation, name))
        self.assertIsInstance(explanation.reason, str)
        self.assertTrue(explanation.reason)
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(record["raw_output"], "  There is no fire hydrant visible in the image.\n")
        with self.assertRaises(FrozenInstanceError):
            explanation.answer = "yes"

    def test_empty_nonstring_and_nonanswer_boundaries_are_invalid(self):
        self.assert_answers((None, 0, False, [], b"No.", "", " \n", "Nobody", "Yesterday",
                             "Nope", "yesman", "There is nobody in the image."), None)


if __name__ == "__main__":
    unittest.main()
