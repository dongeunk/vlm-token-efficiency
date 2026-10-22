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

"""CPU-only tests for direct insertion and physical visual-token deletion.

The fixture models only a token embedding lookup.  No Transformers model,
image processing, generation, cache, network, or CUDA execution is involved.
"""

from types import SimpleNamespace
import unittest

import torch

from src.inference.pruned_vlm import build_pruned_inputs


class LookupModel:
    """Small deterministic stand-in for the model's embedding API."""

    def __init__(self, dimension=3, dtype=torch.float32):
        self.config = SimpleNamespace(image_token_id=7)
        values = torch.arange(32 * dimension, dtype=torch.float32).reshape(32, dimension)
        self.table = torch.nn.Embedding.from_pretrained(values.to(dtype), freeze=True)

    def get_input_embeddings(self):
        return self.table


def fixture(dtype=torch.float32):
    model = LookupModel(dtype=dtype)
    # 1/2/3/4/5 stand for ordinary text, separators and other special tokens.
    # Image placeholders span two runs, separated by the special token 3.
    ids = torch.tensor([[1, 2, 7, 7, 3, 7, 7, 4, 5]], dtype=torch.long)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    visual = (torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1000).to(dtype)
    selected = torch.tensor([1, 3], dtype=torch.long)
    return model, inputs, visual, selected


class PrunedInputsTest(unittest.TestCase):
    def assert_valid_result(self, model, inputs, visual, selected, result):
        ids = inputs["input_ids"]
        image_id = model.config.image_token_id
        original_image_positions = (ids[0] == image_id).nonzero(as_tuple=True)[0]
        chosen_positions = original_image_positions[selected]
        # Independent Python set construction rather than using the helper's mask.
        keep = [i for i, token in enumerate(ids[0].tolist())
                if token != image_id or i in set(chosen_positions.tolist())]
        keep = torch.tensor(keep, dtype=torch.long)
        merged = model.get_input_embeddings()(ids).clone()
        for visual_index, sequence_index in enumerate(original_image_positions.tolist()):
            merged[0, sequence_index] = visual[visual_index]
        actual = result.model_inputs
        self.assertEqual(set(actual), {"input_ids", "inputs_embeds", "attention_mask"})
        self.assertEqual(result.original_length, ids.shape[1])
        self.assertEqual(result.pruned_length, ids.shape[1] - visual.shape[0] + selected.numel())
        self.assertEqual(result.visual_count, visual.shape[0])
        self.assertEqual(result.retained_count, selected.numel())
        self.assertTrue(torch.equal(result.visual_positions, original_image_positions))
        self.assertTrue(torch.equal(result.keep_positions, keep))
        self.assertTrue(torch.equal(actual["input_ids"], ids[:, keep]))
        self.assertTrue(torch.equal(actual["inputs_embeds"], merged[:, keep]))
        self.assertTrue(torch.equal(actual["attention_mask"], inputs["attention_mask"][:, keep]))
        self.assertEqual(int((actual["input_ids"] == image_id).sum()), selected.numel())
        actual_visual_mask = actual["input_ids"] == image_id
        self.assertTrue(torch.equal(actual["inputs_embeds"][actual_visual_mask], visual[selected]))
        original_text = ids != image_id
        self.assertTrue(torch.equal(actual["input_ids"][~actual_visual_mask], ids[original_text]))
        self.assertTrue(torch.equal(actual["inputs_embeds"][~actual_visual_mask],
                                    model.get_input_embeddings()(ids)[original_text]))
        expected_positions = actual["attention_mask"].long().cumsum(dim=-1) - 1
        expected_positions.masked_fill_(actual["attention_mask"] == 0, 1)
        self.assertTrue(torch.equal(result.expected_position_ids, expected_positions))
        for name in ("exact_length", "placeholder_count", "selected_unique", "selected_valid",
                     "selected_sorted", "text_ids_preserved", "text_embeddings_preserved",
                     "text_attention_preserved", "visual_embeddings_preserved",
                     "visual_attention_preserved", "original_order_preserved",
                     "continuous_position_ids", "fresh_cache_inputs"):
            self.assertIs(result.checks[name], True, name)

    def test_physical_deletion_preserves_all_text_separator_and_special_tokens(self):
        model, inputs, visual, selected = fixture()
        result = build_pruned_inputs(model, inputs, visual, selected)
        self.assert_valid_result(model, inputs, visual, selected, result)
        self.assertEqual(result.keep_positions.tolist(), [0, 1, 3, 4, 6, 7, 8])
        self.assertEqual(result.model_inputs["input_ids"].tolist(), [[1, 2, 7, 3, 7, 4, 5]])

    def test_merged_embedding_noop_k_equals_n_is_exact(self):
        model, inputs, visual, _ = fixture()
        selected = torch.arange(visual.shape[0], dtype=torch.long)
        result = build_pruned_inputs(model, inputs, visual, selected)
        self.assert_valid_result(model, inputs, visual, selected, result)
        self.assertTrue(torch.equal(result.model_inputs["input_ids"], inputs["input_ids"]))
        self.assertTrue(torch.equal(result.keep_positions, torch.arange(inputs["input_ids"].shape[1])))
        self.assertEqual(result.original_length, result.pruned_length)

    def test_empty_selection_removes_all_visual_positions_only(self):
        model, inputs, visual, _ = fixture()
        selected = torch.empty(0, dtype=torch.long)
        result = build_pruned_inputs(model, inputs, visual, selected)
        self.assert_valid_result(model, inputs, visual, selected, result)
        self.assertEqual(result.model_inputs["input_ids"].tolist(), [[1, 2, 3, 4, 5]])

    def test_attention_mask_preserved_and_expected_positions_recomputed(self):
        model, inputs, visual, selected = fixture()
        # Left-padding is retained because it is not a visual position.
        inputs["attention_mask"][0, 0] = 0
        result = build_pruned_inputs(model, inputs, visual, selected)
        self.assert_valid_result(model, inputs, visual, selected, result)
        self.assertEqual(result.expected_position_ids.tolist(), [[1, 0, 1, 2, 3, 4, 5]])

    def test_processor_pixel_and_cache_fields_are_not_forwarded(self):
        model, inputs, visual, selected = fixture()
        inputs.update(pixel_values=torch.zeros(1, 2, 3, 4, 4),
                      pixel_attention_mask=torch.ones(1, 2, 4, 4),
                      past_key_values=object(), cache_position=torch.arange(9),
                      position_ids=torch.arange(9).unsqueeze(0))
        result = build_pruned_inputs(model, inputs, visual, selected)
        self.assert_valid_result(model, inputs, visual, selected, result)
        for key in ("pixel_values", "pixel_attention_mask", "past_key_values",
                    "cache_position", "position_ids"):
            self.assertNotIn(key, result.model_inputs)

    def test_inputs_and_embeddings_are_not_mutated(self):
        model, inputs, visual, selected = fixture()
        snapshot = {key: value.clone() for key, value in inputs.items()}
        before_visual, before_selected = visual.clone(), selected.clone()
        build_pruned_inputs(model, inputs, visual, selected)
        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], snapshot[key]))
        self.assertTrue(torch.equal(visual, before_visual))
        self.assertTrue(torch.equal(selected, before_selected))

    def test_float16_embeddings_are_preserved_exactly(self):
        model, inputs, visual, selected = fixture(torch.float16)
        result = build_pruned_inputs(model, inputs, visual, selected)
        self.assert_valid_result(model, inputs, visual, selected, result)
        self.assertEqual(result.model_inputs["inputs_embeds"].dtype, torch.float16)

    def test_dynamic_visual_counts_832_and_1088(self):
        for n in (832, 1088):
            for k in (64, 128, n):
                with self.subTest(n=n, k=k):
                    model = LookupModel()
                    # Different counts mimic 12+1 and 16+1 view layouts without
                    # any dependency on a hardcoded crop layout or image size.
                    first = n // 2
                    ids = torch.tensor([[1] + [7] * first + [2, 3] + [7] * (n - first) + [4]])
                    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                    visual = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + 1000
                    selected = torch.randperm(n, generator=torch.Generator().manual_seed(k))[:k].sort().values
                    result = build_pruned_inputs(model, inputs, visual, selected)
                    self.assert_valid_result(model, inputs, visual, selected, result)
                    self.assertEqual(result.pruned_length, k + 4)

    def test_selection_must_be_1d_integer_sorted_unique_and_in_range(self):
        model, inputs, visual, _ = fixture()
        invalid = [torch.tensor([[1, 3]]), torch.tensor([1.0, 3.0]),
                   torch.tensor([True, False]), torch.tensor([1, 1]),
                   torch.tensor([-1, 2]), torch.tensor([0, 4]), torch.tensor([3, 1])]
        for selected in invalid:
            with self.subTest(selected=selected), self.assertRaises((TypeError, ValueError)):
                build_pruned_inputs(model, inputs, visual, selected)

    def test_ids_and_attention_shapes_and_single_batch_required(self):
        for mode in ("ids_1d", "ids_3d", "batch_2", "mask_1d", "mask_wrong_length", "ids_float"):
            with self.subTest(mode=mode):
                model, inputs, visual, selected = fixture()
                if mode == "ids_1d":
                    inputs["input_ids"] = inputs["input_ids"][0]
                elif mode == "ids_3d":
                    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
                elif mode == "batch_2":
                    inputs = {name: value.repeat(2, 1) for name, value in inputs.items()}
                elif mode == "mask_1d":
                    inputs["attention_mask"] = inputs["attention_mask"][0]
                elif mode == "mask_wrong_length":
                    inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
                elif mode == "ids_float":
                    inputs["input_ids"] = inputs["input_ids"].float()
                with self.assertRaises((TypeError, ValueError)):
                    build_pruned_inputs(model, inputs, visual, selected)

    def test_projected_embedding_shapes_count_dimension_and_finiteness(self):
        model, inputs, visual, selected = fixture()
        invalid = [visual.flatten(), visual.unsqueeze(0), visual[:-1],
                   visual[:, :-1], visual.long(), visual.double(),
                   torch.full_like(visual, float("nan")), torch.full_like(visual, float("inf"))]
        for embeddings in invalid:
            with self.subTest(shape=embeddings.shape, dtype=embeddings.dtype), \
                    self.assertRaises((TypeError, ValueError)):
                build_pruned_inputs(model, inputs, embeddings, selected)

    def test_attention_must_be_binary(self):
        for invalid_value in (-1, 2):
            with self.subTest(invalid_value=invalid_value):
                model, inputs, visual, selected = fixture()
                inputs["attention_mask"][0, 1] = invalid_value
                with self.assertRaises((TypeError, ValueError)):
                    build_pruned_inputs(model, inputs, visual, selected)


if __name__ == "__main__":
    unittest.main()
