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

"""CPU-only audit checks: no model weights, Transformers, images or downloads."""

import json
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.inference.generation_audit import GenerationAudit


class FakeCache:
    def __init__(self, length=0):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeRotary(nn.Module):
    def forward(self, states, position_ids):
        return states


class FakeLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_emb = FakeRotary()
        self.layers = nn.ModuleList([FakeLayer()])
        self.position_offset = 0
        self.broken_future_mask = False
        self.cache_growth_error = 0

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=True):
        cache = past_key_values if past_key_values is not None else FakeCache()
        before, q = cache.get_seq_length(), inputs_embeds.shape[1]
        position_ids = (torch.arange(before, before + q).unsqueeze(0)
                        if position_ids is None else position_ids)
        self.rotary_emb(inputs_embeds, position_ids=position_ids + self.position_offset)
        allowed = torch.arange(before + q).unsqueeze(0) <= torch.arange(before, before + q).unsqueeze(1)
        mask = torch.where(allowed, 0., torch.finfo(inputs_embeds.dtype).min).to(inputs_embeds.dtype)[None, None]
        if self.broken_future_mask:
            mask = torch.zeros_like(mask)
        states = self.layers[0](inputs_embeds, attention_mask=mask)
        cache.length += q + self.cache_growth_error
        return SimpleNamespace(last_hidden_state=states, past_key_values=cache)


class FakeBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = FakeText()
        self.vision_model = nn.Identity()
        self.connector = nn.Identity()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBody()


class GenerationAuditTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.embeddings = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        self.mask = torch.ones(1, 4, dtype=torch.long)

    def run_forwards(self, count=3, cache=None, embeddings=None):
        cache = FakeCache() if cache is None else cache
        text = self.model.model.text_model
        embeddings = self.embeddings if embeddings is None else embeddings
        for step in range(count):
            states = embeddings if step == 0 else torch.full((1, 1, 3), 100. + step)
            text(inputs_embeds=states, attention_mask=torch.ones(1, 4 + step, dtype=torch.long),
                 position_ids=None, past_key_values=cache, use_cache=True)
        return cache

    def hook_count(self):
        return sum(len(module._forward_pre_hooks) + len(module._forward_hooks)
                   for module in self.model.modules())

    def test_prefill_decode_and_final_cache_off_by_one(self):
        with GenerationAudit(self.model, self.embeddings, self.mask, capture_embeddings=True) as audit:
            cache = self.run_forwards(3)
        report = audit.report(3)
        self.assertTrue(report["pass"])
        self.assertEqual(report["actual_prefill_length"], 4)
        self.assertEqual(report["forward_calls"], 3)
        self.assertEqual((report["final_cache_length"], report["expected_final_cache_length"]), (6, 6))
        self.assertEqual(cache.length, 6)  # not prompt + output_tokens = 7
        self.assertEqual([step["cache_before"] for step in report["steps"]], [0, 4, 5])
        self.assertEqual([step["cache_after"] for step in report["steps"]], [4, 5, 6])
        self.assertEqual([step["position_start"] for step in report["steps"]], [0, 4, 5])
        self.assertEqual([step["causal_mask_shape"] for step in report["steps"]],
                         [[1, 1, 4, 4], [1, 1, 1, 5], [1, 1, 1, 6]])
        self.assertTrue(torch.equal(audit.prefill_embeddings, self.embeddings))
        self.assertEqual(audit.prefill_embeddings.device.type, "cpu")
        self.assertNotEqual(audit.prefill_embeddings.data_ptr(), self.embeddings.data_ptr())
        self.assertEqual(self.hook_count(), 0)
        json.dumps(report)

    def test_first_token_eos_requires_no_decode_forward(self):
        with GenerationAudit(self.model, self.embeddings, self.mask) as audit:
            self.run_forwards(1)
        report = audit.report(1)
        self.assertEqual(report["final_cache_length"], 4)
        self.assertIsNone(audit.prefill_embeddings)

    def test_report_rejects_incorrect_output_token_count(self):
        with GenerationAudit(self.model, self.embeddings, self.mask) as audit:
            self.run_forwards(3)
        with self.assertRaisesRegex(RuntimeError, "Forward count"):
            audit.report(4)

    def test_prefill_insertion_mismatch_is_detected_and_hooks_removed(self):
        changed = self.embeddings.clone()
        changed[0, 1, 2] += 1
        with self.assertRaisesRegex(RuntimeError, "prefill embeddings differ"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.run_forwards(1, embeddings=changed)
        self.assertEqual(self.hook_count(), 0)

    def test_nonempty_cache_cannot_leak_from_another_generate(self):
        with self.assertRaisesRegex(RuntimeError, "expected 0, got 8"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.run_forwards(1, cache=FakeCache(8))
        self.assertEqual(self.hook_count(), 0)

    def test_cache_must_grow_by_exact_query_length(self):
        self.model.model.text_model.cache_growth_error = 1
        with self.assertRaisesRegex(RuntimeError, "grow by query length"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.run_forwards(1)

    def test_actual_rotary_positions_must_be_reindexed(self):
        self.model.model.text_model.position_offset = 20
        with self.assertRaisesRegex(RuntimeError, "positions are not contiguous"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.run_forwards(1)

    def test_changed_attention_mask_is_rejected(self):
        broken = self.mask.clone()
        broken[0, 2] = 0
        with self.assertRaisesRegex(RuntimeError, "attention mask or appended ones"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.model.model.text_model(inputs_embeds=self.embeddings, attention_mask=broken,
                                           past_key_values=FakeCache())

    def test_decode_attention_mask_must_include_whole_prefix_and_one_new_one(self):
        with self.assertRaisesRegex(RuntimeError, "attention-mask length"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                cache = self.run_forwards(1)
                self.model.model.text_model(inputs_embeds=torch.zeros(1, 1, 3), attention_mask=self.mask,
                                           past_key_values=cache)

    def test_actual_causal_mask_cannot_expose_future_tokens(self):
        self.model.model.text_model.broken_future_mask = True
        with self.assertRaisesRegex(RuntimeError, "future key"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.run_forwards(1)

    def test_direct_generation_must_not_rerun_vision(self):
        with self.assertRaisesRegex(RuntimeError, "vision-encoder execution"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.model.model.vision_model(torch.ones(1))
        self.assertEqual(self.hook_count(), 0)

    def test_baseline_runs_vision_and_connector_once(self):
        with GenerationAudit(self.model, self.embeddings, self.mask, expect_vision_calls=1) as audit:
            pixels = self.model.model.vision_model(torch.ones(1))
            self.model.model.connector(pixels)
            self.run_forwards(2)
        report = audit.report(2)
        self.assertEqual((report["vision_calls"], report["connector_calls"]), (1, 1))

    def test_missing_expected_connector_call_fails_on_exit(self):
        with self.assertRaisesRegex(RuntimeError, "Expected 1 connector calls"):
            with GenerationAudit(self.model, self.embeddings, self.mask, expect_vision_calls=1):
                self.model.model.vision_model(torch.ones(1))
                self.run_forwards(1)
        self.assertEqual(self.hook_count(), 0)

    def test_no_forward_and_external_exception_always_remove_hooks(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                pass
        self.assertEqual(self.hook_count(), 0)
        with self.assertRaisesRegex(ValueError, "external failure"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                raise ValueError("external failure")
        self.assertEqual(self.hook_count(), 0)

    def test_wrong_query_length_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Text input shape"):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.run_forwards(1, embeddings=self.embeddings[:, :-1])

    def test_partial_registration_failure_removes_already_registered_hooks(self):
        self.model.model.connector = None
        with self.assertRaises(AttributeError):
            with GenerationAudit(self.model, self.embeddings, self.mask):
                self.fail("Unsupported model should fail before entering the context")
        self.assertEqual(self.hook_count(), 0)

    def test_constructor_rejects_padding_and_batched_prompts(self):
        for embeddings, mask in ((self.embeddings.repeat(2, 1, 1), self.mask.repeat(2, 1)),
                                 (self.embeddings, torch.tensor([[1, 1, 0, 1]]))):
            with self.subTest(shape=embeddings.shape), self.assertRaises(ValueError):
                GenerationAudit(self.model, embeddings, mask)


if __name__ == "__main__":
    unittest.main()
