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

"""Read-only hooks for the cached Idefics3/Llama eager generation path.

This module imports no Transformers classes and never runs the model itself.
The expected input is one unpadded prompt. Positions use the shorter sequence's
contiguous coordinates, and every generate call must start with an empty cache.
"""

import operator

import torch


class GenerationAudit:
    """Observe one generate call and reject embedding/mask/cache regressions.

    Call ``report(number_of_generated_tokens)`` after generation. The final
    sampled token has not been forwarded, so the final cache length is
    ``prompt_length + number_of_generated_tokens - 1``.
    """

    def __init__(self, model, expected_inputs_embeds, expected_attention_mask,
                 capture_embeddings=False, expect_vision_calls=0):
        if (not isinstance(expected_inputs_embeds, torch.Tensor)
                or expected_inputs_embeds.ndim != 3
                or expected_inputs_embeds.shape[0] != 1
                or min(expected_inputs_embeds.shape[1:]) < 1
                or not expected_inputs_embeds.is_floating_point()):
            raise ValueError("Expected embeddings must have floating shape [1, L, D] with L,D>0")
        self.length, self.dimension = expected_inputs_embeds.shape[1:]
        if (not isinstance(expected_attention_mask, torch.Tensor)
                or tuple(expected_attention_mask.shape) != (1, self.length)
                or not torch.equal(expected_attention_mask, torch.ones_like(expected_attention_mask))):
            raise ValueError("GenerationAudit requires one unpadded [1, L] attention mask")
        if isinstance(expect_vision_calls, bool):
            raise TypeError("expect_vision_calls must be an integer")
        self.expected_vision_calls = operator.index(expect_vision_calls)
        if self.expected_vision_calls < 0:
            raise ValueError("expect_vision_calls must be nonnegative")
        self.model = model
        self.expected_embeddings = expected_inputs_embeds.detach().clone()
        self.expected_mask = expected_attention_mask.detach().clone()
        self.capture_embeddings = capture_embeddings
        self.prefill_embeddings = None
        self.steps = []
        self.vision_calls = 0
        self.connector_calls = 0
        self._handles = []
        self._used = False
        self._active = None
        self._last_cache_id = None
        self._failure = None

    def _check(self, condition, message):
        if not condition:
            self._failure = message
            raise RuntimeError(f"Generation audit failed: {message}")

    @staticmethod
    def _argument(args, kwargs, name, position):
        return kwargs[name] if name in kwargs else args[position] if len(args) > position else None

    def _cache_length(self, cache, allow_none=False):
        if cache is None and allow_none:
            return 0
        self._check(cache is not None and callable(getattr(cache, "get_seq_length", None)),
                    "Expected a Cache object exposing get_seq_length()")
        # Snapshot now: the same cache object is mutated during every forward.
        return operator.index(cache.get_seq_length())

    def _text_pre(self, module, args, kwargs):
        self._check(self._active is None, "Nested or unfinished text-model forward")
        step = len(self.steps)
        q = self.length if step == 0 else 1
        past_length = 0 if step == 0 else self.length + step - 1
        embeddings = self._argument(args, kwargs, "inputs_embeds", 4)
        mask = self._argument(args, kwargs, "attention_mask", 1)
        cache = self._argument(args, kwargs, "past_key_values", 3)
        self._check(isinstance(embeddings, torch.Tensor)
                    and tuple(embeddings.shape) == (1, q, self.dimension),
                    f"Text input shape must be [1, {q}, {self.dimension}] at step {step}")
        self._check(isinstance(mask, torch.Tensor) and tuple(mask.shape) == (1, self.length + step),
                    f"2D attention-mask length mismatch at step {step}")
        self._check(torch.equal(mask[:, :self.length], self.expected_mask.to(mask.device))
                    and torch.equal(mask[:, self.length:], torch.ones_like(mask[:, self.length:])),
                    f"Original attention mask or appended ones changed at step {step}")
        cache_before = self._cache_length(cache, allow_none=step == 0)
        self._check(cache_before == past_length,
                    f"Cache before step {step}: expected {past_length}, got {cache_before}")
        if step:
            self._check(id(cache) == self._last_cache_id, "Cache object changed during generation")
        if step == 0:
            self._check(embeddings.dtype == self.expected_embeddings.dtype
                        and embeddings.device == self.expected_embeddings.device
                        and torch.equal(embeddings, self.expected_embeddings),
                        "Actual LLM prefill embeddings differ from direct-insertion embeddings")
            if self.capture_embeddings:
                self.prefill_embeddings = embeddings.detach().cpu().clone()
        self._active = dict(step=step, phase="prefill" if step == 0 else "decode",
                            input_length=q, attention_mask_length=mask.shape[1],
                            cache_before=cache_before, _cache_id=id(cache) if cache is not None else None,
                            _rotary_calls=0, _layer_calls=0)

    def _rotary_pre(self, module, args, kwargs):
        self._check(self._active is not None, "RoPE called outside the observed LLM forward")
        step = self._active
        position_ids = self._argument(args, kwargs, "position_ids", 1)
        q, start = step["input_length"], step["cache_before"]
        self._check(isinstance(position_ids, torch.Tensor)
                    and tuple(position_ids.shape) == (1, q)
                    and not position_ids.is_floating_point() and position_ids.dtype != torch.bool,
                    "Actual RoPE position IDs must be integer [1, query_length]")
        expected = torch.arange(start, start + q, device=position_ids.device,
                                dtype=position_ids.dtype).unsqueeze(0)
        self._check(torch.equal(position_ids, expected),
                    f"Actual RoPE positions are not contiguous/reindexed at step {step['step']}")
        step["_rotary_calls"] += 1
        self._check(step["_rotary_calls"] == 1, "Multiple RoPE calls in one LLM forward")
        step.update(position_start=start, position_end=start + q - 1)

    def _layer_pre(self, module, args, kwargs):
        self._check(self._active is not None, "Decoder layer called outside the observed LLM forward")
        step = self._active
        q, before = step["input_length"], step["cache_before"]
        mask = self._argument(args, kwargs, "attention_mask", 1)
        key_length = before + q
        self._check(isinstance(mask, torch.Tensor)
                    and tuple(mask.shape) == (1, 1, q, key_length)
                    and mask.is_floating_point(),
                    f"Actual eager causal-mask shape/dtype mismatch at step {step['step']}")
        query = torch.arange(before, before + q, device=mask.device).unsqueeze(1)
        key = torch.arange(key_length, device=mask.device).unsqueeze(0)
        allowed = key <= query
        values = mask[0, 0]
        self._check((values[allowed] == 0).all().item(), "Causal mask hid a permitted key")
        blocked = values[~allowed]
        self._check((blocked <= torch.finfo(mask.dtype).min / 2).all().item(),
                    "Causal mask failed to block a future key")
        step["_layer_calls"] += 1
        self._check(step["_layer_calls"] == 1, "First decoder layer called multiple times")
        step.update(causal_mask_shape=list(mask.shape), causal_mask_valid=True)

    def _text_post(self, module, args, kwargs, output):
        self._check(self._active is not None, "Text-model post hook has no matching pre hook")
        step = self._active
        self._check(step["_rotary_calls"] == 1 and step["_layer_calls"] == 1,
                    "Actual RoPE/decoder-mask hooks were not both observed")
        cache = getattr(output, "past_key_values", None)
        cache_after = self._cache_length(cache)
        self._check(cache_after == step["cache_before"] + step["input_length"],
                    f"Cache did not grow by query length at step {step['step']}")
        if step["_cache_id"] is not None:
            self._check(id(cache) == step["_cache_id"], "Forward returned a different cache object")
        self._last_cache_id = id(cache)
        step.update(cache_after=cache_after, cache_type=type(cache).__name__)
        for key in ("_cache_id", "_rotary_calls", "_layer_calls"):
            del step[key]
        self.steps.append(step)
        self._active = None

    def _vision_pre(self, module, args, kwargs):
        self.vision_calls += 1
        self._check(self.vision_calls <= self.expected_vision_calls,
                    "Unexpected vision-encoder execution during generate()")

    def _connector_pre(self, module, args, kwargs):
        self.connector_calls += 1
        self._check(self.connector_calls <= self.expected_vision_calls,
                    "Unexpected connector execution during generate()")

    def _validate_completed(self):
        self._check(self._failure is None, self._failure or "A previous hook failed")
        self._check(self._active is None and bool(self.steps), "Generation did not complete an LLM forward")
        self._check(self.vision_calls == self.expected_vision_calls,
                    f"Expected {self.expected_vision_calls} vision calls, got {self.vision_calls}")
        self._check(self.connector_calls == self.expected_vision_calls,
                    f"Expected {self.expected_vision_calls} connector calls, got {self.connector_calls}")

    def __enter__(self):
        self._check(not self._used, "GenerationAudit instances can observe only one generate call")
        self._used = True
        try:
            body = self.model.model
            text = body.text_model
            # Record each handle immediately so partial registration also cleans
            # up if an unsupported model is missing one of the later modules.
            self._handles.append(text.register_forward_pre_hook(self._text_pre, with_kwargs=True))
            self._handles.append(text.register_forward_hook(self._text_post, with_kwargs=True))
            self._handles.append(text.rotary_emb.register_forward_pre_hook(self._rotary_pre, with_kwargs=True))
            self._handles.append(text.layers[0].register_forward_pre_hook(self._layer_pre, with_kwargs=True))
            self._handles.append(body.vision_model.register_forward_pre_hook(self._vision_pre, with_kwargs=True))
            self._handles.append(body.connector.register_forward_pre_hook(self._connector_pre, with_kwargs=True))
        except Exception:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self._validate_completed()
        finally:
            self._remove_hooks()
        return False

    def report(self, generated_token_count):
        if isinstance(generated_token_count, bool):
            raise TypeError("generated_token_count must be an integer")
        generated_token_count = operator.index(generated_token_count)
        self._check(generated_token_count >= 1, "Expected at least one generated token")
        self._validate_completed()
        self._check(len(self.steps) == generated_token_count,
                    "Forward count differs from generated-token count")
        final_length = self.length + generated_token_count - 1
        self._check(self.steps[-1]["cache_after"] == final_length,
                    "Final cache length must equal prompt length + generated count - 1")
        return dict(
            **{"pass": True}, actual_prefill_length=self.steps[0]["input_length"],
            prefill_embeddings_exact=True, attention_mask_preserved=True,
            position_ids_contiguous=True, fresh_cache=self.steps[0]["cache_before"] == 0,
            forward_calls=len(self.steps), generated_token_count=generated_token_count,
            final_cache_length=self.steps[-1]["cache_after"],
            expected_final_cache_length=final_length,
            expected_vision_calls=self.expected_vision_calls,
            vision_calls=self.vision_calls, connector_calls=self.connector_calls,
            steps=[dict(step) for step in self.steps],
        )
