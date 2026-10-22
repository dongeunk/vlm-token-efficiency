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

"""Physically shorten a single VLM prompt after direct visual insertion.

This helper follows the validated examples/run_random_pruning.py path without
changing a selector or invoking feature extraction/generation. Every non-image
position survives. Selected visual embeddings replace their original image
placeholders, and unselected placeholders are removed from IDs, embeddings, and
attention together. No pixels, image features, cache, or position IDs are sent
to generate(); its fresh preparation derives positions from the shorter mask.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import operator

import torch


@dataclass(frozen=True)
class PrunedInputs:
    model_inputs: dict[str, torch.Tensor]
    original_length: int
    pruned_length: int
    visual_count: int
    retained_count: int
    keep_positions: torch.Tensor
    visual_positions: torch.Tensor
    checks: dict[str, bool]
    expected_position_ids: torch.Tensor


def _require_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


@torch.no_grad()
def build_pruned_inputs(model, inputs, visual_embeddings, selected_indices) -> PrunedInputs:
    """Insert [N,D] visual features and retain exactly the sorted K indices.

    Inputs must contain one nonempty LongTensor input_ids row and a same-shape
    binary attention_mask. Features must be finite floating point and exactly
    match the embedding layer's dimension, dtype, and device: no silent lossy
    cast is performed. Selection is a strictly ascending, unique 1D LongTensor
    in [0,N); CPU indices are accepted and copied to the prompt's device.

    ``keep_positions`` and ``visual_positions`` address the original sequence.
    ``expected_position_ids`` is audit-only: mask.cumsum(-1)-1 with masked slots
    set to 1, matching generate's convention; it is never forwarded to the model.
    K=0 is supported if at least one nonvisual sequence position survives.
    K=N additionally checks equality to the full manually merged prompt, not an
    independently captured ordinary model forward (the caller may audit that).
    The caller owns a fresh generation call and its inference/eval settings.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping containing input_ids and attention_mask")
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise ValueError("inputs must contain input_ids and attention_mask")
    input_ids = _require_tensor(inputs["input_ids"], "input_ids")
    attention = _require_tensor(inputs["attention_mask"], "attention_mask")
    features = _require_tensor(visual_embeddings, "visual_embeddings")
    selected = _require_tensor(selected_indices, "selected_indices")
    if input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a nonempty [1,L] LongTensor")
    if (input_ids < 0).any().item():
        raise ValueError("input_ids must contain nonnegative vocabulary indices")
    if attention.shape != input_ids.shape or attention.device != input_ids.device:
        raise ValueError("attention_mask must match input_ids shape and device")
    if attention.is_complex() or not torch.isfinite(attention).all().item():
        raise ValueError("attention_mask must be finite, real, and binary")
    if not ((attention == 0) | (attention == 1)).all().item():
        raise ValueError("attention_mask must contain only zero and one")
    if not (attention == 1).any().item():
        raise ValueError("attention_mask must contain at least one attended position")
    if features.ndim != 2 or not features.is_floating_point() or min(features.shape) < 1:
        raise ValueError("visual_embeddings must be a nonempty floating tensor [N,D]")
    if not torch.isfinite(features).all().item():
        raise ValueError("visual_embeddings must contain only finite values")
    if features.device != input_ids.device:
        raise ValueError("visual_embeddings and input_ids must be on the same device")
    n, dimension = features.shape
    if selected.dtype != torch.long or selected.ndim != 1:
        raise ValueError("selected_indices must be a 1D LongTensor")
    k = selected.numel()
    selected_valid = bool(((selected >= 0) & (selected < n)).all().item())
    selected_unique = selected.unique().numel() == k
    selected_sorted = bool((selected[1:] > selected[:-1]).all().item())
    if not selected_valid or not selected_unique or not selected_sorted:
        raise ValueError("selected_indices must be unique, sorted, and valid in [0,N)")
    selected = selected.to(device=input_ids.device)

    token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if isinstance(token_id, bool):
        raise ValueError("model.config.image_token_id must be a nonnegative integer")
    try:
        token_id = operator.index(token_id)
    except TypeError as error:
        raise ValueError("model.config.image_token_id must be a nonnegative integer") from error
    if token_id < 0:
        raise ValueError("model.config.image_token_id must be a nonnegative integer")
    image_mask = input_ids == token_id
    visual_positions = image_mask[0].nonzero(as_tuple=True)[0]
    if visual_positions.numel() != n:
        raise ValueError(f"Image placeholder count {visual_positions.numel()} differs from N={n}")

    embedding_layer = model.get_input_embeddings()
    weight = getattr(embedding_layer, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        if input_ids.max().item() >= weight.shape[0]:
            raise ValueError("input_ids contain an index outside the embedding vocabulary")
        if weight.device != input_ids.device:
            raise ValueError("Embedding layer and input_ids must be on the same device")
    text_embeddings = embedding_layer(input_ids)
    if (not isinstance(text_embeddings, torch.Tensor) or text_embeddings.ndim != 3
            or tuple(text_embeddings.shape) != (1, input_ids.shape[1], dimension)
            or not text_embeddings.is_floating_point()):
        raise ValueError("Embedding layer output must have shape [1,L,D] matching visual embeddings")
    if text_embeddings.device != features.device or text_embeddings.dtype != features.dtype:
        raise ValueError("Visual embeddings must exactly match the text embeddings' dtype and device")
    if not torch.isfinite(text_embeddings).all().item():
        raise ValueError("Text embeddings must contain only finite values")

    full_embeddings = text_embeddings.clone()
    full_embeddings[image_mask] = features
    keep = ~image_mask[0]
    keep = keep.clone()
    selected_original_positions = visual_positions.index_select(0, selected)
    keep[selected_original_positions] = True
    keep_positions = keep.nonzero(as_tuple=True)[0]
    original_length, pruned_length = input_ids.shape[1], keep_positions.numel()
    if pruned_length == 0:
        raise ValueError("Pruning cannot leave an empty LLM sequence")
    pruned_ids = input_ids.index_select(1, keep_positions)
    pruned_embeddings = full_embeddings.index_select(1, keep_positions)
    pruned_attention = attention.index_select(1, keep_positions)
    if not (pruned_attention == 1).any().item():
        raise ValueError("Pruning cannot leave a sequence with no attended positions")
    pruned_image_mask = pruned_ids == token_id
    expected_position_ids = pruned_attention.long().cumsum(dim=-1) - 1
    expected_position_ids.masked_fill_(pruned_attention == 0, 1)
    attended_positions = expected_position_ids[pruned_attention != 0]

    model_inputs = dict(input_ids=pruned_ids, inputs_embeds=pruned_embeddings, attention_mask=pruned_attention)
    checks = dict(
        exact_length=pruned_length == original_length - (n - k),
        placeholder_count=int(pruned_image_mask.sum().item()) == k,
        selected_unique=selected_unique, selected_valid=selected_valid, selected_sorted=selected_sorted,
        text_ids_preserved=torch.equal(pruned_ids[~pruned_image_mask], input_ids[~image_mask]),
        text_embeddings_preserved=torch.equal(pruned_embeddings[~pruned_image_mask], text_embeddings[~image_mask]),
        text_attention_preserved=torch.equal(pruned_attention[~pruned_image_mask], attention[~image_mask]),
        visual_embeddings_preserved=torch.equal(pruned_embeddings[pruned_image_mask], features.index_select(0, selected)),
        visual_attention_preserved=torch.equal(pruned_attention[pruned_image_mask], attention[0].index_select(0, selected_original_positions)),
        original_order_preserved=bool((keep_positions[1:] > keep_positions[:-1]).all().item()),
        continuous_position_ids=torch.equal(
            attended_positions, torch.arange(attended_positions.numel(), device=input_ids.device)),
        fresh_cache_inputs=set(model_inputs) == {"input_ids", "inputs_embeds", "attention_mask"},
    )
    if k == n:
        checks.update(
            noop_input_ids_equal=torch.equal(pruned_ids, input_ids),
            noop_embeddings_equal=torch.equal(pruned_embeddings, full_embeddings),
            noop_attention_equal=torch.equal(pruned_attention, attention),
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Physical pruning preservation checks failed: {failed}")
    return PrunedInputs(
        model_inputs=model_inputs, original_length=original_length, pruned_length=pruned_length,
        visual_count=n, retained_count=k, keep_positions=keep_positions,
        visual_positions=visual_positions, checks=checks, expected_position_ids=expected_position_ids,
    )
