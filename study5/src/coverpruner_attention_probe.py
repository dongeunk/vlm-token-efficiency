"""SmolVLM first-attention Q/K-only probe. Study4 adaptation, not exact paper code.
Native full-prefix softmax is primary; visual-only is an explicit diagnostic.
No V/output/FFN/cache, generation, pruning or spatial selection in this module.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv, create_causal_mask,
)
from coverpruner_coverage import demand_weights


@dataclass(frozen=True)
class InputPositions:
    visual: tuple[int, ...]
    instruction: tuple[int, ...]
    original_body_span: tuple[int, int]
    expanded_body_span: tuple[int, int]
    sequence_length: int


def instruction_span(processor, question):
    """Trace a user-body slot through native template, never search question IDs."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question body must be nonempty")
    marker = "STUDY4_UNIQUE_USER_BODY_SLOT_7D2A"
    if marker in question or any(t in question for t in processor.tokenizer.all_special_tokens):
        raise ValueError("Question must not contain template marker or special tokens")
    def render(body):
        return processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": body}]}], add_generation_prompt=True)
    marked, actual = render(marker), render(question)
    if marked.count(marker) != 1:
        raise ValueError("Template does not preserve one explicit user body slot")
    prefix, suffix = marked.split(marker)
    if actual != prefix + question + suffix:
        raise ValueError("Template body rendering is not traceable")
    return actual, (len(prefix), len(prefix) + len(question))


def remap_body_span(span, replacements):
    start, end = span
    if start < 0 or end <= start:
        raise ValueError("Invalid body span")
    shift, body_shift, previous_end = 0, 0, 0
    for replacement in replacements:
        left, right = map(int, replacement["span"])
        new_left, new_right = map(int, replacement["new_span"])
        if left < previous_end or right <= left or new_left != left + shift or new_right < new_left:
            raise ValueError("Inconsistent processor replacement offsets")
        if left < end and right > start:
            raise ValueError("Image replacement overlaps user body")
        delta = (new_right - new_left) - (right - left)
        if right <= start:
            body_shift += delta
        shift += delta
        previous_end = right
    return start + body_shift, end + body_shift

def extract_positions(ids, offsets, validity, visual_id, special_ids, body_span):
    if not (len(ids) == len(offsets) == len(validity)):
        raise ValueError("Token metadata length mismatch")
    start, end = body_span
    if start < 0 or end <= start:
        raise ValueError("Invalid expanded body span")
    visual, queries = [], []
    for index, (token, pair, valid) in enumerate(zip(ids, offsets, validity)):
        a, b = map(int, pair)
        if b < a or a < 0 or valid not in (0, 1):
            raise ValueError("Invalid offset/validity")
        if token == visual_id:
            if not valid:
                raise ValueError("Padded visual token")
            visual.append(index)
        if b > start and a < end and b > a:
            if a < start or b > end:
                raise ValueError("Instruction boundary straddles a tokenizer token")
            if not valid or token in special_ids or token == visual_id:
                raise ValueError("Instruction includes padding/special/visual token")
            queries.append(index)
    if not visual or not queries or set(visual).intersection(queries):
        raise ValueError("Missing or overlapping visual/instruction positions")
    return tuple(visual), tuple(queries)


def prepare_processor_input(processor, image, question):
    if not processor.tokenizer.is_fast:
        raise ValueError("Fast tokenizer offsets required")
    prompt, body = instruction_span(processor, question)
    raw = processor(text=prompt, images=[image], return_tensors=None,
                    return_offsets_mapping=True, return_text_replacement_offsets=True)
    replacements = raw["text_replacement_offsets"][0]
    expanded = remap_body_span(body, replacements)
    ids, offsets, valid = raw["input_ids"][0], raw["offset_mapping"][0], raw["attention_mask"][0]
    visual, instruction = extract_positions(ids, offsets, valid, processor.image_token_id,
                                            set(processor.tokenizer.all_special_ids), expanded)
    covered = {i for t in instruction for i in range(*offsets[t])}
    if any(not char.isspace() and expanded[0] + i not in covered for i, char in enumerate(question)):
        raise ValueError("Instruction character coverage incomplete")
    inputs = {name: torch.as_tensor(np.asarray(raw[name])) for name in
              ("input_ids", "attention_mask", "pixel_values", "pixel_attention_mask")}
    inputs["input_ids"] = inputs["input_ids"].long()
    inputs["attention_mask"] = inputs["attention_mask"].long()
    return inputs, InputPositions(visual, instruction, body, expanded, len(ids))


def _indices(values, length, device):
    if isinstance(values, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in values):
        raise ValueError("Boolean indices prohibited")
    a = torch.as_tensor(values, device=device)
    if a.ndim != 1 or a.numel() == 0 or a.dtype not in (torch.int32, torch.int64):
        raise ValueError("Require nonempty integer indices")
    if a.unique().numel() != a.numel() or a.min() < 0 or a.max() >= length:
        raise ValueError("Duplicate/out-of-range indices")
    return a.long()


def native_layout(config, embeddings, validity, position_ids=None):
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or not embeddings.is_floating_point():
        raise ValueError("Only batch1 [1,L,D] embeddings supported")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    if validity.shape != embeddings.shape[:2] or not ((validity == 0) | (validity == 1)).all():
        raise ValueError("Invalid token validity mask")
    if position_ids is None:
        position_ids = torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0)
    if position_ids.shape != validity.shape or position_ids.dtype != torch.int64 or (position_ids < 0).any():
        raise ValueError("Invalid native position_ids")
    mask = create_causal_mask(config=config, inputs_embeds=embeddings,
        attention_mask=validity, past_key_values=None, position_ids=position_ids)
    if mask is None:
        raise ValueError("Explicit eager native causal mask required")
    return position_ids, mask


@dataclass
class AttentionSlice:
    scaled_logits: torch.Tensor
    probabilities: torch.Tensor
    visual_probabilities: torch.Tensor
    allowed_visual_counts: tuple[int, ...]
    query_mask: torch.Tensor


def query_attention(q, k, mask, instruction, visual, scaling, *, domain="full_prefix"):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("Q/K must be batch1 head-major tensors")
    if any(x == 0 for x in (*q.shape, *k.shape)) or not q.is_floating_point() or q.shape[-2:] != k.shape[-2:] or q.shape[1] % k.shape[1] or q.dtype != k.dtype:
        raise ValueError("Invalid Q/K head, length, dimension or dtype")
    if not (torch.isfinite(q).all() and torch.isfinite(k).all()):
        raise ValueError("Nonfinite Q/K")
    if domain not in ("full_prefix", "visual_only") or not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Invalid domain/scaling")
    length = q.shape[-2]
    tq, vk = _indices(instruction, length, q.device), _indices(visual, length, q.device)
    if torch.isin(tq, vk).any():
        raise ValueError("Query and visual indices overlap")
    if mask.shape != (1, 1, length, length) or not mask.is_floating_point():
        raise ValueError("Require additive native [1,1,L,L] mask")
    good_mask = (mask == 0) | torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min / 2)
    if not good_mask.all():
        raise ValueError("Unexpected additive mask entries")
    query_mask = mask.index_select(2, tq)
    allowed = query_mask.index_select(3, vk) == 0
    counts = allowed[0, 0].sum(-1)
    if (counts == 0).any():
        raise ValueError("Instruction cannot causally access any visual key")
    logits = torch.matmul(q.index_select(2, tq), repeat_kv(k, q.shape[1] // k.shape[1]).transpose(2, 3)) * scaling
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite scaled logits")
    if domain == "full_prefix":
        probabilities = torch.softmax(logits + query_mask, dim=-1, dtype=torch.float32).to(q.dtype)
    else:
        selected = torch.softmax((logits + query_mask).index_select(3, vk), dim=-1,
                                 dtype=torch.float32).to(q.dtype)
        probabilities = torch.zeros_like(logits)
        probabilities.index_copy_(3, vk, selected)
    visual_probabilities = probabilities.index_select(3, vk)
    if not torch.isfinite(probabilities).all() or (visual_probabilities.masked_select(~allowed.expand_as(visual_probabilities)) != 0).any():
        raise ValueError("Invalid softmax or masked-key probability")
    return AttentionSlice(logits, probabilities, visual_probabilities,
                          tuple(int(x) for x in counts.cpu()), query_mask)


def relevance_from_attention(visual_probabilities):
    if visual_probabilities.ndim != 4 or any(n == 0 for n in visual_probabilities.shape):
        raise ValueError("Expected nonempty [1,H,Tq,Nvisual] probabilities")
    if visual_probabilities.shape[0] != 1 or not torch.isfinite(visual_probabilities).all() or (visual_probabilities < 0).any():
        raise ValueError("Invalid visual probabilities")
    if not visual_probabilities.is_floating_point() or not torch.isfinite(visual_probabilities).all() or ((visual_probabilities < 0) | (visual_probabilities > 1)).any():
        raise ValueError("Invalid visual probabilities")
    return visual_probabilities.float().mean(dim=1).mean(dim=1)[0]


@dataclass
class ProbeResult:
    normalized_input: torch.Tensor
    q_pre_rope: torch.Tensor
    k_pre_rope: torch.Tensor
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    position_embeddings: tuple
    attention: AttentionSlice
    relevance: np.ndarray
    weights: np.ndarray


@torch.no_grad()
def first_attention_probe(layer, rotary, embeddings, validity, position_ids, mask,
                          instruction, visual, *, domain="full_prefix"):
    if layer.training:
        raise ValueError("Probe requires eval mode")
    n, length, _ = embeddings.shape
    if n != 1 or not torch.isfinite(embeddings).all() or validity.shape != (1, length):
        raise ValueError("Only batch1 supported")
    tq = _indices(instruction, length, embeddings.device)
    vk = _indices(visual, length, embeddings.device)
    if not (validity[0, tq].bool().all() and validity[0, vk].bool().all()):
        raise ValueError("Invalid instruction/visual padding")
    if position_ids.shape != (1, length) or position_ids.dtype != torch.int64 or (position_ids < 0).any():
        raise ValueError("Invalid position IDs")
    normalized = layer.input_layernorm(embeddings)
    attention = layer.self_attn
    shape = (1, length, -1, attention.head_dim)
    q = attention.q_proj(normalized).view(shape).transpose(1, 2)
    k = attention.k_proj(normalized).view(shape).transpose(1, 2)
    cos_sin = rotary(embeddings, position_ids=position_ids)
    qr, kr = apply_rotary_pos_emb(q, k, *cos_sin)
    result = query_attention(qr, kr, mask, instruction, visual, attention.scaling, domain=domain)
    r = relevance_from_attention(result.visual_probabilities).cpu().numpy().copy()
    return ProbeResult(normalized, q, k, qr, kr, cos_sin, result, r, demand_weights(r))