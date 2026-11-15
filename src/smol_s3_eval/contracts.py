# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# See LICENSE for terms. Publication adds comments only; experimental body unchanged.

"""Pure-Python sequence/cache contracts for later S3-B runtime observations.

These validate metadata and synthetic lists, not HF tensors or a decoder.
"""
from __future__ import annotations

import math

IMAGE_TOKEN = 49190
EOS_TOKEN = 49279
LAYERS = 30


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def index_list(values, size, name, count=None):
    if not isinstance(values, (list, tuple)) or any(type(x) is not int for x in values):
        raise ValueError(f"{name} must contain integer indices")
    if list(values) != sorted(set(values)) or any(x < 0 or x >= size for x in values):
        raise ValueError(f"{name} must be sorted, unique and in range")
    if count is not None and len(values) != count:
        raise ValueError(f"{name} count mismatch")
    return list(values)


def layout_sizes(rows, cols):
    integer(rows, "rows", 1)
    integer(cols, "cols", 1)
    local = 64 * rows * cols
    return dict(rows=rows, cols=cols, local_tokens=local, global_tokens=64, visual_tokens=local+64,
                coarse_region_capacities=[4 * rows * cols] * 16)


def local_region(ordinal, rows, cols):
    sizes = layout_sizes(rows, cols)
    integer(ordinal, "local ordinal")
    if ordinal >= sizes["local_tokens"]:
        raise ValueError("not a local token")
    crop, offset = divmod(ordinal, 64)
    r, c = divmod(crop, cols)
    v, u = divmod(offset, 8)
    return ((8*r+v) // (2*rows)) * 4 + (8*c+u) // (2*cols)


def sequence_contract(input_ids, attention_mask, rows, cols, selected_visual_indices, method):
    """Compute positions from explicit original IDs; never guess text length."""
    if method not in ("full", "s2_l", "s2_u", "s2_c"):
        raise ValueError("unknown method")
    sizes = layout_sizes(rows, cols)
    if not isinstance(input_ids, list) or any(type(i) is not int or i < 0 or i >= 49280 for i in input_ids):
        raise ValueError("explicit batch1 token IDs required")
    if not isinstance(attention_mask, list) or len(attention_mask) != len(input_ids) or any(type(m) is not int or m != 1 for m in attention_mask):
        raise ValueError("S3 contract requires one unpadded image, batch1, all-valid original mask")
    visual = [i for i, token in enumerate(input_ids) if token == IMAGE_TOKEN]
    n = sizes["visual_tokens"]
    if len(visual) != n:
        raise ValueError("placeholder count disagrees with observed R/C")
    selected = index_list(selected_visual_indices, n, "selected_visual_indices", n if method == "full" else 64)
    if method == "full" and selected != list(range(n)):
        raise ValueError("Full must retain every visual token")
    local_selected = sum(i < sizes["local_tokens"] for i in selected)
    if method != "full" and local_selected != 48:
        raise ValueError("pruning must retain local48/global16")
    visual_set = set(visual)
    nonvisual = [i for i in range(len(input_ids)) if i not in visual_set]
    if not nonvisual:
        raise ValueError("no text/special tokens")
    keep = sorted(nonvisual + [visual[i] for i in selected])
    # Full Layer0 is dense; all upper-layer prefill/cache positions are short.
    return dict(**sizes, original_sequence_length=len(input_ids), nonvisual_tokens=len(nonvisual),
                input_visual_positions=visual, nonvisual_positions=nonvisual,
                selected_visual_indices=selected, kept_sequence_positions=keep,
                original_position_ids_after_prune=keep, suffix_attention_mask=[1]*len(keep),
                pruned_sequence_length=len(keep), layer0_input_length=len(input_ids),
                suffix_input_length=len(keep), nonvisual_preserved=True,
                original_last_token_preserved=(keep[-1] == len(input_ids)-1))


def causal_allowed(query_positions, key_positions):
    """Scalar causal oracle in original logical coordinates (no tensor mask)."""
    if any(type(x) is not int or x < 0 for x in list(query_positions)+list(key_positions)):
        raise ValueError("invalid positions")
    return [[key <= query for key in key_positions] for query in query_positions]


def mock_cache_prune_append(sequence, generated_count, layers=LAYERS):
    """Lists stand in for K/V sequence axes; first generated ID comes from prefill."""
    integer(generated_count, "generated_count", 1)
    integer(layers, "layers", 1)
    if generated_count > 16:
        raise ValueError("generation exceeds lock")
    length = sequence["original_sequence_length"]
    keep = index_list(sequence["kept_sequence_positions"], length, "keep")
    cache = []
    for layer in range(layers):
        # Layer0 starts dense then gathers; upper layers fill the same short order.
        keys = [list(range(length))[i] for i in keep] if layer == 0 else list(keep)
        values = [-(p+1) for p in keys]
        for logical_position in range(length, length+generated_count-1):
            keys.append(logical_position)
            values.append(-(logical_position+1))
        cache.append(dict(keys=keys, values=values))
    return dict(layers=cache, prefill_lengths=[len(keep)]*layers,
                final_lengths=[len(keep)+generated_count-1]*layers,
                decode_position_ids=list(range(length, length+generated_count-1)),
                physical_next_position=len(keep)+generated_count-1,
                logical_next_position=length+generated_count-1,
                kind="synthetic Python lists; NOT HF DynamicCache validation")


def generation_diagnostics(ids, max_new_tokens=16, eos_token_id=EOS_TOKEN):
    if not isinstance(ids, list) or not ids or len(ids) > max_new_tokens:
        raise ValueError("generated token IDs must have length1..16")
    if any(type(i) is not int or i < 0 or i >= 49280 for i in ids):
        raise ValueError("generated token ID out of vocabulary")
    eos = [i for i, token in enumerate(ids) if token == eos_token_id]
    if eos and eos != [len(ids)-1]:
        raise ValueError("EOS occurs before final generated token or multiple times")
    at_limit = len(ids) == max_new_tokens
    if not eos and not at_limit:
        raise ValueError("unexplained early stop: no EOS and below token budget")
    return dict(generated_tokens=len(ids), eos_generated=bool(eos), eos_index=eos[0] if eos else None,
                at_token_limit=at_limit, max_new_tokens_without_eos=at_limit and not eos,
                possible_truncation=at_limit and not eos,
                stop_reason="eos" if eos else "max_new_tokens",
                semantic_completion="not inferred from token count or EOS")


def validate_runtime(row):
    """Validate required observed metadata, including retained text and all30 layers."""
    method, seq, runtime = row["method"], row["sequence"], row["runtime"]
    sizes = layout_sizes(seq["rows"], seq["cols"])
    for field in ("local_tokens", "global_tokens", "visual_tokens"):
        if seq[field] != sizes[field]:
            raise ValueError(f"invalid {field}")
    original = integer(seq["original_sequence_length"], "original_sequence_length", 1)
    visual = index_list(seq["input_visual_positions"], original, "input_visual_positions", sizes["visual_tokens"])
    n = sizes["visual_tokens"]
    selected = index_list(seq["selected_visual_indices"], n, "selected_visual_indices", n if method == "full" else 64)
    nonvisual = sorted(set(range(original)) - set(visual))
    keep = sorted(nonvisual + [visual[i] for i in selected])
    if seq["nonvisual_tokens"] != len(nonvisual) or seq["nonvisual_positions"] != nonvisual:
        raise ValueError("nonvisual token contract mismatch")
    if seq["nonvisual_preserved"] is not True or seq["original_last_token_preserved"] is not True or not keep or keep[-1] != original-1:
        raise ValueError("nonvisual/last token lost")
    for name in ("kept_sequence_positions", "original_position_ids_after_prune"):
        if seq[name] != keep:
            raise ValueError("original 1D positions changed or visual selection mapping invalid")
    short = len(keep)
    if (seq["pruned_sequence_length"] != short or seq["layer0_input_length"] != original
            or seq["suffix_input_length"] != short or seq["suffix_attention_mask"] != [1]*short):
        raise ValueError("sequence/mask lengths invalid")
    diagnostics = generation_diagnostics(row["generated_token_ids"])
    appends = diagnostics["generated_tokens"]-1
    if runtime["prefill_cache_lengths"] != [short]*LAYERS or runtime["final_cache_lengths"] != [short+appends]*LAYERS:
        raise ValueError("all30 cache lengths/append count invalid")
    if runtime["decode_position_ids"] != list(range(original, original+appends)):
        raise ValueError("logical decode RoPE positions use physical instead of original length")
    if runtime["append_counts"] != [appends]*LAYERS or runtime["causal_mask_validated"] is not True:
        raise ValueError("cache append or causal attention not validated")
    if method == "full":
        if selected != list(range(n)) or row["local_budgets"] is not None:
            raise ValueError("Full unexpectedly pruned or allocated")
    else:
        budget = row["local_budgets"]
        if (not isinstance(budget, list) or len(budget) != 16 or any(type(b) is not int or b < 1 or b > sizes["coarse_region_capacities"][i] for i, b in enumerate(budget))
                or sum(budget) != 48):
            raise ValueError("local allocation invalid")
        region_counts = [0]*16
        for i in selected:
            if i < sizes["local_tokens"]:
                region_counts[local_region(i, seq["rows"], seq["cols"])] += 1
        if region_counts != budget or len(selected)-sum(region_counts) != 16:
            raise ValueError("selected tokens disagree with regional/global budgets")
        if method == "s2_u" and budget != [3]*16:
            raise ValueError("Uniform allocation changed")
        if row["erc_finite"] is not True:
            raise ValueError("ERC nonfinite or unverified")
    for key in ("ttft_e2e_ms", "ttft_decoder_ms", "selection_overhead_ms", "peak_allocated_mib", "peak_reserved_mib"):
        value = row["measurement"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid measurement {key}")
    if row["measurement"]["peak_reserved_mib"] < row["measurement"]["peak_allocated_mib"]:
        raise ValueError("reserved memory below allocated")
    return diagnostics
