# Methods and controlled comparisons

Phase-one availability: Random/GC/Ours and the supplied-quota A/B engine are included. Image-to-Laplacian quota orchestration remains withheld. See [CODE_AVAILABILITY](CODE_AVAILABILITY.md) and [LICENSING](LICENSING.md). Method definitions below are unchanged.

## 1. Representation and objective

The source methods receive the projected visual embeddings immediately before insertion into the language model, not raw pixels or unprojected vision features. FP32 row normalization gives U. Let M = U U^T and A[i,j] = max(M[i,j] - mean_j M[i,j], 0). Every demand has weight one; no text-query weighting, softmax temperature or alpha mixture is added.

For selected set S, F(S) = sum_i max_{j in S} A[i,j]. Maintain best[i]. At each completion step compute gain[j] = sum_i max(A[i,j] - best[i], 0). Already selected candidates are masked by -infinity. Greedy order is retained internally; only the final set is sorted. Equal maximal gains use the first/lowest original candidate index through argmax. Exact K unique in-range sorted indices are required.

This objective measures representation under this affinity. It is not a definition of retained object evidence or VQA correctness.

## 2. Dynamic spatial mapping

The processor supplies actual crop rows and columns. Local views are row-major and the final view is global. The historical model/config produced an 8x8 projected grid per view. Counts are checked from actual tensors/config; neither N=832 nor N=1088 is a universal constant.

For a token grid (h,w), local count = rows*cols*h*w. A local index decomposes into view index and intra-view row/column. The local lattice is formed as:

```text
arange(N_local).reshape(rows, cols, h, w)
    .permute(0, 2, 1, 3).reshape(rows*h, cols*w)
```

The final global view is excluded from the local lattice. Near-equal region boundaries use round(i*size/parts), matching the retained implementation. Floor is infeasible when G>K or any region would be empty.

This mapping is a nominal token/crop coordinate system. It does not prove exact source-pixel footprints, neural receptive fields, segmentation masks, or token-object correspondence.

## 3. Five historical methods

| Method | Floor | Completion candidates | Demand pool | Extra constraint |
| --- | --- | --- | --- | --- |
| Random | None | All N | Not an objective-based selector | CPU randperm |
| Pure GC | None | All N | All N | None |
| Ours | One medoid per local region | All N (local+global) | All N | None after floor |
| A | Same local floor as B/Ours | Local only | All N | None after floor |
| B | Same local floor as A/Ours | Local only | All N | Regional final quotas |

Random resets a CPU torch.Generator to seed 42 for every image/budget, calls randperm(N)[:K], and sorts. It does not use a continuously advancing generator or a seed sweep.

A regional floor medoid maximizes sum_{i in region} A[i,j] over j in that region. Ties use the lowest visual index because region candidate indices are sorted. After seeding, Ours and A/B evaluate marginal gains over all N demands, including global-view demands.

A and B share the same projected features, affinity, floor tensor, demand pool, candidate pool and completion scorer. Historical B differs only by enforcing final Laplacian-based regional budgets. A region becomes unavailable once its quota is filled.

The historical quota uses original-image grayscale/full-image four-neighbor Laplacian, regional variance, normalization and minimum-one capped allocation with deterministic rounding. The S2-reference implementation and orchestration are withheld pending distribution review; no substitute is introduced here.

B uses neither measured decoder ERC nor recursive-cell token selection during the POPE comparison. It is an allocation-only control. Ours/B differs in more than the quota because Ours also allows global candidates. The clean quota comparison is A/B.

## 4. Physical pruning and generation

The common adapter embeds the original IDs, inserts the N projected visual features at image-placeholder positions, and retains every nonvisual position plus the K selected visual positions. The number of visual placeholders must equal the feature count before merging and K after shortening.

Input embeddings, token IDs and attention mask are gathered with the same keep positions. Audit position IDs follow the contiguous mask-cumsum convention after deletion; the audit-only tensor is not manually forwarded as an independent positional scheme. Each generate call owns a fresh cache. The audited historical path is single-image, unpadded, batch1, Idefics3/Llama eager.

The generation audit checks actual prefill length, RoPE/position handling, causal masking, cache progress and unwanted vision re-execution. Selected tokens remain in original sequence order. Zero masking is not used as a substitute for sequence shortening.

Vision encoder/projector computation is not pruned: all N features are computed before selection.

## 5. Controls and limits

K=N compares the direct-insertion path with the ordinary full model under identical greedy settings before small-K tests. Passing equality checks establishes adapter identity on the tested cases, not correctness of their answers or support for arbitrary models/batching.

Current copied selectors, adapters and tests retain their original algorithms and guards. Withheld modules are listed in [CODE_AVAILABILITY](CODE_AVAILABILITY.md). New portability work, model execution and additional experiments were not performed for this candidate.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
