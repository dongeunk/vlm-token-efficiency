# Historical five-method controlled comparison

This is a public documentation derivative, not the original run guide.

The private historical study evaluated 30 fixed questions x five methods x two budgets. Its Full baseline was rescored from stored outputs. The original five-method runner, S2-reference quota orchestration and dependent tests are held from this candidate for distribution/dependency review.

Do not follow old commands as if this candidate contains that runner or its private inputs. See [reproduction boundaries](../docs/REPRODUCIBILITY.md) and [method definitions](../docs/METHODS.md).

Historical invariants: seed42 reset per image/budget for CPU Random; shared A/B affinity and floor; K64/G16 and K128/G64; FP16/eager; greedy16, beam1, batch1; original questions/template; fresh cache; exact K and real sequence shortening. B was an allocation-only control, not full S2Prune.

The original system required a separately approved full run, a matching successful 20-generation smoke, unchanged protected-file hashes and a new empty output directory. Those guards are not removed or bypassed here.

Timing records distinguish feature extraction, shared A/B affinity/floor, quota, completion and debug-hook generation. Shared costs must not be double-counted; generation timing is not end-to-end speedup.

Current aggregate historical results are in [RESULTS](../docs/RESULTS.md). Raw predictions, original manifests, data and historical path-dependent configs are withheld.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
