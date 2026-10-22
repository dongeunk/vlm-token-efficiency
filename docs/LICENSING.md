# Licensing and component scope

## Grant and boundary

The owner authorizes Apache License 2.0 for original research-code and
documentation contributions in this phase-one working copy. The public
copyright identifier is **dongeunk**. The unmodified English license is
[LICENSE](../LICENSE); component attribution is in [NOTICE](../NOTICE).

This is an owner-authorized phase-one publication payload. Publication success
is established by remote verification, not assumed by this document. A
repository-level license does not create rights in code, data, papers, or
assets owned by others. See [held components](CODE_AVAILABILITY.md).

| Files / component | License treatment |
| --- | --- |
| Included Python files under src/, examples/, tests/ | Apache-2.0 for original contributions; relevant MMTok rights retained below |
| README.md, docs/, examples/README_pope_controlled_comparison.md | Apache-2.0 for original project text; quotations, titles and bibliography do not acquire a new project copyright |
| .gitignore, .gitattributes and publication metadata | Apache-2.0 for original authored expression; hashes and factual values are provenance records |
| Selected result JSON/CSV and three photo-free figures | Apache-2.0 for the owner's authored presentation/compilation/figure expression only; no exclusive claim over facts or grant over source datasets |
| LICENSE | Verbatim official license instrument; no project authorship claim |
| vlm.py | Retained unchanged; new license grant HOLD (outside the grant above) |
| Excluded upstream sources, models, annotations, images, raw predictions and HOLD files | Not licensed or redistributed by this working copy |

## MMTok-referenced portions

The greedy update structure in `src/pruning/coverage.py:global_coverage_select`
explicitly references `greedy_merged_jit_kernel` from
[the fixed upstream source](https://github.com/Ironieser/MMTok/blob/8e2686836ed5bbe05d064f1d1c20e50c73c70b94/mmtok/core/semantic_selector.py#L24).
Relevant upstream notice: Copyright (c) 2025 Zoom Communications, Inc.;
Author: Sixun Dong; Apache-2.0.

Completion in `src/pruning/spatial_floor_coverage.py:spatial_floor_coverage_select`
and `src/pruning/matched_allocation.py:_complete` follows that local baseline.
File comments and NOTICE preserve this source chain and distinguish original
regional floor/quota-control work from the referenced greedy structure.
They do not assert that Zoom authored every line in these files.

The current affinity is row-mean-centered clipped cosine similarity over
projected embeddings with unit visual demand weights. It does not implement
MMTok's text-vision P, softmax/temperature/alpha combination, JIT wrapper, or
Qwen integration. The row-centering must not be attributed to MMTok merely
because its greedy update was referenced. CoverPruner equivalence remains
unverified in the retained evidence.

The publication annotations do not establish literal copy history or a legal
determination about derivative-work boundaries. Related portions are managed
with Apache-2.0 attribution rather than described as wholly unrelated code.

## Legacy vlm.py

[vlm.py](../vlm.py) is the pre-existing PyTorch/CUDA matrix-multiplication check.
The included bytes match GitHub/Git blob
`a1d7c229b4c67eeaf75b3cdb690ad44588e51c85` at repository commit
`e3057a3cb9b8e8dc4e464665334f822420b73d62` in
[dongeunk/vlm-token-efficiency](https://github.com/dongeunk/vlm-token-efficiency/blob/e3057a3cb9b8e8dc4e464665334f822420b73d62/vlm.py).
The local working-tree copy differs only in CRLF versus LF. Neither copy is
modified. No new header is added to the retained file.

The owner's general first-party research-code grant is not used to infer uncertain legacy rights. No new license is assigned to this file pending explicit confirmation; it remains retained as requested.

Git source identity is evidence of bytes and repository history, not proof of
sole authorship. No email address or other Git identity metadata is exported.
This legacy commit is not assigned to later pruning experiments.

## S2Prune HOLD

The eleven paths are listed in [CODE_AVAILABILITY](CODE_AVAILABILITY.md).
One reference-implementation file is on rights HOLD; one mixed-provenance
test is on rights/fixture HOLD; nine original orchestration/analysis/test
files are withheld for execution dependency. Import dependency alone does
not establish source-code borrowing.

No project license is applied to the excluded reference implementation.
Future permission or independently reviewed work requires a separate release
decision; renaming, a submodule, or a download instruction does not resolve
the permission question.

## Publication lineage

Comments and release documentation are new publication changes, not historical
experiment modifications. [PUBLICATION_MANIFEST](../PUBLICATION_MANIFEST.json)
retains original source SHA values, the reviewed-candidate SHA, current SHA,
and the transformations separately. Historical project execution commit:
**not recorded**. The publication commit is identified by the containing
repository history; its own hash is not embedded in these contents.

Official Apache text:
https://www.apache.org/licenses/LICENSE-2.0.txt

Initial static preparation involved no model inference, selector execution,
test suite, package installation, Git staging/commit/push, remote modification
or Notion update. The owner subsequently authorized publication in a separate
clone; no new model/test execution or Notion change accompanies publication.
Dataset/model acquisition and experimental execution need their own applicable
permissions and environment.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
