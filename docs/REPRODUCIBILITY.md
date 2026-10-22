# Reproducibility, provenance and execution boundaries

## Status

This is a phase-one research snapshot with explicitly incomplete historical replay coverage. It is not a verified fresh-install package or a one-command reproduction of the historical five-method evaluation. No model, generation, benchmark, pytest/unittest suite or installation was executed to create it. Preparation checks use standard-library file parsing, hashing and scans only.

## Historical identifiers

| Component | Recorded identifier |
| --- | --- |
| Model | HuggingFaceTB/SmolVLM-256M-Instruct |
| Model revision | 7e3e67edbbed1bf9888184d9df282b700a323964 |
| POPE upstream revision | 08d957b917e5a378a2f99d35b6293c536a66298b |
| MMTok reference revision | 8e2686836ed5bbe05d064f1d1c20e50c73c70b94 |
| S2Prune reference revision | b0ac197e5e765f8a151669fdcbce10b621bbf44b |
| Historical project execution commit | Not recorded |
| Publication commit | Identified by Git history for this snapshot; not embedded in its own contents |

Upstream revisions are not project execution commits. The existing initial project commit contained only a GPU check and ignore rules, not the later experiment source. It is not assigned to historical runs.

## Environment

The observed environment is recorded in [observed_versions.json](../environment/observed_versions.json). These are observed package versions, not a solver lock, wheel-origin proof or tested fresh-install recipe. CUDA/Torch binary availability on a different OS or at a later date has not been established.

Recorded core: Python 3.11.15, Torch 2.13.0+cu126, Torchvision 0.28.0+cu126, Transformers 5.15.1, Pillow 12.3.0, NumPy 2.4.6, Matplotlib 3.11.1; GTX 1660 SUPER 6 GB, FP16/eager. Tests use unittest; pytest is not required.

Do not install upstream third-party requirements into the research environment as a substitute: their requested versions differ from the historical SmolVLM environment.

## Available checks

From this candidate root, after separately preparing compatible dependencies:

```console
python -B -m unittest discover -s tests -p "test_*.py" -v
```

This is an example command, not a claim that it was run on the candidate. Retained tensor/mock tests do not require model weights. Some diagnostic tests import Transformers. Artifact-dependent checks can skip when private files are absent; a skipped reference check is not a reproduced experiment.

The data-free adapter tests inspect direct insertion and true sequence shortening with synthetic/mock inputs. Full historical runtime evidence is summarized under results; raw inputs/outputs are not redistributed.

## Model demos and side effects

- run_small_vlm.py and inspect_visual_tokens.py can download model files because their loaders are not local_files_only and do not pin a revision.
- Other copied model runners use local/offline loading, but a local cache alone does not guarantee the intended snapshot.
- download_pope_pilot_images.py performs network downloads and writes a download report if invoked.
- Original example defaults may require images/meta.webp or the five diagnostic images; none is included.
- POPE pilot and Ours smoke require withheld annotations, manifests, images and/or stored baseline responses.
- plot/report tools require the full historical analysis schema and image inputs; the aggregate exports are not drop-in replacements.

No examples above were executed during preparation. Supplying data, cache, download permission and a fresh output directory is a separate action. Never direct a rerun at a nonempty historical output directory.

## Why exact replay is unavailable

1. Raw predictions and original annotations/manifests/images are withheld for distribution/privacy review.
2. The S2-reference module and dependent five-method/H-analysis code are held; see [CODE_AVAILABILITY](CODE_AVAILABILITY.md).
3. Historical protected-file dictionaries contain absolute project and installed-library paths, including Windows-specific separators.
4. Their checks use exact raw-byte SHA values and can reject relocation, different installed sources, changed file sets, or line-ending changes.
5. Public aggregates omit raw answers, prompts, token IDs/indices and some identity lists; they cannot replace original evaluation inputs.

Do not edit historical configs, recalculate old hashes over changed content, remove approval gates or treat an old successful smoke as validation of a new checkout. A future portable reproduction would need separately approved tooling, new provenance and new validation in its own output directory.

## Original and public-copy hashes

PUBLICATION_MANIFEST.json records every candidate file except itself. Original-derived entries record source-relative paths, original raw SHA-256, public-copy SHA-256 and transformations. Newly authored documentation is marked generated, with its evidence sources. The private preparation audit outside this candidate retains the original project baseline and verification results.

Original copies remain unchanged in the private project. A changed public hash means a documented derivation, not corruption of the original. The manifest avoids absolute personal paths.

A SHA establishes byte correspondence to the inspected local source; by itself it does not prove when the historical computation ran, that a raw response is authentic, or that the complete experiment has been independently reproduced.

## Line endings and Git

The original research Python sources were LF-only while effective core.autocrlf was true. This candidate includes .gitattributes to request LF for source/text and preserve PNG bytes. During initial static preparation, no Git add/checkout was performed to test this policy. Publication requires staged Git blob bytes to match the final publication manifest. Public-file hashes are computed from the final payload bytes.

## Approval boundary

During initial static preparation, no Git staging, commit, push, remote changes, model inference, environment update or Notion modification was performed. Preparation fetched only the official Apache license text and read GitHub metadata/source for legacy provenance. No model or dataset was downloaded. The owner subsequently authorized status-only publication edits, a separate clone, a new commit and a normal push to the existing main branch. Publication completion is verified against the remote. Additional data clearance, automatic updates, experiments and Notion changes remain outside this approval.

## Publication annotations and legacy bytes

License/attribution comments change public raw-byte hashes but preserve parsed Python AST structure. Previous source and reviewed-candidate hashes remain recorded; no historical hash gate or approval record is rewritten. The retained legacy file uses the existing GitHub LF bytes instead of the equivalent local CRLF representation; its provenance and license treatment are in [LICENSING](LICENSING.md). Static validation does not establish runtime execution, fresh installation or complete historical replay.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
