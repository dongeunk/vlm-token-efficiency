# Visual Token Pruning: Representational Coverage and Spatial Floors

**Phase-one research snapshot | historical 30-question pilot**

Can query-independent representational coverage preserve enough visual evidence for a small vision-language model, and does a minimum spatial allocation help? This personal research project studies that question with SmolVLM-256M-Instruct. It connects projected visual-token selection to actual language-model sequence shortening, checks the inference path, and examines failures rather than assuming spatial coverage guarantees answer quality.

**Main finding:** Spatial Floor + Global Coverage (Ours) guarantees local-region support, but is **not consistently better than Pure Global Coverage** on the fixed pilot. Ours scores 25/30 versus Pure GC 28/30 at K=64; both score 27/30 at K=128. These are exploratory results, not official full-POPE benchmark scores.

## Release scope

This phase-one research snapshot contains 72 selected files and is authorized
for GitHub publication. No experiment or test suite was run to prepare it.
Publication completion is established by remote verification, not assumed here.

- Random, Pure Global Coverage, Ours, shared inference/auditing/evaluation
  source and selected tests are included.
- The A/B coverage engine accepts externally supplied quotas. The image-to-quota
  orchestration and historical five-method runner remain withheld.
- The S2Prune reference implementation and ten associated files stay on HOLD.
  Rights HOLD and dependency HOLD are distinguished in [CODE_AVAILABILITY](docs/CODE_AVAILABILITY.md).
- Model weights/cache, dataset annotations/manifests, raw predictions,
  original images and photograph-containing figures are not included.
- Apache-2.0 applies within the file-level scope in [LICENSING](docs/LICENSING.md).
  MMTok attribution and changes are recorded in [NOTICE](NOTICE).
- Existing [vlm.py](vlm.py) is retained byte-for-byte from its GitHub source.
  Its license treatment is stated explicitly in LICENSING.

The five-method results below are historical numerical evidence. They do not
mean the full historical evaluation can be replayed from this release.
See [included files](PUBLIC_FILES.md), [exclusions](EXCLUSIONS.md), and
[release/privacy scope](docs/ETHICS_AND_RELEASE_SCOPE.md).

## Background and related work

Visual token pruning reduces the sequence passed to a VLM's language model. Attention, similarity, query relevance and spatial structure motivate different selection rules. This project separates an embedding-space coverage objective from a local minimum-allocation constraint.

The implementation history references MMTok's vectorized greedy coverage structure and S2Prune's spatial allocation ideas. CoverPruner attribution for the row-centered affinity remains unverified in the retained evidence. A 20-paper local literature inventory is documented with separate review states; file possession is not treated as completed reading. See [Literature Review](docs/LITERATURE.md).

## Methods

For projected visual embeddings V in R^(N x D), normalize each row and define:

```text
U = L2Normalize(V.float())
M = U @ U.T
A = clamp(M - M.mean(dim=1, keepdim=True), min=0)
F(S) = sum_i max_{j in S} A[i,j]
```

All demand weights are one. No query weighting is used. Greedy completion maintains current coverage, masks selected candidates with -infinity, and sorts only the final selection into the original visual-token order.

| Method | Candidate pool after seeding | Spatial rule |
| --- | --- | --- |
| Random | All local + global tokens | Fresh CPU Generator(seed=42) for each image/budget; randperm, keep K, sort |
| Pure GC | All local + global tokens | No floor or regional quota |
| Ours | All local + global tokens | One regional affinity medoid per local region, then free global completion |
| A: Local-only Coverage | Local tokens only | Same floor, affinity and all-N demand as B; free completion |
| B: Laplacian Quota | Local tokens only | Same A inputs/scorer; completion additionally constrained by final regional quotas |

B is an **allocation-only control**, not a full S2Prune reproduction: historical B inference uses neither recursive-cell selection nor actual ERC scores. A/B isolates the quota constraint; Ours/A isolates the completion candidate scope. Full definitions are in [METHODS](docs/METHODS.md).

## Model and experiment settings

| Item | Historical setting |
| --- | --- |
| Model | HuggingFaceTB/SmolVLM-256M-Instruct |
| GPU | NVIDIA GTX 1660 SUPER, 6 GB |
| Precision / attention | FP16 / eager |
| Runtime | Python 3.11.15; Torch 2.13.0+cu126; Transformers 5.15.1 |
| Data | Fixed original COCO POPE random subset: 30 questions, 15 Yes / 15 No, 30 distinct images |
| Selection / sampling seed | 42 |
| Budgets | K=64 with 4x4 regions (G=16); K=128 with 8x8 (G=64) |
| Decoding | Greedy, do_sample=False, max_new_tokens=16, num_beams=1, batch=1 |
| Prompt | Original annotation question, standard processor chat template, no appended answer instruction |
| Cache | Fresh KV-cache for each generation |
| Evaluation | Strict and visual_semantic_yes_no_v1; invalid is incorrect for accuracy |

See [recorded environment](environment/observed_versions.json) and [reproduction limits](docs/REPRODUCIBILITY.md). G only labels spatial diagnostics for Random/Pure GC; it constrains floor-based selectors.

## POPE 30-question results

Each cell is correct answers / 30. All 300 pruning generations completed in the historical experiment. Stored Full Model responses were reused, not regenerated for this comparison.

| Method | K64 Semantic | K64 Strict | K128 Semantic | K128 Strict |
| --- | ---: | ---: | ---: | ---: |
| Random | 25/30 | 19/30 | 28/30 | 19/30 |
| Pure GC | 28/30 | 22/30 | 27/30 | 21/30 |
| Ours | 25/30 | 23/30 | 27/30 | 24/30 |
| A: Local-only | 28/30 | 24/30 | 26/30 | 22/30 |
| B: Laplacian quota | 27/30 | 23/30 | 25/30 | 20/30 |

The stored Full Model baseline is **Semantic 26/30 (86.67%)** and **Strict 18/30 (60.00%)**. Its 12 strict-invalid responses fell to 2 after separate semantic rescoring of the same raw outputs. That change is a parser/evaluation change, not a new model result.

Semantic v1 was developed after examining output forms. Its function does not receive labels or question IDs, but held-out parser generalization is unverified. Strict results remain reported alongside it. See [RESULTS](docs/RESULTS.md) for invalid counts, paired outcomes and linked aggregate artifacts.

## Inference-path validation

Pruning is applied **after the vision encoder/projector and before LLM prefill**. Unselected visual placeholders are physically removed; all text, separators and other nonvisual tokens are retained. The selected embeddings and placeholders stay aligned and in original order.

Historical K=N gates on N=832 and N=1088 report exact input-embedding and first-step-logit equality (maximum absolute difference 0), identical generated token IDs and identical raw answers. Identity validates the plumbing, not answer correctness.

Actual Ours smoke examples demonstrate sequence shortening:

| Case ID | N | K | Original LLM sequence | Pruned / audited prefill length |
| --- | ---: | ---: | ---: | ---: |
| q2794 | 832 | 64 | 879 | 111 |
| q2794 | 832 | 128 | 879 | 175 |
| q1603 | 1088 | 64 | 1143 | 119 |
| q1603 | 1088 | 128 | 1143 | 183 |

These are stored historical measurements; no model was loaded for this preparation. Vision/projector computation still processes all N tokens. Debug-hook generation times are not end-to-end speedup measurements.

## Structural heterogeneity follow-up

The post-hoc analysis uses regional Laplacian-variance complexity and H = population_std(C) / (mean(C) + 1e-8), separately for the two K/grid configurations. Low/Medium/High groups contain 10 images each, ranked by (H, image ID, question ID), without using correctness.

- Ours has zero empty local regions at both budgets.
- Pure GC has mean 0.2 empty regions at K64 and 9.8 at K128.
- At K64, Pure GC already covers all regions on 25/30 images; all three Ours-only losses occur within that stratum.
- High heterogeneity does not establish a general Ours advantage. The stored Spearman H-versus-paired-correctness-difference values are approximately 0.1861 and 0.0000.

![Semantic accuracy by H group](results/pope_spatial_heterogeneity_30/figures/accuracy_by_heterogeneity_group.png)

These photo-free plots were copied from the historical analysis. This is an S2Prune-inspired diagnostic, not a reproduction of its official heterogeneity figure. Nominal crop/token coordinates are not object masks or verified neural receptive fields. See [HETEROGENEITY](docs/HETEROGENEITY.md).

## Available functionality and reproduction boundary

| Task | Scope in this working copy |
| --- | --- |
| Read methods, source, aggregate JSON and anonymous CSV | Available without model/data downloads |
| Synthetic checks of Random/GC/Ours, adapters and parsers | Source provided; requires a compatible environment; not run during preparation |
| A/B synthetic engine checks with supplied quotas | Source provided; image-derived Laplacian quota workflow is withheld |
| Single-image pruning examples | Need a separately supplied image, compatible environment and model cache |
| Replay original POPE pilot/smoke | Requires withheld manifests, images and baseline/evidence artifacts |
| Replay all historical comparison generations | Not available: integrated runner, raw baseline/input artifacts and other dependencies are withheld |
| Recompute complete image-based H analysis | Not available: analysis runner/helper and original image inputs are withheld |
| Feed aggregate exports to historical raw-data tools | Not supported; aggregate schemas are not raw-prediction replacements |

The included test command is documentation only; it was not executed:

~~~console
python -B -m unittest discover -s tests -p "test_*.py" -v
~~~

Some tests import Transformers; tensor tests require Torch. Artifact-dependent
checks may skip without private inputs. Passing static import/AST inspection
does not establish runtime compatibility or a successful test suite.

### Example requirements and side effects

- `run_small_vlm.py` and `inspect_visual_tokens.py` may download model files;
  their loaders are not offline-safe by default and do not pin a revision.
- Other model examples require their documented local cache and inputs.
  Example defaults such as `images/meta.webp` and diagnostic photos are absent.
- `download_pope_pilot_images.py` performs network requests and writes reports
  if explicitly invoked. It was not invoked for this preparation.
- Plot/report utilities require the full historical schema and, for panels,
  original images. The exported aggregate files are not drop-in replacements.
- `vlm.py` is the original PyTorch/CUDA matrix-multiplication check. It is not
  the VLM inference entry point and is not a data-free static check.

No model, test, generation or benchmark ran during preparation. No automatic
transition to GPU evaluation is configured. See [REPRODUCIBILITY](docs/REPRODUCIBILITY.md).

## Limitations and open questions

- One model, a single fixed 30-question sample and one Random seed; no statistical or benchmark-wide superiority claim.
- K and G change together; the floor consumes 25% versus 50% of K.
- Coverage objective value, regional support and VQA accuracy measure different things.
- Object-specific evidence retention is unverified; zero-region counts do not prove object loss.
- Semantic parser generalization is unverified; answer-format effects must be separated from recognition failures.
- Original experiment code was untracked. Its execution commit is **not recorded**. No later commit is assigned retroactively.
- Public copies are incomplete derived records, not byte-identical historical configs or complete execution evidence.

Next work is proposed, not executed: cross K and G independently, match local/global budgets in downstream comparisons, use larger prespecified samples and additional seeds, validate object-coordinate evidence, and evaluate parsers on held-out responses.

## Documentation and provenance

- [Methods](docs/METHODS.md)
- [Results and evaluation contract](docs/RESULTS.md)
- [Heterogeneity analysis](docs/HETEROGENEITY.md)
- [Reproducibility and provenance](docs/REPRODUCIBILITY.md)
- [Literature and implementation references](docs/LITERATURE.md)
- [Release scope](docs/ETHICS_AND_RELEASE_SCOPE.md)
- [License scope](docs/LICENSING.md)
- [License text](LICENSE) and [attribution notices](NOTICE)
- [Public file manifest](PUBLICATION_MANIFEST.json)

The manifest separates original SHA-256 from public-copy SHA-256 and lists transformations. Local paths are not represented as externally accessible links. Initial preparation did not stage, commit or upload files. Publication uses a separate clone and preserves existing repository history; the publication commit is identified by Git history. No Notion update or model execution accompanies publication.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
