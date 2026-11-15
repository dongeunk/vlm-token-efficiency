> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01/SPATIAL_ALLOCATION_ADAPTATION.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S0 — Spatial Allocation 이식 설계

작성: 2026-11-14. **설계만 작성했다. 새 selector/adapter 구현과 테스트는 하지 않았다.** 기존 Qwen 연구는 원본 재현 트랙, 이 문서는 SmolVLM 구조에 맞춘 이식 트랙이다. S²Prune HOLD는 그대로 유지한다.

## 1. 권고 결정

**B: Local stitched lattice에 공간 예산을 배분하고 Global은 별도 고정 예산으로 관리한다.** 1차 구현 및 Smoke의 사전 설정은 다음과 같다. 최적값을 찾았다는 의미가 아니며 실제 layout gate 통과 후 실행 설정으로 잠근다.

| 항목 | 설정 |
|---|---|
| 원본 visual 집합 | 모든 local RC views + global 1 view, 각64개; N=64(RC+1) |
| local 공간 격자 | 8R×8C, 원본 normalized image plane과 대응 |
| local coarse regions | 4×4, G_local=16 |
| local 예산 | K_local=48, 지역 최소1, 가변 배분32 |
| global coarse regions | 8×8 전체를 단일 region으로 관리, G_global=1 |
| global 예산 | K_global=16 고정; 공통 recursive partition으로16 cells |
| 전체 | K=64, coarse regions17, final cells64 |
| 최종 대표 | 각 cell의 동일 Layer0 ERC argmax, native visual ordinal 순서 정렬 |

여기서 16개 global cells를 16개 coarse regions라고 부르지 않는다. local16+global1=17 coarse regions에서 q의 합64만큼 final cells가 생긴다. Global의 최종16개도 ERC로 고르며 고정 uniform token subsampling을 쓰지 않는다.

## 2. A/B/C 방식 비교

| 방식 | 좌표와 수요/후보 집합 | Layer0 ERC·K 관리 | 복잡도와 판단 |
|---|---|---|---|
| A. Local+Global 함께 공간 배분 | 같은 원본 공간을 겹치는 다중 해상도 후보가 차지함. region에 두 역할을 함께 넣을지, 층을 분리할지 새 ownership 정의가 필요. Coverage는 all-N이지만 중복 역할의 비중이 섞임 | ERC 자체는 모든 visual에서 가능. region capacity와 cells를 단일 disjoint lattice로 정의할 수 없음 | 원본 rectangular Cell 규칙을 바꿔야 할 가능성이 큼. primary 비추천 |
| **B. Local 배분+Global 고정** | Local은 단일8R×8C disjoint lattice. 배분용 demand/candidate는 local 전체. Global은 별도8×8 | full Layer0 ERC는 전체N에서 계산. local48+global16으로 항상64 | mapping adapter만 추가하고 공통 rectangular cells 유지 가능. **권고** |
| C. crop별 분할+전체 배분 | 각 crop 내부8×8에 coarse regions. crop 간 quota 배분을 추가 정의해야 함 | ERC 가능. crop 수가 달라지면 base floor가 변함 | 12/16 local views에 crop당2×2면 local floor만48/64. global16까지 넣으면 K64에서 여유0 또는 불가능. crop당1region도 새로운 crop quota 연구축. 후속 별도 ablation 후보 |

B의 local16 regions는 crop 경계와 항상 같지 않다. 예를 들어3×4 crops는24×32 lattice이며4×4 regions는6×8 token rectangles다. 지역이 crop 경계를 가로질러도 원본 공간에서 연속된 사각형이다. 이 방식은 모든 crop의 최소1개 보존을 보장하지 않는다. 보장 대상은16개 원본 공간 지역이다.

## 3. 공통 geometry·순서·Cell 규칙

`spatial_floor_coverage.local_region_indices`의 view-major→stitched mapping을 근거로 한다. lattice row-major 순서로 feature를 재배열한 뒤 이를 native 순서라고 부르면 안 된다.

- coarse boundaries: 원본 `build_coarse_regions`의 `round(i·size/parts)` 규칙.
- Cell: 가장 큰 splittable rectangle부터 이분. 면적 동률은 현재 cell 순서 우선, 긴 변을 분할하고 길이 동률은 row 우선, midpoint floor.
- region별 q_g개 nonempty cells, 전체 tokens를 중복·누락 없이 덮고 coarse 경계를 넘지 않는다.
- Cell의 lattice 좌표를 native visual ordinals로 변환한다. 후보를 native ordinal 오름차순으로 정렬하고 ERC argmax한다. exact tie는 최저 native ordinal이다.
- 최종 local/global union도 native ordinal로 정렬한다. 전체 prompt의 `visual_positions[ordinal]`로 변환한 뒤 모든 nonvisual positions와 합쳐 원래 순서를 유지한다.
- Global ordinal은 N_local부터 시작하고, global8×8 내부는 row-major다.

기존 단일-grid selector의 "row-major=original index" 전제는 Smol에서 성립하지 않는다. **Cell geometry 함수는 재사용 가능하지만 기존 `select_cells_with_budgets` 전체를 직접 호출하면 tie 규칙이 바뀔 수 있다.** 별도 mapping-aware glue가 필요하다.

## 4. S2-L/U/C의 유일한 차이

### S2-L — Laplacian local 예산

원본 RGB 이미지에 기존 grayscale·4-neighbor replicate Laplacian을 적용한다. local coarse rectangles를 원본 image plane으로 비례 매핑하여 population variance를 계산하고, 이미지 내 FP32 min-max 후 기존 capacity-aware largest-remainder로48개를 나눈다. 최소1/지역이며 수용량은 실제 region token 수다. 이미지가 상수일 때의 fallback·tie도 기존 allocator를 유지한다.

Laplacian은 crop JPEG를 새로 만들거나 Global resized view에서 별도로 계산하지 않는다. 기존 original-image complexity 정의를 사용한다. processor의 actual R/C와 원본 W/H mapping을 입력 명세에 남긴다.

### S2-U — Reconstructed Uniform

16개 local regions의 동일 weight를 같은 capped largest-remainder에 넣는다. 현재 K_local48/G16에서는 capacity가 충분하면 각3개다. **원저자 Uniform ablation의 정확한 공개 구현이라고 주장하지 않는다.**

### S2-C — Query-independent Coverage local 예산

입력 V는 **projector 출력의 local tokens 전체**이며 native 순서를 유지한다. 질문·텍스트·attention·ERC·최종 hidden을 affinity에 사용하지 않는다.

```text
z_i = FP32(V_i) / max(||FP32(V_i)||_2, 1e-12)
S_ij = z_i^T z_j
A_ij = max(S_ij - mean_j S_ij, 0)
F_local(T) = sum_i max_{j in T} A_ij
```

단위 수요이며 demand와 candidate는 둘 다 N_local 전체다. 행 평균은 local 전체에서 계산한다. all-N으로 calibration한 행렬의 local block을 잘라 쓰는 것과 다르다.

빈 집합에서 임시32개를 greedy marginal gain으로 고른다. region별 추가 한도는 capacity_g−1, zero gain에서도32개를 채운다. exact tie는 최소 native ordinal이고 epsilon tie class를 도입하지 않는다. `q_g=1+임시선택 개수_g`로 local budgets만 만든다. **임시32개는 최종 선택 tokens가 아니다.** 이후 모든 방법이 같은 Cell/ERC로 최종48개를 고른다.

Global q=16 및 global cells/ERC는 L/U/C에서 동일하다. 같은 H0/H1이면 global final indices도 동일해야 한다.

## 5. Coverage 진단의 정의

배분용 local affinity와 최종 품질 측정을 구분한다.

- `F_local(S_local)`, `F_local/N_local`: 배분과 같은 local-only calibration으로 최종 ERC local48 평가.
- `F_all(S)`, `F_all/N`: 별도로 all-N projector V에서 calibration한 affinity로 최종64 평가. Local/Global 모두 unit demands/candidates다.
- 가능하면 local-demand 및 global-demand 기여도도 같은 A_all 기준으로 분해한다.
- 임시32의 F와 최종 ERC64의 F를 혼합하지 않는다. N과 calibration domain이 다른 raw F를 이미지 간 직접 평균 비교할 때는 normalized 값과 N strata를 함께 제시한다.

S2-C는 예산 배분에만 Coverage를 사용하므로 최종 ERC 집합의 F나 POPE Accuracy 우위를 보장하지 않는다. 평가용 affinity는 방법 공통이며 Full/L/U에서도 동일한 정의로 계산한다.

## 6. 기존 CPU 코드 재사용 경계

| 파일·함수 | 판단 |
|---|---|
| `src/s2_coverage_study/allocation.py::calibrated_affinity_fp32:166` | 일반 N,D 지원. CPU FP32 그대로 재사용 가능 |
| `greedy_capacity_select:191` | 일반 region_of/capacities 지원. native local 순서+mapping 제공 가능 |
| `build_coarse_regions:92`, `region_layout:111` | grid_h/w 명시 시 임의 rectangular lattice에 사용 가능 |
| `laplacian_region_budgets:284`, `uniform_region_budgets:308` | explicit local grid/regions/budget48로 재사용 가능 |
| `coverage_region_budgets` | 내부 flattened lattice ownership과 native feature 순서가 다름. 그대로 호출 금지; 일반 affinity/greedy helper를 조합하는 새 adapter 필요 |
| `third_party/S2Prune/s2prune/allocation.py::recursive_region_cells:175` | geometry 원본 helper를 private reference로 호출 가능. HOLD 유지, body 복사 금지 |
| `src/s2_coverage_study/selection_adapter.py::select_cells_with_budgets:82` | 단일 row-major grid 가정. geometry만 참고하고 native tie/membership를 새 adapter에서 검증 |
| `src/pruning/spatial_floor_coverage.py::local_region_indices:62` | mapping 근거. medoid floor+global greedy selector 자체는 새 S2-C가 아님 |
| `src/pruning/coverage.py::global_coverage_select` | 1차 연구의 direct final selection. 새 S2-C budget allocator로 대체 사용 금지 |
| `src/pruning/s2_style_allocation.py::select_s2_style_tokens` | 1차 연구의 S2-style 통제용. Qwen 원본 재현과 구분 |
| Phase1.5 `adapter.py`, Phase2-A orchestrator, Phase2-B runtime | 576/D3584/24×24/G25/Qwen/BF16/M-RoPE 고정. Smol 실행 모듈로 직접 재사용 불가 |

## 7. 원본과 달라지는 점·경계 사례

원본의 single restored24×24/G25/K64 대신 **multiview local8R×8C/G16/K48 + global8×8/q16**을 사용한다. Local/Global 역할별 quota, view/native mapping 및1D Llama RoPE가 추가/변경된다. 이를 원본 Qwen2.5-VL 재현 성공으로 기록하지 않는다.

R,C≥1이면 local capacity는 최소64이고 각4×4 region도 nonempty여서48 배분이 가능하다. 실제 metadata와 V/placeholder 수가 맞지 않으면 자동 resize나 region 수 변경 없이 중단한다. no-local N64는 별도 K=N no-op stratum으로 기록하며 L/U/C 비교의 정보가 없음을 표시한다. 이 경로의 동작은 CPU 계약에서 먼저 확정한다.

K64는 기존832/1088 tokens의 약7.69%/5.88%만 남기는 강한 압축이다. 첫 Smoke에서는 구조 정합성을 확인하며 정확도 최적화 값으로 간주하지 않는다. K 또는 global quota 변경은 pilot 결과를 보고 main에 조용히 반영하지 않고 별도 protocol revision으로 잠근다.
