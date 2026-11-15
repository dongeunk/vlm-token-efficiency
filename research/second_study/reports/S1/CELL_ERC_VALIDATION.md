> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s1/phase_s1_20261114_01/CELL_ERC_VALIDATION.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# 공통 Cell Partition과 Synthetic ERC 검증

## 고정 선택 규칙

L/U/C의 local q16만 다르다. 동일 `select(layout,plan,erc)`가 아래 경로를 처리한다.

1. Local16 coarse rectangles 각각을 q_g개의 nonempty cells로 나눈다.
2. Global8×8 단일 region을 고정16 cells로 나눈다.
3. 각 cell을 native visual ordinals로 변환하고 오름차순 정렬한다.
4. supplied synthetic ERC 최대1개를 고른다. exact tie는 최소 native ordinal이다.
5. 최종64개를 native ordinal 순서로 정렬하고 membership/unique/range를 검사한다.

private reference `recursive_region_cells`를 그대로 호출하며 body를 복사하지 않는다. 가장 큰 splittable cell부터 나누고 면적 동률은 현재 leaf 순서, 긴 변 분할/길이 동률 row 우선/midpoint floor 규칙을 유지한다. Qwen24×24를 강제하지 않고 실제 local8R×8C와 global8×8에서 같은 geometry 함수를 사용한다.

Global16의 **예산·분할·ERC/tie 규칙**은 방법 공통이다. 입력 ERC가 같으면 최종 global ordinals도 동일하다. Global이 fixed16이라는 것은 위치를 정해 두고 uniform subsample한다는 뜻이 아니다.

## 검증 결과

CellERCTests8개와 기존 Phase1 CellPartition/OriginalParity/SyntheticGather tests가 통과했다.

- local48+global16=64 cells, cell별1개, 전체 original N을 정확히 한 번 덮는다.
- coarse별 local cell/대표 수=q_g, global cell/대표 수16.
- 모든 candidates가 region 내부에 있고 cells 간 중복·누락0.
- 독립 scalar priority-queue recursive oracle와 rectangle 목록 일치.
- native와 lattice 순서가 뒤집히는 crop-crossing cell에서 tied maxima를 만들고 **최저 native ordinal**을 선택하는지 검증.
- final64 sorted/unique/range, token_to_cell과 cell_selected_indices 일치.
- final ERC 선택을 임시Coverage32 그대로 사용하지 않음.
- 잘못된 q/layout/nonfinite/shape/negative ERC 거부.

새 ERC 인터페이스는 실제 residual norm의 범위에 맞춰 nonnegative score를 요구한다. 과거 Phase1 단일-grid 테스트가 허용했던 임의 negative synthetic ERC를 Smol runtime score로 받지 않는다.

## ERC 인터페이스

`compute_erc_scores(H0,H1,visual_positions)`는 공급된 matching CPU `[1,L,D]` tensors만 받는다. native dtype subtraction 후 FP32 L2 norm을 수행하고[N]을 반환한다. H0/H1 dtype이 다르거나 native subtraction/FP32 norm에서 nonfinite가 생기면 중단한다. 실제 decoder 또는 final norm을 호출하지 않는다.

FP16 native subtraction과 FP32-first subtraction이 달라지는 fixture, visual rows만의 gather, overflow 검사가 통과했다. 이 결과는 실제 Llama Layer0 hidden에 연결한 ERC 검증이 아니다.

## 기록과 해석

`cpu_run_02/SYNTHETIC_DIAGNOSTICS.json`의6 records에는 q/cells/candidates/token_to_cell/ERC[N]/selected ordinals/원본 nominal 좌표를 저장했다. 두 layouts에서 각각 L/U/C의 Global16 집합은 exact 동일했다.

최종 F_all 및 F_local은 각 공통 affinity에서 **최종 ERC tokens**로 계산했다. Local/all calibration domain을 분리해 normalized 값도 저장했다. 합성 진단6건에서 C의 F가 높게 나온 사실은 이 synthetic RGB/features/score fixture에 한정된다. 실제 POPE Accuracy/F1·SmolVLM 출력·TTFT 우위를 주장하지 않는다.
