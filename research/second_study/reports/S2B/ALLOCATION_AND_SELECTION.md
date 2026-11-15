> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s2b/phase_s2b_20261114_01/ALLOCATION_AND_SELECTION.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Actual Allocation·Cell/ERC·Feature Coverage 진단

## 변경하지 않은 알고리즘

기존 S1 `allocate`와 `select`를 직접 호출했다. L은 원본 이미지 Laplacian/min-max/capped largest remainder, U는 같은16 regions의 동일 weight, C는 projector-space query-independent Coverage다. Global은8×8 단일 region→공통 recursive16cells→actual ERC argmax다.

C의 FP32 cosine은 전체832×832로 만들고 **전체 열 평균**으로 row-centering/clipping한 뒤 Local768열을 후보로 제한한다. demand832 모두 weight1, 후보는Local만, 추가 capacity−1 한도에서 temporary32개를 골라 q=1+count를 만든다. Query Attention은 사용하지 않는다. temporary32는 final ERC64의 대체물이 아니다.

## 실제 budgets

| 방법 | q_local16(row-major), 합48 |
|---|---|
| L | 1,4,3,2,5,1,2,2,3,5,4,2,1,5,6,2 |
| U | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 |
| C | 2,3,2,4,4,4,1,2,4,4,3,4,2,2,3,4 |

Local coarse lattice는24×32이고16개 regions의 capacity는 각각48이다. 모든 q는 capacity 이내이며 Global q16과 합쳐 final64cells를 만든다.

C temporary native visual ordinals(greedy order):

```text
[195,651,510,319,193,515,620,686,759,122,718,357,342,352,49,520,
 449,204,486,281,105,744,682,374,550,22,154,274,120,94,728,425]
```

이32개의 histogram만 budget으로 사용하고, actual Layer0 ERC로 final64개를 따로 선택한다.

## actual ERC

actual CUDA FP16 H1_visual−H0_visual을 먼저 계산하고 FP32로 올려 L2 norm을 구했다. score[N832]는6회 모두 finite이며 같은 입력이므로 방법 간에도 exact 같았다.

| statistic | 값 |
|---|---:|
| min | 19.057848 |
| Q25 | 25.494242 |
| median(quantile0.5) | 27.995129 |
| mean | 28.194265 |
| population std | 3.854837 |
| Q75 | 30.699400 |
| max | 42.632877 |
| finite / zero | 832 / 0 |

각 Cell의 후보를 native ordinal로 정렬하고 동점에서는 최소 native index를 선택한다. 사전 준비한 Cell과 S1 selector의 Cell이 exact 일치했고, 전체832개 tokens의 중복·누락0, 모든 Cell 대표의 actual ERC 최대 조건을 확인했다.

## 최종 token identity

여기서 token IDs는 original projected V의 **visual ordinals**를 뜻하며, 모든 image placeholder의 어휘ID49190이나 generated word IDs와 구분한다. 각 결과에 Local48/Global16의 native ordinals와 원래 LLM sequence positions를 저장했다.

Global16은 모든 방법·반복에서 다음과 같은 집합이었다.

```text
[770,773,776,783,786,789,792,799,804,809,810,815,818,824,829,831]
```

Local48은 방법별로 달랐다. 전체 배열은 `METHOD_SELECTIONS.json`과 각 `*_RESULT.json`에, 전체 Cell/candidates/membership 및 ERC832는 raw evidence에 보존했다.

## Final Feature Coverage

공통 all-N affinity에서 final ERC64를 평가한 F_all과, 별도의 local-only calibration에서 final Local48을 평가한 F_local을 구분한다. 둘 다 생성 후 CPU FP32로 측정하며 runtime gate의 선택을 바꾸지 않는다.

| 方法 | F_all | F_all/832 | F_local | F_local/768 |
|---|---:|---:|---:|---:|
| L | 104.641022 | 0.12577046 | 87.241104 | 0.11359519 |
| U | 104.075966 | 0.12509131 | 81.763855 | 0.10646335 |
| C | 105.054344 | 0.12626724 | 83.538063 | 0.10877352 |

반복2에서도 같은 값이었다. C의 F_all이 높지만 F_local은 L보다 낮다. 이 한 이미지의 기술통계를 VQA Accuracy나 일반적 우위로 해석하지 않는다.

## 최종 집합 overlap

| pair | intersection/64 | union | Jaccard | local intersection | global intersection |
|---|---:|---:|---:|---:|---:|
| L–U | 53 | 75 | 0.706667 | 37 | 16 |
| L–C | 52 | 76 | 0.684211 | 36 | 16 |
| U–C | 57 | 71 | 0.802817 | 41 | 16 |

방법 간 선택이나 답변 차이는 실패 조건으로 삼지 않았다. 같은 방법의 반복만 exact gate로 삼아 budget/ERC/선택/hidden/logits/finalKV/IDs를 비교했고 모두 통과했다. POPE parser/scorer는 호출하지 않았다.
