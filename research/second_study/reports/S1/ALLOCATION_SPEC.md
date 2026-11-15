> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s1/phase_s1_20261114_01/ALLOCATION_SPEC.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# S2-L / S2-U / S2-C Allocation 명세

## 공통 고정값 및 S0 변경 기록

local48, global16, total64, local4×4/G16, 최소1, 추가32를 고정한다. `allocate`는 image-only budget plan을 만들며 query weight 인자를 제공하지 않는다. ERC나 final tokens를 allocation 입력으로 받지 않는다.

**사용자의 S1 지시가 S0의 local-only demand 제안을 대체한다. 주 설정은 `demand_scope='all'`, 후보는 항상 Local이다.** S0 원문은 보존하고 이 문서에서 변경을 기록한다. Global16 선택은 S0의 공통 Cell/ERC 규칙 그대로다.

## S2-L

Phase1 `laplacian_region_budgets`와 private SHA-pinned reference allocator를 호출한다. 원본 RGB→Pillow grayscale→4-neighbor replicate Laplacian→원본 이미지에 비례 매핑한 region별 population variance→FP32 min-max→capacity-aware largest remainder/minimum1 규칙을 재사용한다.

변경된 것은 입력 geometry와 budget이다. Qwen24×24/G25/B64 대신 local8R×8C/G16/B48을 명시한다. crop/global resize image에서 별도로 Laplacian을 계산하거나 source pixels를 변경하지 않는다. 상수 image fallback과 정수 반올림/tie는 기존 reference를 유지한다.

## S2-U

같은16개 regions와 같은 capped allocator에 동일 weight를 넣는다.32개의 추가분이 균등하므로 충분한 capacities에서 q=[3]×16이다. 원저자 unavailable Uniform ablation의 exact 구현이라는 주장은 하지 않는다.

## S2-C 주 설정

전체 original projector feature V[N,D]를 CPU floating tensor로 받는다. implicit transfer를 거부하며 normalization 전에 FP32로 cast한다.

```text
z_i = FP32(V_i)/max(||FP32(V_i)||₂,1e-12)
S = z zᵀ                         # full N×N
A_ij = max(S_ij - mean_{k=0..N-1} S_ik, 0)
B = A[:, 0:Nlocal]               # N demands × Nlocal candidates
F(T) = sum_{i=0..N-1} max_{j in T} B_ij
```

Row mean에는 Global 열 및 자기 유사도도 포함된다. 후보를 먼저 Local로 자른 뒤 평균내는 다른 calibration을 사용하지 않는다. query weights/softmax/대각 제외가 없다. 전체 원본 affinity 수학은 기존 Phase1 FP32 helper를 재사용하고, greedy는 square-only helper를 수정하지 않고 **rectangular demand extension**으로 새로 구현했다.

빈 임시 집합에서32회 수행한다. candidate j의 marginal은 Σ_i max(B_ij−best_i,0), region별 추가 한도는 capacity_g−1다. exact gain tie는 낮은 native visual ordinal, zero gain에서도 예산을 채운다. epsilon tie나 사후Uniform fallback은 없다. q_g=1+temporary_count_g, 합48이며 Global q16은 고정이다. 임시32개는 final ERC 집합이 아니다.

## optional scope와 cache identity

`demand_scope='local'`은 Vlocal만으로 calibration하여[Nlocal,Nlocal] affinity를 만든다. 주 설정의 `all_visual`과 구분되는 `calibration_scope='local_only'`를 기록한다. 이는 S0 variant와의 민감도 비교이며 **demand 행뿐 아니라 calibration domain도 바뀐다.** 순수 demand-row ablation으로 해석하지 않는다. S1의 주 diagnostics/후속 실행 기본값은 all이다.

AllocationPlan은 method/layout/q/global quota/demand scope/calibration scope와 임시 histogram trace를 검증한다. Local-only features를 주 설정에 전달하거나 Global 인덱스를 임시 greedy 후보·Local budget histogram에 넣을 수 없다. 공통 Global16 최종 선택은 별도 Cell/ERC 경로에서 수행한다.

OfflineAllocationCache key는 layout/method/budget/scope/calibration, 전체 supplied feature dtype/shape/bytes, image mode/size/pixels 및 실제 grayscale bytes, provenance, 신규 및 재사용 helper/source identity를 포함한다. palette index bytes가 같아도 grayscale이 다르면 key가 다르다. 저장본은 affinity/best와 features를 제외한 lean plan이며 매 반환 시 deep copy한다. question/ERC/final indices/hidden/KV를 저장하지 않는다.

## 합성 진단의 실제 q

지역 순서는 coarse row-major다. 아래 값은 synthetic RGB/8D random features에서 얻었으며 자연 이미지/모델 feature가 아니다.

| layout | 방법 | q_local16 |
|---|---|---|
| 3×4 | L | 1,2,2,2,1,2,3,4,2,3,4,5,2,4,5,6 |
| 3×4 | U | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 |
| 3×4 | C/all | 4,5,2,3,1,3,2,2,1,3,1,4,3,6,6,2 |
| 4×4 | L | 1,2,2,2,1,2,3,4,2,3,4,5,2,4,5,6 |
| 4×4 | U | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 |
| 4×4 | C/all | 5,3,3,2,1,2,1,3,5,3,3,4,3,3,3,4 |

모두 sum48·capacity 이내다. optional local scope에서는 다른 q를 얻었으며 정확한 배열은 synthetic JSON의 scope_contrasts에 저장했다. 이를 POPE 성능 차이로 해석하지 않는다.
