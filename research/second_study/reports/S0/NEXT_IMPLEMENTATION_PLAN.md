> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01/NEXT_IMPLEMENTATION_PLAN.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S0 — 결론과 다음 CPU 구현 범위

작성: 2026-11-14. **이번 단계는 이 폴더의 Markdown6개 작성으로 종료한다. 다음 구현·테스트·GPU 작업을 자동 시작하지 않는다.**

## 1. 네 가지 최종 결론

| 질문 | Phase S0 결론 |
|---|---|
| SmolVLM에서 Layer0 ERC Pruning이 가능한가? | **정적 구조상 가능.** 실제 cached architecture는 Idefics3+Llama이며 merger H0와 실제 decoder0 H1에서 native residual ERC를 얻을 수 있다. full Layer0 후 hidden/mask/original1D positions/**layer0KV**를 gather하는 별도 실행부가 필요하다. 구현/실행 성공은 아직 아님 |
| Local/Global을 어떻게 분할할 것인가? | **B 권고.** local view-major tokens를8R×8C original-space lattice로 대응시켜4×4 coarse regions로 분할. Global8×8은 독립 단일 region에 고정 quota를 두고 동일 Cell/ERC 사용 |
| K64와 Region 수는? | **최초 구현/Smoke의 사전 설정 K64=local48+global16. localG16+globalG1=coarse17, 최종cells64.** 성능 최적값으로 확정한 것이 아님. actual layout gate 후 실행 설정으로 잠금. no-local64는 별도 no-op |
| GTX1660SUPER 최소 GPU Smoke는? | 후속 CPU 계약 통과 뒤 old376959/N832에 고정2질문, native Full/custom Full/K=N no-op/L/U/C 순으로 최대12 generations(각16). exact no-op 및 all-layer compact KV/actual append 관측 우선. pilot 전 old436127/N1088의6 forward-only layout 검사 추가 |

## 2. 기존 조사 완료 범위

불필요한 전체 과거 로그를 읽지 않고 주요 보고서/manifest와 관련 함수만 확인했다. 보호 파일 SHA를 다시 계산하지 않았다.

| 분야 | 주요 파일·함수 |
|---|---|
| 기존 Smol inference | `examples/run_small_vlm.py::main`, `inspect_visual_tokens.py::main`, `inspect_llm_input.py::main` |
| 기존 physical pruning | `src/inference/pruned_vlm.py::build_pruned_inputs`, `generation_audit.py::GenerationAudit` |
| 1차 SpatialFloor/Coverage | `src/pruning/spatial_floor_coverage.py::local_region_indices`, `spatial_floor_coverage_select`, `coverage.py::global_coverage_select`, `local_gc_no_floor.py::local_gc_no_floor_select`, `s2_style_allocation.py` |
| S2 CPU | `src/s2_coverage_study/allocation.py`의 affinity/greedy/L/U, `selection_adapter.py::select_cells_with_budgets`, `reference.py::reference_allocation` |
| 원본 참조 | `third_party/S2Prune/s2prune/allocation.py::recursive_region_cells`, `qwen.py::s2prune_prefill`, `greedy_decode`; 고정 commit b0ac197e5e765f8a151669fdcbce10b621bbf44b, private HOLD |
| Phase1 | `results/s2_coverage_allocation_phase1/phase1_20261114_01/REPORT.md`, `CODE_FILES.json` |
| Phase1.5 | `results/s2_coverage_allocation_phase15/phase15_20261114_01/ADAPTER_SPEC.md`, `INTEGRATION_AUDIT.md`, `FINAL_RESULTS.json`, `CODE_FILES.json`; REPORT.md는 없음 |
| Phase2-A | `results/s2_coverage_allocation_phase2a/phase2a_20261114_01/REPORT.md`, `CODE_FILES.json` |
| Phase2-B | `results/s2_coverage_allocation_phase2b/phase2b_20261114_01/REPORT.md`, `CODE_FILES.json`, `RUNTIME_CONNECTION_AUDIT.md` |
| 기존 평가 | old30 Full 보고, old80 five-method 보고, LocalFloor paired 보고, old30/80 image manifests |
| 설치 모델/API | cached4개 config JSON, `modeling_idefics3.py`, image processing PIL/torchvision/processor 소스, `modeling_llama.py`, `cache_utils.py`, `masking_utils.py`, `generation/utils.py` |

Phase2-B `CODE_FILES.json`의11개 코드·symbol·과거 SHA/old_files_changed0/actual_GPU_executed=false/HOLD 기록을 참조했다. 이 manifest를 새로 실행하거나 과거205개 테스트를 재실행하지 않았다.

## 3. 권장 다음 단계: S1 CPU-only adapter와 계약 검증

다음은 **향후 사용자 지시 시 구현할 범위 제안**이다. 이번 S0에서 아래 파일이나 테스트를 생성하지 않았다. 기존 source를 수정하기보다 독립 namespace(예: `src/smol_s2_study/`)를 두는 편이 적절하다.

### S1-A: 순수 layout/metadata

- explicit R/C/8×8/global-tail을 받는 layout 객체와 native↔lattice↔prompt mapping.
- original normalized nominal boxes 및 coarse/cell ownership.
- no-local64, portrait/landscape/uneven region/최소 layout 및 metadata mismatch 처리.
- 모델 import·processor instantiate 없이 JSON/합성 tensors만 사용.

### S1-B: 공통 allocation/Cell/ERC glue

- 기존 CPU FP32 affinity/greedy helper, Laplacian/Uniform/recursive geometry를 reference로 재사용.
- local48/G16/globalq16 및 capacities 검증. S2-C local32 temporary histogram만 budgets에 반영.
- native ordinal tie를 보존하는 mapping-aware Cell selector. 최종 union native order 복원.
- supplied synthetic ERC에서만 대표 선택; 실제 decoder/ERC 계산은 아직 하지 않음.
- private reference import 경계를 유지하고 원본 함수 body를 새 package에 복사하지 않음.

### S1-C: sequence/position/cache mock 계약

- fake30-layer KV 모델에서 layer0 full fill→keep gather→upper empty fill→single-token append.
- gappy original1D positions와 compact physical length를 구분.
- suffix mask의 empty-upper-past 기준, packed-sequence 오인 방지, original position 보존.
- 모든 text/special 보존, batch1/unpadded 범위, fresh cache 및 method/질문 간 aliasing 차단.
- K=N no-op/shape mismatch/zero gain/tie/포화/nonfinite 등 의미 있는 경계 검증.

### S1-D: 새 평가 schema/manifest 준비 도구

- 기존 parser/scorer를 호출하되 old80 고정 validator/runner는 수정하지 않음.
- old30∪old80=110 exclusion과 pilot/main disjoint 검증.
- label별 공식 질문 선택·raw/strict/semantic·layout/budget/cell/ERC/length/timing metadata schema를 정의.
- 모델 없이 cohort lock 및 planned-run completeness를 검증할 수 있도록 설계.
- 실제 새 이미지 선정/manifest 생성은 해당 후속 지시의 범위에 맞춰 수행. S0에서는 후보390개의 가용성까지만 확인.

S1 완료 기준은 모델 없는 CPU contract 통과 및 모든 검증 한계 명시다. CPU mock 통과를 실제 Llama cache/RoPE 또는 GPU 동치로 부르지 않는다. 별도 namespace의 필요한 CPU 검증만 수행하고 보호 파일 전체 SHA 재계산을 기본 절차로 추가하지 않는다.

## 4. 다음 CPU 단계에서 제외할 작업

실제 model/processor load, inference, CUDA availability query/초기화, weights 접근, GPU prefill/decode adapter의 실제 연결/실행, generate 내부 변경, GPU-native Coverage, 패키지 설치·업그레이드, checkpoint/데이터 다운로드, Qwen runtime 변경, 기존 평가 재채점, Git commit/push는 S1 권장 범위에 포함하지 않는다.

실제 runtime binding 및 manual greedy driver 구현은 CPU contracts 검토 후 별도 S2 구현 단계로 분리하는 것을 권한다. "S1 CPU 테스트"를 명목으로 모델을 CPU에 로드하거나 inference하는 방식도 사용하지 않는다.

## 5. 미해결 사항과 후속 판정 시점

| 항목 | 현재 상태 | 판정 단계 |
|---|---|---|
| actual image backend | 설치 metadata상 torchvision 사용 가능, runtime class 미확인 | 실제 processor 준비 단계 |
| mapping-aware selector | 설계만 있음 | S1 CPU |
| Llama suffix mask/cache coordination | 설계만 있음 | S1 mock → S2 binding → GPU |
| native Full/custom Full/no-op numeric parity | 미검증 | 최초 GPU Smoke |
| logical position + compact cache 실제 append | 미검증 | 최초 GPU Smoke |
| GTX1660SUPER 새 경로 peak | 기존Full 약2.4GiB 근거만 있음 | 최초 GPU Smoke |
| K64/global16/G16 정확도 적합성 | 사전 시작 설정, 최적성 없음 | 새 pilot; 변경 시 protocol revision |
| 신규 pilot/main 정확한 manifest | 미선정 | 후속 데이터 준비 지시 |
| native generate pruned 경로 | 초기 지원 범위 밖, explicit positions만으로 충분한지 전체 state 검증 필요 | 후속 독립 확장 |
| S²Prune 공개/라이선스 | HOLD 유지 | 별도 권한/라이선스 검토 전 해제하지 않음 |

## 6. 이번 단계 산출물과 중단 상태

1. [SMOLVLM_ARCHITECTURE_AUDIT.md](SMOLVLM_ARCHITECTURE_AUDIT.md): 실제 checkpoint/installed API/토큰 좌표.
2. [SPATIAL_ALLOCATION_ADAPTATION.md](SPATIAL_ALLOCATION_ADAPTATION.md): A/B/C 비교, B 권고, 재사용 경계.
3. [LAYER0_ERC_FEASIBILITY.md](LAYER0_ERC_FEASIBILITY.md): native ERC/Layer0KV/positions/mask/decode 계약.
4. [GPU_MEMORY_AND_SMOKE_PLAN.md](GPU_MEMORY_AND_SMOKE_PLAN.md): 기존 실측과 추정 분리, 최소 Smoke.
5. [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md): 독립 신규 코호트/공정성/평가·성능 지표.
6. 이 문서: 결정·미해결·다음 CPU 구현 범위.

**S0 완료 후 중단. 신규 알고리즘 코드0, 테스트 실행0, 모델/inference0, CUDA/GPU0, 설치/다운로드0, 기존 연구 파일 수정0, commit/push0.** S²Prune HOLD 유지. 이는 새 문서 작성에 사용한 파일 읽기/편집과 구분한 작업 상태이며 새로운 실험 결과가 아니다.
