> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01/EXPERIMENT_PROTOCOL.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S0 — SmolVLM 2차 연구 실험 프로토콜

작성: 2026-11-14. **사전 설계 제안이며 데이터 선정·추론·재채점·테스트는 실행하지 않았다.** 기존 평가 결과와 Qwen 연구는 보존한다.

## 1. 연구 질문과 트랙 구분

주 질문은 "동일 SmolVLM Layer0 ERC pruning에서 Local region budget을 Laplacian/Uniform/Coverage로 배분하면 어떤 차이가 생기는가"다.

| 트랙 | 현재 상태 | 이 연구가 주장할 수 있는 범위 |
|---|---|---|
| Qwen2.5-VL-7B 원본 S²Prune 재현 | Phase2-B 준비 및 과거 CPU205/205 보고, 실제 GPU0회 | 원본 GPU parity/성능은 계속 미검증 |
| 1차 SmolVLM Spatial Floor+Coverage | 기존 GPU 평가 완료 | pre-LLM pruning 결과이며 Layer0 ERC 근거가 아님 |
| **2차 SmolVLM S2-L/U/C 이식** | **S0 정적 분석·설계 완료** | 다중 view/Global 고정 quota/Llama1D positions를 사용하는 별도 이식 실험 |

이번 문서의 S2-U는 Reconstructed Uniform, S2-C는 CoverPruner 계열의 query-independent calibrated Coverage를 **budget에만** 사용하는 변형이다. 원 논문 모델/환경/공개 ablation과 exact 재현이라고 표시하지 않는다.

## 2. 기존 증거와 해석 한계

| 기존 실험 | 주요 관측 | 해석 |
|---|---|---|
| old30 Full pilot | strict18/30, frozen semantic26/30=86.67%, semantic invalid2/30 | parser 재해석이지 추가 inference 개선이 아님 |
| old80/160, five methods | semantic Acc Random77.5%, PureGC80.625%, Ours81.875%, LocalFloor81.875%, LaplacianQuota78.125% | 새160문항 Full baseline 없음. Ours−GC+1.25pp, CI[-1.875,+4.375]로 우위 입증 아님 |
| old80 Local-only NoFloor vs Floor | 둘 다131/160=81.875%, F1≈0.8085, invalid3/160; 차이0pp CI[-3.750,+3.125] | 동일80을 본 뒤 설계한 탐색 통제. 독립 holdout 아님 |
| 같은 Local 통제의 spatial/coverage | zero-region10/80→0/80, F(S)153.876474→152.428513 | 공간 보장이 정확도 향상이나 Coverage 극대화와 같지 않음 |

1차 SpatialFloor는 지역 medoid floor 뒤 direct global greedy selection이다. 새 S2-C는 임시 Coverage histogram으로 q를 만든 뒤 Cell/ERC를 적용한다. 기존 Floor vs NoFloor 결과를 새 S2-L/U/C 결론으로 합치지 않는다. 기존 통제는 local candidates만 쓰더라도 demand는 global 포함 all-N이었으며, 새 local-only allocation affinity와 calibration domain도 다르다.

근거:

- `results/pope_pilot/full_model_report.md` 및 기존 semantic 재채점 요약.
- `results/pope_spatial_floor_80_evaluation/full_v2_20261024_01/REPORT.md`.
- `results/pope_local_floor_paired/full_v1_20261025_01/REPORT.md`.

## 3. 고정 비교 조건

주 평가 arm은 **Full, S2-L, S2-U, S2-C**다. Full은 압축 손실의 공통 baseline이고, L/U/C가 예산 배분만 바뀌는 통제 비교다. K=N no-op는 Smoke 검증용이며 정규 Accuracy arm에 중복 추가하지 않는다.

| 조건 | L/U/C 공통 |
|---|---|
| model | cached SmolVLM revision7e3e67edbbed1bf9888184d9df282b700a323964 |
| image path | 같은 원본 JPEG/RGB, 같은 processor/backend/resize/splitting |
| original V | 같은 전체 local+global projector token 집합, native ordering |
| budget | local48+global16=64 |
| regions | local4×4/G16, global single region/G1 |
| Cell | 같은 recursive rectangle partition 함수·tie |
| ERC | 같은 full layer0 native residual, 동일 Cell argmax |
| 삭제 | Layer0 이후 hidden/mask/original position/layer0KV gather |
| 보존 | 모든 text/special token과 original order |
| execution | FP16/eager, batch1, fresh KV, 같은 CPU FP32 allocation backend |
| decoding | greedy16, beams1, EOS49279/PAD2/BOS0, 같은 explicit driver |
| scoring | 동일 strict+semantic parser 및 invalid 처리 |

결과를 보고 Global quota, K, G, parser, prompt, EOS/stop을 방법별로 바꾸지 않는다. Full의 decoder/stop 정책도 L/U/C와 맞추고 native generate와의 parity를 먼저 확인한다. 추가 Yes/No 지시문은 원문 질문에 붙이지 않는다.

## 4. 데이터 제외와 신규 코호트 제안

정확한 이미지 단위 제외 입력:

1. `results/pope_pilot/sample_manifest.jsonl`: old30 unique images.
2. `results/pope_spatial_floor_80_preparation/cohort80_five_methods_v2/selected_images.jsonl`: old80 unique images.

두 집합의 교집합0, 합집합110은 파일의 image_id를 읽어 확인했다. 질문만 바꾸어 같은 이미지를 새 평가에 넣지 않는다. Smoke376959/436127도 old80 소속이므로 기능 검사에만 재사용한다.

`data/coco/pope_random_new470_v1/`에는 old30을 제외한470 JPEG가 있고 old80까지 제외하면 **390개 신규 이미지 후보**가 남는다. 이는 파일명/manifest 기반 가용성 확인이며390개의 새 pixel 검증·모델 전처리를 이번에 실행한 것은 아니다. 공식 질문은 기존 `data/pope/coco_pope_random.json`을 사용한다. 새 다운로드는 필요하지 않은 상태다.

후속 제안:

- **pilot20이미지/40문항**, Full/L/U/C로160 generation.
- **main80이미지/160문항**, Full/L/U/C로640 generation.
- old110/pilot20/main80은 image_id 기준 모두 disjoint. pilot과 main을 동일 seed 규칙으로 미리 나눠 holdout을 예약한다.
- canonical image ID 정렬 후 seed42 기반의 결정적 표본 추출 규칙을 새 metadata에 잠근다. 정확한 RNG/정렬 구현은 다음 CPU 단계에서 명시한다.
- 이미지당 공식 Yes1/No1. label별 공식 후보 중 seed+image_id+label에 따른 결정적 균등 선택을 사용하고 원문 text/label/question_id를 보존한다. COCO object annotation support나 모델 답변에 따라 질문을 거르지 않는다.
- primary는 신규 후보에서의 사전 무작위 코호트다. 기존80의 C/D 분류/그룹 균형을 본 실험의 eligibility 조건으로 재사용하지 않는다. N/layout/aspect별 결과는 보조 기술통계로 기록할 수 있다.

샘플 수는 검정력 계산으로 보장된 크기가 아닌 로컬 비용을 고려한 파일럿/본 실험 제안이다. S0에서는 실제 sample manifest를 만들거나 이미지를 선정하지 않았다.

## 5. 단계별 gate

| 단계 | 목적·통과 조건 |
|---|---|
| S1 CPU | synthetic mapping/budget/cells/ERC/gather/cache contracts; 모델/CUDA 없이 수행하는 후속 구현 범위 |
| 최초 GPU Smoke | old376959 + 공식질문/description, native Full→custom Full→no-op→L/U/C. 상세 호출 수는 GPU 계획 참조 |
| second-layout smoke | old436127/N1088 first-logit/shape/cache 검사. 첫 image의 append 관측을 대체하지 않음 |
| 새 pilot20 | 실행·semantic invalid·memory·timing schema 점검. S2-C accuracy 우위를 통과 조건으로 삼지 않음 |
| main80 | protocol/schema/backend/파라미터를 잠근 후 새 holdout에서 실행 |

pilot에서 구현 결함을 고치면 version과 변경 이유를 기록한다. K/G/quota나 평가 규칙을 수정하면 protocol revision을 만들고 변경 전후 pilot을 분리한다. main 결과를 보고 설정을 선택하지 않는다. failure/missing 결과를 조용히 제외하지 않으며 incomplete run은 complete Accuracy와 구분한다.

## 6. 지표·기록 schema

### 답변 지표

Primary는 frozen `visual_semantic_yes_no_v1`의 Accuracy/F1 및 semantic invalid ratio다. strict 결과도 함께 보존한다. parser 입력은 raw answer뿐이며 질문·GT·image_id를 전달하지 않는다.

- Accuracy=(TP+TN)/전체 문항 수. invalid는 오답이며 No로 치환하지 않는다.
- Precision=TP/(TP+FP), Recall=TP/(TP+FN+invalid_yes). zero-denominator 규칙은 기존 `compute_metrics`를 따른다.
- F1, Yes prediction ratio, yes/no class accuracy, invalid_yes/invalid_no 및 혼동표를 함께 보고한다.
- truncation/EOS/불확실·미완결 등 semantic rule_id도 기록한다. inference 오류를 semantic invalid와 섞지 않는다.

Primary contrast는 C−L Accuracy다. C−U 및 L−U, F1/세부 strata는 secondary/exploratory로 명시한다. 이미지 단위 paired bootstrap10,000회/seed20261114의95% CI를 제안한다. Yes/No 두 문항을 같은 image cluster로 resample한다. 여러 secondary 비교에서 우위 주장을 하면 별도 다중비교 기준을 사전 고정한다.

### 선택·공간 진단

image/question/method별 N/R/C/T/K, local budgets 배열 q_local16 (local artifact: `합48`)/global quota16, capacities, final cells의 coordinates/membership, temporary Coverage32 및 greedy gain/tie trace, actual ERC[N], final native ordinals, original nominal boxes를 보존한다. 최종 Feature Coverage는 `F_local`, `F_all` 및 각 normalized 값을 공통 정의로 기록한다.

L/U/C pruning arm의 Layer0 length L와 Layer1 이후 T+64, 각 layer KV length/append, kept text/special exact flags를 별도 기록한다. Full은 모든 layers의 입력 길이 L을 유지하고 K=N no-op도 동일하다. decoder0 이전 pruning 결과와 같은 "sequence length" 열에 섞지 않는다.

### 시간·메모리

- E2E TTFT: 로컬 이미지 decode/전처리 시작부터 첫 generated token까지. model load 제외. CPU↔GPU transfer와 selection을 포함한다.
- decoder-only TTFT: merged H0 준비 직후부터 첫 token까지. allocation을 아직 수행하지 않은 지점에서 시작하여 CPU handoff/affinity/budget/Cell, dense Layer0/ERC/최종선택/suffix/head를 모두 포함한다. Full도 같은 H0 준비 직후가 시작점이다. 사전 계산한 allocation을 재사용했다면 별도 cached 지표로 표시한다.
- Selection Overhead: Laplacian/affinity/greedy-budget, Cell partition, ERC vector 처리/Cell argmax, 관련 device handoff를 구간별로 기록한다. Layer0 공통 연산 시간은 별도 표시해 이중 합산을 피한다.
- Decode: 첫 token 이후 시간, 실제 forwarded/appended token 수, 종료 사유. token 수 차이를 숨긴 총시간 비교를 하지 않는다.
- GPU peak: allocated/reserved, reset 이전 baseline, view count, 실제 dtype/backend, Windows 환경 점유를 구분한다.

기능 Smoke의 hooks/CPU snapshots/hash/I/O 시간은 기능 기록이다. 성능 비교는 별도 사전 반복계획으로 **방법별 warm-up1회+측정3회**, 동일 입력 subset 및 method 순서 회전을 잠근 뒤 수행하는 후속 제안이다. native/custom/no-op Smoke를 성능 반복으로 세지 않는다. cold image E2E에는 feature/allocation cache를 끄고, 공유 feature decoder-only 측정은 별도 표로 표시한다. GPU 동기화 없는 perf_counter 값을 TTFT로 보고하지 않는다.

## 7. 재사용 가능한 평가 코드와 직접 재사용 금지 지점

| 파일·함수 | 사용 범위 |
|---|---|
| `src/evaluation/pope.py::parse_yes_no:180`, `compute_metrics:204` | 기존 parsing/scoring 수학 유지 |
| `src/evaluation/semantic_yes_no.py::explain_semantic_yes_no:74` | frozen raw-response semantic parsing |
| `src/evaluation/pope80.py::validate_cohort:87` | provenance/check 패턴 참고. old80/methods/K64/G16/기존그룹 고정이므로 새 cohort validator 필요 |
| `src/evaluation/local_floor_paired.py::scope_rows:13`, `validate_record:40` | 기존160 통제 결과 확인용; 새 연구 runner로 사용 금지 |
| `src/evaluation/controlled_comparison.py::validate_full_baseline:121`, `build_comparison_report:202` | old30 기준/기존 schema 고정. 새 manifest/schema adapter 필요 |
| `examples/run_pope_pilot.py::GENERATION:39` | 기존 greedy16 및 원문 질문 조건 근거 |
| `examples/run_pope_pruning_smoke.py::choose_cases:49`, `measured:77`, `generate_record:142` | outcome-blind smoke 선정과 실제 관측 패턴 참고. 기존 measured는 total generation이고 TTFT가 아님 |
| `src/inference/generation_audit.py::GenerationAudit` | cache identity/append 관측 원칙 참고. contiguous position 전제는 새 audit로 교체 |

기존 파일·보호 manifest·평가 산출물은 수정하지 않는다. source/설정/신규 입력의 잠금은 다음 단계에서 필요한 범위만 수행하며 기존 보호 파일 전체 SHA를 매 단계 다시 계산하지 않는다. 신규 결과는 별도 versioned 폴더에만 작성한다.
