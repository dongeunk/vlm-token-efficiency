> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3a/phase_s3a_20261114_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# SmolVLM S²Prune + Coverage — Phase S3-A 최종 보고서

2026-11-14. **PASS — 데이터 고정·평가 프로토콜·CPU 준비 범위.** 최종 CPU **138/138 PASS**, failures/errors/skips 모두0. 실제 CUDA 초기화·모델/processor 실행·inference/generation·다운로드는0회다. 실제 POPE 성능·TTFT·GPU 메모리는 이번 단계에서 측정하지 않았다. 여기서 중단한다.

## 고정 데이터

| 검증 항목 | 결과 |
|---|---|
| 공식 original POPE Random | 500이미지, 3000문항, Yes1500/No1500 |
| 기존 후보 manifest와 로컬 JPEG | 470 ID exact 일치 |
| old30 / old80 | unique30/80, 교집합0, 합집합110; 실제 이미지 파일 누락0 |
| exclusion 이후 후보 | 실제390장; JPEG decode·원본 SHA·크기·공식 Yes3/No3 검증 모두 통과 |
| 불완전/부적격 후보 | 0장; incomplete image는 선정 전 전체 제외하도록 사전 고정 |
| 파일럿 | **10이미지·60문항**, Yes30/No30, 4방법 최대240 generation |
| 본 평가 | **200이미지·1200문항**, Yes600/No600, 4방법 최대4800 generation |
| 집합 중복 | pilot/main/old110 모두 image-disjoint |

seed42와 eligibility·불완전 처리·무작위 추출·parser·bootstrap 규칙을 **선정 전에** `PROTOCOL_PRECOMMIT.json`에 저장했다. main200을 정렬된390 ID에서 먼저 단순 무작위 추출하고, 남은190장에서 pilot10을 골랐다. 새 표본에는 이미지 SHA, qid/원문/공식 label, annotation commit/file SHA/line number/line SHA/canonical record SHA, prompt/UTF-8 SHA를 보존했다. 예상 run keys도240/4800개로 고정했다. 기존 C/D 분류·annotation support·모델 응답을 선정 기준으로 쓰지 않았다.

핵심 파일:

- pilot_images.jsonl (local artifact: `pilot_images.jsonl`), pilot_questions.jsonl (local artifact: `pilot_questions.jsonl`)
- main_images.jsonl (local artifact: `main_images.jsonl`), main_questions.jsonl (local artifact: `main_questions.jsonl`)
- SELECTION_LOCK.json (local artifact: `SELECTION_LOCK.json`), DATA_AUDIT.json (local artifact: `DATA_AUDIT.json`), DATA_VALIDATION.json (local artifact: `DATA_VALIDATION.json`)
- PROTOCOL_PRECOMMIT.json (local artifact: `PROTOCOL_PRECOMMIT.json`), [EVALUATION_PROTOCOL.md](S3A/EVALUATION_PROTOCOL.md)

## Layout 판정과 중요한 미검증 사항

신규210장에 processor를 실행하지 않았다. actual_processor_visual_tokens/actual_sequence_length는 null이다. JPEG 크기와 설치된 resize 소스에 따른 **정적 예상값**은 pilot832×9/1088×1, main832×172/1088×27/**576×1**이다.

pilot anchors image554002(예상832)와 image134075(예상1088)는 실제 processor/GPU 확인 전이다. old80 metadata의 image376959/N832 및436127/N1088은 기존 증거로만 참조하며 새 평가에 넣지 않았다. S2-B 실제 GPU PASS는 한832 입력의 증거다.

main **image358255(640×234)**는 예상2×4/local512+global64=576이다. main 무작위 표본에서 제외하거나 교체하지 않았다. 합성576 CPU 계약은 추가 검증했지만, 이 이미지의 실제 processor 구조와576 GPU runtime은 미검증이다. main 이전 추가 기능 Gate가 필요하다.

## 평가 판정·분석 구현

신규 namespace는 `src/smol_s3_eval/`이다. 기존 parser/runner/S1/S2 runtime를 수정하지 않았다.

| 신규 파일 | 주요 함수·역할 |
|---|---|
| `lock.py` | `protocol`, `precommit`, `capture_protected`, `verify_protected`; write-once source/config lock |
| `dataset.py` | `audit_inputs`, `question_group_issues`, `choose_splits`, `prepare`, `validate_lock`; 기존 데이터 audit·manifest 고정 |
| `contracts.py` | `sequence_contract`, `validate_runtime`, `generation_diagnostics`, `mock_cache_prune_append`, `causal_allowed`; CPU metadata/cache 계약 |
| `scoring.py` | `parse_response`, `metric_bundle`, `score_complete`, `token_overlap`; 원문 parser adapter·완전 paired 평가·budget/overlap |
| `statistics.py` | `metrics_from_counts`, `paired_bootstrap`; image-cluster10000/seed42/95%CI |
| `cpu_guard.py` | `CPUOnlyGuard`; model stack import·network·subprocess·기존 파일 write 차단 |

전체 함수·라인·SHA 목록은 `CODE_MANIFEST.json`에 있다. CLI는 `examples/prepare_smol_s3a_cpu.py`, `verify_smol_s3a_cpu.py`, `analyze_smol_s3_cpu.py`, `finalize_smol_s3a_cpu.py`다. 마지막 analyzer는 후속 저장 결과용 CPU 도구이며 실제 GPU 결과에 실행하지 않았다.

주 지표는 **기존 frozen semantic parser `visual_semantic_yes_no_v1`의 POPE-style F1**, 주 비교는 **S2-C−S2-L**이다. POPE Accuracy는 Semantic Accuracy와 동일한 alias이며, Precision/Recall/F1/Invalid/Strict Accuracy를 함께 보고한다. 원본 POPE의 permissive parser와 다른 protocol이며 공식 전체 POPE 결과와 등치하지 않는다. strict/semantic parser 모두 raw text만 받는다. 제한된 semantic 문법이 clear paraphrase를 invalid로 둘 수 있는 한계도 사전에 명시했다.

invalid는 별도 confusion column에 유지하고 Accuracy에서 오답이다. invalid_yes는 Recall/F1의 FN에 포함하고, invalid_no를 TN으로 세지 않는다. positive F1이 invalid_no를 직접 벌점화하지 않는 점 때문에 Accuracy·Invalid·Strict를 함께 보고한다. missing/runtime error/OOM은 parser invalid와 구분하고, 불완전 run에는 완전한 지표·CI를 만들지 않는다.

EOS 없이16개 생성은 possible_truncation으로 기록하며 실제 문장 미완결이라고 단정하지 않는다. EOS가16번째이면 truncation으로 세지 않는다. raw text/IDs/EOS/종료 사유를 보존한다. 생성 길이 민감도는 추후 별도 실험이다.

bootstrap은 같은 이미지의6문항을 함께, 네 방법에 같은 resampling indices로10,000회 추출한다. sum-confusion 후 micro F1을 재계산한다. PCG64 seed42, percentile95% CI, linear quantile다. 이미지별 F1 평균은 사용하지 않는다. 다른 대비는 exploratory다.

기록 스키마와 CPU 분석 사용법은 [RUN_RECORD_SCHEMA.md](S3A/RUN_RECORD_SCHEMA.md)에 있다. paired item/image 결과, actual sequence positions·모든 비시각 token 보존, 지역 q[16], 최종 native visual ordinals, local/global overlap, 실제30층 prefill/final cache 길이·logical decode positions와 지연/메모리 필드를 준비했다. 실제 Feature Coverage·Cell/ERC trace를 만드는 GPU writer는 S3-B에서 별도로 필요하다.

## CPU 검증 증거와 범위

최종 evidence: `cpu_run_03/TEST_RESULTS.json`, `TEST_OUTPUT.txt`, `SOURCE_LOCK.json`, `CPU_BOUNDARY.json`, `SYNTHETIC_BOOTSTRAP_10000.json`.

- **138/138 PASS**, failures0/errors0/skips0. CPU boundary에서 forbidden model modules0, blocked attempts0. 실행 시간 약1.14초는 CPU unittest suite 시간이며 모델 latency가 아니다.
- parser 수작업 정답·제한 문법·부정/불확실/empty·strict/semantic 차이; 여섯 confusion cell과 invalid 포함 metric oracle.
- 실제 후보·고정 manifest·SHA 링크·old110/pilot/main disjoint·6문항3yes/3no·seed 결정성.
- 합성832/1088/576 및 가변T, 원래 visual/nonvisual 위치, local region capacity,64개 local48/global16, original1D positions·causal scalar oracle.
- 합성30층 list KV의 Layer0 gather·m−1 append·physical/logical position 분리. **실제 HF cache 실행 검증이 아니다.** 저장된 실제832 input_ids만 CPU에서 읽어878→110 계약을 확인했다.
- 완전4방법 paired 결과, 입력 hash 공통성·global16 공통성, missing/duplicate/foreign/error/잘못된 cache/mask/길이/예산 거부.
- 독립 scalar formula와3이미지 exhaustive27 resample oracle, seed42/10000 반복 결정성, 합성200이미지×10000 bootstrap workload.

첫 `cpu_run_01`은126개 중8 errors였다. NumPy testing helper가 Windows platform 조회 subprocess를 시도하면서 엄격한 boundary가 NUL open을 차단한 테스트 도구 문제다. 도우미를 표준 Python scalar 단언으로 교체했고 guard를 완화하지 않았다. `cpu_run_02`는136/136 PASS, 이후 새576 입력 검증2개를 추가한 `cpu_run_03`이138/138 PASS다. 실패와 중간 evidence도 보존했다. 데이터·seed·선정 manifest는 재선정하지 않았다.

합성 진단의 임의 정답·F1·bootstrap 값은 실제 POPE 정확도나 SmolVLM 성능을 의미하지 않는다. 실제 prediction file/POPE 평가 결과는 생성하지 않았다.

## 검토한 기존 주요 파일·함수

- S0 `SMOLVLM_ARCHITECTURE_AUDIT.md`, `EXPERIMENT_PROTOCOL.md`의 checkpoint/processor/layout/평가 계약; S1/S2-A/S2-B `REPORT.md`, S2-B `S3_ENTRY_CONDITIONS.md`.
- `src/evaluation/pope.py::{load_annotations,parse_yes_no,compute_metrics}`, `semantic_yes_no.py::explain_semantic_yes_no`. 문제점은 검토했으나 source는 변경하지 않았다.
- `src/smol_s2_study/mapping.py::{VisualLayout,require_layout}`, `selection.py::select`, allocation/Cell/ERC·prefill 계약. 평가 모듈에서는 torch를 import하는 S1 코드를 실행하지 않았다.
- `src/smol_s2_runtime/plan.py`의 checkpoint/generation 계약, `runtime.py::_validate_runtime`; `src/smol_s2_pruning_runtime/runtime.py::{_validate,compute_native_erc}`와 boundary 계약. 기존 제한된 GPU runner를 범용 POPE runner로 사용하지 않았다.
- cached config/processor/chat template JSON과 설치된 Idefics3의 `_resize_output_size_rescale_to_max_len`, `resize_for_vision_encoder` 소스는 정적으로 확인했다. 가중치는 읽지 않았다.
- old30/old80 exclusion manifest, candidate_pool.csv, 470 screening의 source SHA/크기, old80 cpu_images metadata, S2-A `ACTUAL_INPUT.json`, 공식 annotation/SOURCE.json. 과거 예측 로그 전체나 Qwen 연구 전체는 재분석하지 않았다.

## 기존 연구 보호

`PROTECTED_BEFORE.json`과 `INTEGRITY_AFTER.json`은 이번에 검토한 **46개 기존 파일**의 SHA가 동일함을 기록한다. 프로젝트 전체 SHA 전수검사를 하지 않았다. 새 CPU 프로세스는 신규 S3-A 결과 폴더 밖 파일 쓰기를 차단했고 Python은 `-B`로 실행했다. 기존 평가 코드·결과·old30/80 데이터·Qwen·S0/S1/S2-A/S2-B·checkpoint를 수정하지 않았다. cached config8개도 동일하며 weights는 재해시하지 않았다. 미열람 과거 파일 전체를 독립적으로 전수 검증했다는 주장은 하지 않는다. S²Prune HOLD, commit/push 없음, 설치/다운로드/유료 서비스 없음.

## 다음 단계와 중단 상태

[PHASE_S3B_GPU_GATE_AND_PILOT_PLAN.md](S3A/PHASE_S3B_GPU_GATE_AND_PILOT_PLAN.md)와 `S3B_GATE_PLAN.json`에 Gate A→B→C→D를 고정했다.

| 항목 | 상태 |
|---|---|
| 데이터 lock·중복·annotation·CPU 평가/통계 준비 | **PASS** |
| 기존46파일·cached config8파일 보존 | **PASS** |
| 신규 metadata-general S3 runtime/GPU writer | **미구현 — 다음 단계 작업** |
| 실제 새832/1088·가변T·cache/decode GPU Gate | **미검증 / 미실행** |
| main358255 예상576 실제 processor·GPU Gate | **미검증 / 미실행** |
| pilot240 generation·실제 정확도/latency/VRAM | **미실행** |
| main4800 generation | **미실행 / 별도 승인 필요** |

다음 권장 범위는 기존 S1/S2 코드를 보호한 별도 S3 runtime의 metadata generalization과 record writer 구현·CPU 검증, 이후 승인된 Gate A의 실제832/1088·가변T 기능 확인이다. Gate B pilot은 Gate A 통과 뒤에만, main은 Gate C 검토와 별도 승인 및576 등 실제 대상 layout 검증 뒤에만 진행한다. **이번 작업은 S3-A 완료로 중단하며 추가 GPU 작업을 시작하지 않는다.**
