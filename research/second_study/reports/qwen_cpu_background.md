> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/s2_coverage_allocation_phase2b/phase2b_20261114_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# S²Prune + Coverage — Phase 2-B 실행 준비 보고서

2026-11-14. **원격 GPU Smoke 실행 코드·잠금·관측·중단 절차를 준비했다. 실제 GPU Smoke를 실행하지 않았다.** 모델/데이터 다운로드·설치·원격 접속·임대·generation은0회다.

## 1. 검증 결과

| 범위 | 결과 |
|---|---:|
| 기존 Phase1/1.5/2-A CPU tests | 143/143 통과 |
| 신규 Phase2-B metadata·lock·ordered compare·trace·ERC contract tests | 62/62 통과 |
| 합계 | **205/205**, failure/error/skip0 |
| 실행 | `cpu_run_03/`,7.7794초 |
| CLI help | run/lock 두 entrypoint exit0; torch/transformers import0 |
| CUDA initialized / entrypoint attempt | False /0 |
| 실제 model load / inference / generation / cloud connection | 전부0 |

Phase1 golden과 기존 원본 mock parity는 기존143개 suite에서 재통과했다. Phase2-A의 기존24개 저장 진단은 원본 SHA/manifest로 보존·검증하며 이번에 실제 Qwen 검사로 대체하지 않는다. 신규 tests는 tiny CPU tensor·직접 입력한 trace·임시 bytes만 사용한다. 원격 loader나 실제 hook을 모델에 연결한 성공을 주장하지 않는다.

보호 대상 기존2,232개 파일은 사전·사후 크기/SHA를 대조한다. Phase0/1/1.5/2-A manifest도 각각 재대조하며 최종 결과는 `ORIGINAL_FILES_VERIFICATION.json`에 보존한다. 기존 코드·원본 S²Prune·결과·manifest를 수정하지 않는다.

## 2. 신규 코드11개

| 파일 | 역할 |
|---|---|
| `src/s2_gpu_smoke/__init__.py` | private namespace·HOLD 경계 |
| `src/s2_gpu_smoke/plan.py` | stdlib file/source/spec lock, remote launch gate,5방법 순서 gate |
| `src/s2_gpu_smoke/environment.py` | 제안 direct pin과 실제 설치 metadata 전체 lock의 exact 대조 |
| `src/s2_gpu_smoke/compare.py` | q→Cell→ERC→index→prefill/cache→logits→IDs 최초 불일치 기록 |
| `src/s2_gpu_smoke/contracts.py` | 기존 Phase1 geometry/Cell/ERC helper로 실제 선택 재검수 |
| `src/s2_gpu_smoke/observer.py` | read-only actual layer/rotary/mask/KV/append/logit hooks와 tokenizer proxy |
| `src/s2_gpu_smoke/runner.py` | 별도 승인 후 실행할 실제 private driver, source불변·freshcache·nativeFull/L parity·실패 보존 |
| `examples/lock_s2_gpu_smoke.py` | 이미 존재하는 snapshot/image/question/code/environment를 잠금. 설치·다운로드 없음 |
| `examples/run_s2_gpu_smoke.py` | `--plan-only`는 stdlib만, 실제 실행은 explicit flag/승인/원격 Linux gate |
| `examples/inspect_s2_gpu_smoke_cpu.py` | 모델 없는 CPU regression·실행 source snapshot |
| `tests/test_s2_gpu_smoke.py` | 신규62개 unit tests |

source SHA·함수 위치는 `CODE_FILES.json`, 실행 당시 사본은 `cpu_run_03/executed_sources/`에 있다. 원본 vendor body를 이 새 폴더에 복사하지 않았다. 추가 metadata 봉인 도구 `finalize_artifacts.py`는 결과 폴더 내부에만 두며 모델/알고리즘을 실행하지 않는다.

## 3. 실제 실행 준비 상태

고정 조건은 Qwen2.5-VL-7B-Instruct, BF16/eager,576 visual/24×24 grid, B64/G25, batch1, greedy16, fresh KV다. L/U/C의 변경은 기존 지역 budget 산출만이며 Cell/ERC/Layer0 이후 삭제 경로는 동일하다. 원본 참조 소스는 HOLD를 유지한다.

전체576 토큰의 Layer0, native ERC, 실제 Layer1 input gather, nonvisual·position 보존, 실제 cos/sin·causal mask, KV physical/public length·counter·append를 관측하도록 준비했다. 원본 함수·symbol을 monkey-patch하지 않는다. Full은 allocation/ERC/삭제를 호출하지 않는 별도 경로이며 native outer-model prefill과 먼저 비교한다.

원본과 S2-L은 예산 배열 → Cell → ERC → index → prefill/cache → 실제 greedy IDs 순으로 검사한다. float tolerance는0이며 max-abs는 실패 진단용이다. L 불일치가 해결되기 전 U/C로 진행하지 않는다. 실제 원본/API 검토와 계산 일정 차이는 Runtime 감사 (local artifact: `RUNTIME_CONNECTION_AUDIT.md`)에 기록했다.

## 4. 최소 Smoke 제안

기존 manifest 순서의 첫 smoke 이미지376959를 재사용한다. 이미지 JPEG·크기640×480·191,688 bytes·SHA `f781f3ff43240353be20f4e58b67a70db2226ff95a7b15bba4d04db9cf97ed0d`를 확인했으며 이미지 복사는 하지 않았다.

1. 공식 Q1153 `Is there a chair in the image?`
2. 기능검증 전용 `Describe this image in one sentence.` — **POPE 질문·accuracy 평가가 아니다.** decoder append를 관측하기 위한 outcome-blind 사전 제안이다.

두 질문이면 **10 generation + native Full prefill2회**, 한 질문이면5 generation+참조1회다. 두 질문과 모델 revision은 실제 승인 전에 다시 잠근다. Warm-up/forced continuation/probe/성능 반복/POPE200은 기본 실행에 없다.

원본 Yes/No early-stop 때문에 append가 없으면 `unexercised`다. 모든 질문을 합쳐도 해당 방법의 append0이면 gate를 중단하며, 질문이나 stop 규칙을 자동 변경하지 않는다.

## 5. 실행 전 필요한 사용자 작업

1. A10040GB급 Linux GPU의 공급자·실제 SKU·기간·시간당/저장/egress 비용·상한과 종료 방식을 결정하고 임대를 별도 승인한다. 지금 대여·접속하지 않았다.
2. 환경 구축과 private HOLD 소스/이미지 전송을 승인하거나 직접 준비한다. 제안 pin은 **Python3.11.18, Torch2.4.1+cu121, Transformers4.49.0**, 나머지는 `DIRECT_PINS_PROPOSAL.json`이다. 설치·전이 package resolve·실제 GPU 호환성은 미실행이다.
3. 공식 checkpoint의 실제40-hex revision을 결정하고 모델 취득을 승인하거나 직접 제공한다. 원본 논문 당시 revision은 미확인이다. symlink 없는 materialized snapshot과 모든 config/tokenizer/processor/shard SHA가 필요하다.
4. 질문/이미지·source·environment lock을 생성하고 CPU `--plan-only`를 통과시킨다.
5. 실제5/10 generation과1/2 native prefill, wall limit900초, 최소 free30GiB 등의 GPU 실행을 별도 승인한다.

### 첫 원격 CPU-safe 명령

환경·snapshot·private 입력을 준비한 뒤 lock을 만든다. 아래 placeholder는 실제 값으로 바꿔야 하며 현재 실행하지 않았다.

```bash
python -B examples/lock_s2_gpu_smoke.py \
  --snapshot /workspace/models/Qwen2.5-VL-7B-Instruct \
  --revision REAL_40_HEX_REVISION \
  --image /workspace/private_inputs/COCO_val2014_000000376959.jpg \
  --questions /workspace/private_inputs/smoke_questions.json \
  --local-hostname ORIGINAL_WINDOWS_DEVELOPMENT_HOSTNAME \
  --output-dir /workspace/private_locks/smoke01

python -B examples/run_s2_gpu_smoke.py --plan-only \
  --spec /workspace/private_locks/smoke01/SMOKE_SPEC.json
```

### 별도 승인 후 첫 GPU 실행 명령

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -B examples/run_s2_gpu_smoke.py \
  --spec /workspace/private_locks/smoke01/SMOKE_SPEC.json \
  --allow-gpu --authorization-id ACTUAL_EXPLICIT_USER_GPU_APPROVAL_ID \
  --output-dir /workspace/results/smoke01
```

기존 output이 있으면 중단한다. 동일 config라도 자동 resume/retry는 하지 않는다. Python의 timeout/종료는 cloud billing 종료가 아니며, 결과 반출·SHA 확인 후 provider에서 instance 종료/삭제와 비용 중지를 사용자가 확인해야 한다.

## 6. 남은 위험과 재현 한계

실제 M-RoPE·DynamicCache·checkpoint·native Full·original/L parity는 **GPU 미검증**이다. 40GB도 동작을 보장한다고 기록하지 않는다. 실제 config/grid/token/cache가 다르면 자동 patch나 dtype/backend 변경 없이 중단한다. 새 GPU-native Coverage나 다른 decoding을 도입하지 않는다.

기능검사에는 hook·CPU tensor snapshot·hash·동기화 오버헤드가 있다. 지금 기록한 CPU 시간 또는 미래 기능검사 시간을 TTFT/속도 향상으로 해석하지 않는다. 성능 측정 계획 (local artifact: `MEASUREMENT_PROTOCOL.md`)의 별도 warm-up/반복은 기능 통과 후 추가 승인 대상이다.

초기 `cpu_run_01/02`는 Python TemporaryDirectory의 Windows0700 ACL 때문에 신규 scratch setup/cleanup에서 각각21 error가 발생했다. 기존143개는 두 실행에서도 통과했다. 새 테스트 scratch만 결과 폴더의 inherited ACL로 생성하도록 바꿔 `cpu_run_03`205/205를 얻었다. 기존 환경/파일 ACL이나 알고리즘은 바꾸지 않았다. 실패 source/log는 보존한다. 초기 scratch 중 접근 불가 디렉터리는 강제 권한 변경·삭제하지 않으며, 결과 manifest의 scan exclusion으로 명시한다. 이 경로에 실제 모델/실험 결과는 생성하지 않았다.

## 7. 산출물

- 환경 재현 가이드 (local artifact: `ENVIRONMENT_GUIDE.md`)
- Runtime 연결 감사 (local artifact: `RUNTIME_CONNECTION_AUDIT.md`)
- 원본 동일성·중단 절차 (local artifact: `ORIGINAL_EQUIVALENCE_PROTOCOL.md`)
- GPU 메모리·비용 관리 (local artifact: `GPU_COST_MEMORY_PLAN.md`)
- 기능/효율 측정 분리 (local artifact: `MEASUREMENT_PROTOCOL.md`)
- 미해결 Runtime blocker (local artifact: `RUNTIME_BLOCKERS.md`)
- 실행 전 체크리스트 (local artifact: `PRE_EXECUTION_CHECKLIST.md`)
- `INPUT_PROPOSAL.json`, `SMOKE_QUESTIONS.json`, `DIRECT_PINS_PROPOSAL.json`
- `cpu_run_01/02/03`, `CLI_HELP_CHECKS.json`, 보호 파일 전후 SHA, code/result manifest

준비 단계에서 중단한다. GPU·설치·다운로드·임대·Git·Notion 작업을 자동 시작하지 않는다.
