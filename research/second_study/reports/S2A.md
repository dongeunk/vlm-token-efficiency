> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s2a/phase_s2a_20261114_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# SmolVLM Phase S2-A — 실제 GPU 동일성 검증 보고

2026-11-14, Asia/Seoul. **S2-A PASS. Native Full / Custom Full / No-op의 실제 GPU 생성3회를 완료했다.** 세 경로의 token IDs, Vision/Projector, Layer0 hidden, 모든 decoder layer 관측값, mask/RoPE, 첫 logits 및 최종 전체 KV가 허용 오차0으로 일치했다. S2-L/U/C 실제 Pruning과 POPE 평가는 실행하지 않았다.

## 1. PASS / FAIL / 미검증

| 항목 | 판정 | 근거 |
|---|---|---|
| Gate A Native Full | **PASS** | 정상 EOS 생성11tokens, 실제 memory/shape/cache 관측 |
| Gate B Custom Full | **PASS** | Native exact 일치, 첫 logits/Layer0 최대 절대 오차0 |
| Gate C No-op | **PASS** | Layer0 뒤 identity gather를 실제 통과, visual832개 전부 보존, Native exact 일치 |
| CPU fake/계약 tests | **PASS** | 최종33/33, failure/error/skip0, 모델/CUDA 실행0 |
| 준비 invocation01/02 | **FAIL → 원인 해결** | NUL 쓰기 guard 및 TF5 unset config 처리 오류. generation0회에서 중단, 기록 보존 |
| 실제 K64 삭제·S2-L/U/C | **미검증 / 미실행** | 이번 승인 범위에서 금지 |
| POPE Accuracy/F1·성능 비교 | **미검증 / 미실행** | parser/scorer 미호출, 반복 timing 미실행 |
| N1088·다른 이미지/질문/batch/backend | **미검증** | 실제 Smoke는 이미지1장·질문1개 |

최종 실제 실행은 `gpu_run_03/`, CPU 계약 실행은 `cpu_run_04/`다. generation은 전체 invocation을 합쳐 **정확히3회**이며 추가 probe/warm-up/forward-only 실험을 하지 않았다.

## 2. 실제 입력과 환경

| 항목 | 실제 확인 |
|---|---|
| model/revision | SmolVLM-256M-Instruct / `7e3e67edbbed1bf9888184d9df282b700a323964` |
| architecture | Idefics3 + Llama30layers, hidden576, KV3heads×64 |
| GPU | 로컬 NVIDIA GeForce GTX1660SUPER, NVIDIA-SMI total6144MiB, driver621.24 |
| environment | Python3.11.15 / Torch2.13.0+cu126 / Transformers5.15.1 / torchvision0.28.0+cu126 |
| precision/backend | FP16/eager, batch1, fresh cache/route |
| image | 기존 `COCO_val2014_000000376959.jpg`, SHA f781f3ff43240353be20f4e58b67a70db2226ff95a7b15bba4d04db9cf97ed0d |
| question | Q1153: `Is there a chair in the image?` |
| processor | Idefics3Processor / Idefics3ImageProcessor |
| actual local/global | processor R3×C4, Local768 + Global64 = **832 visual tokens** |
| original sequence | **878**, nonvisual46; 세 경로에서 전부 유지 |
| vision output | [13,1024,768] |
| projector output | [13,64,576] |
| Layer0 input/output | [1,878,576] / [1,878,576] |
| decode | greedy, max16, EOS49279/PAD2/BOS0 |

R/C는 실제 processor metadata와 pixels를 대조해 얻었고 N은 실제 placeholders 및 projector output으로 확인했다. 이전832 기록을 실행 조건으로 추정하지 않았다. 상세 native ordinal/visual/nonvisual positions 및 nominal 좌표는 `gpu_run_03/ACTUAL_INPUT.json`에 있다.

## 3. 생성·수치 결과

세 경로의 raw answer는 ` Yes, there is a chair in the image.`이며 실제 IDs는 다음과 같다. 이것은 정답률 평가가 아니다.

```text
[9230,28,665,314,253,9192,281,260,2443,30,49279]
```

Custom/No-op 모두 Vision, Projector, Layer0 input/output, first FP16 head logits[1,49280]의 **max_abs=0**이다. 모든 decode step의 layer hidden/mask/actual cos·sin/KV lengths·tails/logits 및 final30층 전체 K/V도 exact 비교를 통과했다. 첫 FP32 greedy-decision logits도 일치했다. 최대 오차 위치와 값은 각 `*_RESULT.json`의 `numeric_differences`에 보존했다. 오차0일 때 argmax가 반환한[0,0,…]은 불일치 위치가 아니다.

Actual cache는 `transformers.cache_utils.DynamicCache`/DynamicLayer다. Prefill878 이후10개 token이 실제 append되어 모든30층 final physical/public length가888, KV shape는[1,3,888,64]다. 마지막으로 생성한 EOS는 forward되지 않아878+11−1=888이다. 원래1D positions는 prefill0..877, decode878..887이며 설치 Llama API에 없는 `cache_position`을 주입하지 않았다.

## 4. GPU 메모리·시간

| 경로 | generation+관측 시간 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|
| Native Full | 8.0102초 | 1959.534 | 2034 |
| Custom Full | 6.8856초 | 1959.518 | 2032 |
| No-op | 7.0851초 | 1959.518 | 2034 |

최종 invocation은 모델을1회 로드하고 세 경로를 순차 실행했다. load0.6413초, 전체35.2367초다. **준비 실패 invocation02에서의 load1회를 포함하면 전체 작업의 모델 로드는 총2회**다. 병렬 모델 인스턴스는 없었다. preparation failure로 reload한 이유를 오류 기록 (local artifact: `ERRORS_AND_LIMITS.md`)에 명시했다.

OOM0. 위 시간은 CPU snapshot·hooks·동기화·초기 kernel 상태가 포함된 단회 기능 관측이며 TTFT/속도향상 비교가 아니다. 경로 간 짧은 시간 차이를 성능 이득으로 해석하지 않는다.

최종 프로세스 정리 시 allocated8.19MiB/reserved20MiB가 남아 있었으며 이를0으로 보고하지 않는다. 프로세스가 exit0으로 종료된 뒤 NVIDIA-SMI는 used445/free5522MiB, 시작 전 used447/free5520MiB로 관측됐다. 실행 전후 실제 장치 상태와 per-stage bytes는 별도 JSON에 보존했다.

## 5. 보호·격리

기존 S0/S1·관련 dependencies/parser/runner·주요 기존 보고서 등37개 bounded SHA 전후 대조에서 변경0. checkpoint 파일들의 size/mtime도 unchanged였고 가중치 전체 SHA를 다시 계산하지 않았다. 프로젝트/기존 prediction logs 전체 재검증은 반복하지 않았다.

신규 코드는 `src/smol_s2_runtime/`와 신규 CPU/GPU examples/tests뿐이다. 설치 패키지·checkpoint·S1·기존 평가 코드를 수정하지 않았다. 로컬 offline snapshot만 사용했고 다운로드/외부GPU/유료서비스/Git commit·push0이다. S²Prune HOLD 유지.

## 6. S2-B 진입 판정과 중단

**S2-B의 실제 pruning runtime 개발로 진입할 기반 조건은 충족했다.** 단, 실제 토큰 삭제, short KV, gappy1D positions 및 S2-L/U/C 성능은 아직 미검증이다. 이번 결과는 한 입력의 K=N parity에 한정된다.

추가 GPU generation 또는 S2-L/U/C 실행은 시작하지 않는다. **S2-A 완료 후 중단했다.**

## 산출물

- Runtime 코드·사용법 (local artifact: `RUNTIME_USAGE.md`)
- 경로별 비교 결과 (local artifact: `COMPARISON_RESULTS.md`)
- [RoPE/KV/append 감사](S2A/ROPE_KV_APPEND_AUDIT.md)
- 오류·해결·남은 한계 (local artifact: `ERRORS_AND_LIMITS.md`)
- S2-B 진입 판정 (local artifact: `S2B_ENTRY_DECISION.md`)
- 기존 파일 무결성 (local artifact: `INTEGRITY_VERIFICATION.md`)
- `gpu_run_03/`: source snapshot, input mapping, JSON trace/results 및 실제 CPU tensor evidence3개
- `cpu_run_04/`: CPU33/33 원문/metadata/source snapshot
- `gpu_run_01/02/`: 생성 전 준비 실패 원문과 보존 검사
