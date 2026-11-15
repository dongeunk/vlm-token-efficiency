> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s2b/phase_s2b_20261114_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# SmolVLM Phase S2-B — 실제 Layer0 ERC Pruning Smoke 보고

2026-11-14, Asia/Seoul. **S2-B PASS. S2-L/U/C 실제 GPU Pruning을 각각2회, 총6회 완료했다.** 모델은1회 로드하고 같은 이미지·Q1153을 반복했다. 각 실행에서832→64 visual,878→110 sequence, 비시각46개 보존 및 실제 compact KV decode append를 검증했다. **POPE 정확도 채점·파일럿·본 평가는 실행하지 않았다.**

## 1. PASS / FAIL / 미검증

| 항목 | 결과 |
|---|---|
| Gate A Allocation/actual ERC/Cell | 6/6 PASS, Local48+Global16, finite ERC832개, 공통 Cell별 최대 선택 |
| Gate B sequence/RoPE/KV | 6/6 PASS, full Layer0 878→upper29층110, 실제 append10회/final KV120 |
| Gate C greedy generation | 6/6 PASS, 각11tokens/EOS, runtime error·NaN·OOM0 |
| 동일 방법 반복 결정성 | L/U/C 각각 PASS, budget/ERC/선택/hidden/logits/전체 final KV/IDs exact |
| 이전 Native dense prefix | 6/6 PASS, 저장된 S2-A CPU evidence와 Vision/Projector/Layer0 H0/H1/KV exact |
| CPU/Mock·회귀 tests | **132/132**, 신규31 + S1/S2-A101, failure/error/skip0 |
| 초기 CPU 소유권 fixture | 1error→수정·재검증 완료. 실제 GPU 실행 전 발생 |
| 추가 질문·N1088 GPU·다른 이미지 | 미검증/미실행 |
| POPE Accuracy/F1·안정적 TTFT/효율 우위 | 미검증/미실행 |

최종 CPU 실행은 `cpu_run_02/`, 실제 GPU 실행은 `gpu_run_01/`이다. 승인된6회 생성 예산을 모두 사용했고 추가 Full generation, probe, warm-up, forced continuation, 자동 retry를 수행하지 않았다.

## 2. 실제 실행 설정

| 항목 | 실제 설정 |
|---|---|
| model/revision | HuggingFaceTB/SmolVLM-256M-Instruct / `7e3e67edbbed1bf9888184d9df282b700a323964` |
| architecture | Idefics3 + Llama,30 layers/hidden576/KV3heads×64 |
| GPU/environment | 로컬 GTX1660SUPER6GB, Torch2.13.0+cu126/Transformers5.15.1, FP16/eager/batch1 |
| 입력 | 기존 image376959, Q1153 `Is there a chair in the image?` |
| 실제 layout | R3×C4,13views, Local768+Global64=832 |
| sequence | 원래878, nonvisual46, 유지visual64, suffix110 |
| budget | Local48/G16/base16+extra32, Global16/단일8×8 region의16 cells |
| C demand/candidate | 전체 Local+Global demand / Local candidates, query weight 없음 |
| decoding | greedy16, EOS49279/PAD2/BOS0, fresh cache/시도 |
| 순서 | L1→L2→U1→U2→C1→C2, 같은 모델 인스턴스 |

R/C·placeholder·pixel metadata를 실제 processor 출력으로 확인하고 S2-A의 동일 input IDs와 대조했다. 단순히832를 추정하지 않았다. S2-A Native CPU tensor를 읽어 dense prefix를 비교했으며 새 Native/Full GPU 실행은 추가하지 않았다.

## 3. 실제 Pruning 경로

`projector V→기존 S1 allocate→공통 cells 사전 준비→full Layer0→GPU native FP16 residual 후 FP32 norm→기존 S1 select→64 visual+모든 nonvisual gather→실제 Layer0 KV gather→explicit suffix mask/original1D RoPE→upper29 layers→greedy decode`다.

S1 Allocation·Cell/ERC 수식을 바꾸지 않았다. 새 Cell 준비 함수는 같은 private reference geometry를 호출하고 actual ERC를 받은 S1 selector의 cell 목록과 exact 대조한다. ERC는 실제 CUDA H0/H1 visual rows의 `(H1−H0).float()` L2 norm이며832개 전부 finite다. 모든 방법의 dense prefix와 ERC가 exact 같았고 Global16 최종 선택도 공통이었다.

원래 position IDs를 재번호화하지 않는다. suffix positions는 keep gather한 원래 값이며 decode logical position은878부터, physical cache length는110부터 증가한다. 설치 Llama에 없는 `cache_position` 인자는 주입하지 않았다. 모든30층 final physical/public length120=110+11−1, final KV[1,3,120,64]를 확인했다.

## 4. 예산·Feature Coverage·출력

| 방법 | Local 지역 예산16개(row-major) | 최종 F_all(S) |
|---|---|---:|
| S2-L | 1,4,3,2,5,1,2,2,3,5,4,2,1,5,6,2 | 104.641022 |
| S2-U | 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 | 104.075966 |
| S2-C | 2,3,2,4,4,4,1,2,4,4,3,4,2,2,3,4 | 105.054344 |

각 local budget의 합은48이며 Global16과 함께 최종64개다. C의 temporary32는 budget histogram 전용이며 final ERC64와 구분한다.

각 방법의 두 실행은 선택 집합 및 F(S)도 같았다. F_all은 전체 original V의 FP32 cosine/full-row mean/clipping으로 만든 공통 affinity에서 최종 ERC64개를 평가한 값이다. 한 이미지의 representation 진단이며 Accuracy나 C의 일반적 우위가 아니다.

세 방법 모두 raw answer는 ` Yes, there is a chair in the image.`였고11개 IDs가 같았다. **다른 답변이 나와도 방법 간 차이만으로 실패하지 않는 사전 규칙**을 유지했다. 이번에는 우연히 같았으며 GT 비교/semantic parser를 실행하지 않았다.

최종 집합 intersection은 L–U53, L–C52, U–C57개다. Global16은 전부 공통, 나머지는 local 차이다. 원본 visual ordinals·원래 sequence positions·ERC832개·temporary32 trace는 각 `*_RESULT.json`과 raw CPU tensor evidence에 저장했다.

## 5. GPU 메모리와 시간

전체 모델 로드1회(0.6290초), 실제 invocation62.9335초. 모든6회 peak allocated1959.524MiB/reserved2010MiB, OOM0이다.

| 방법 | 반복1 generation+관측 | 반복2 generation+관측 | 반복1/2 allocation |
|---|---:|---:|---:|
| L | 7.3024초 | 7.6537초 | 4.550/4.869ms |
| U | 7.0171초 | 7.8765초 | 0.723/0.933ms |
| C | 7.3666초 | 7.8985초 | 45.182/44.520ms |

Layer0/selection/gather/prefill 등 상세 동기화 구간은 진단 표 (local artifact: `GPU_MEMORY_LATENCY.md`)에 있다. 초기 L1의 ERC handoff/gather에는 큰 first-use 차이가 관측됐다. debug hooks·CPU copies·동기화·운영체제 상태가 포함된 기능 Smoke이며 안정적인 TTFT/속도향상 결과가 아니다. peak가 S2-A와 비슷하다는 사실도 전체 속도/메모리 효율 결론으로 확장하지 않는다.

## 6. 보존과 다음 단계

기존45개 bounded size/SHA 대조에서 변경0이며 checkpoint 파일 size/mtime도 unchanged다. S0/S1/S2-A 코드·결과와 기존 parser/runner/vendor를 수정하지 않았다. 기존 연구 전체 SHA·prediction logs 분석은 반복하지 않았다. S²Prune HOLD 유지, 다운로드·설치·유료서비스·외부GPU·Git commit/push0이다.

**S3 파일럿 준비로 진입할 기능적 기반은 PASS다.** 다만 현재 runtime은 검증된 N832/L878/Q1153 Smoke에 잠겨 있다. 다른 N/T와 신규 코호트를 지원하는 범위 검증·manifest lock은 S3 진입 조건 (local artifact: `S3_ENTRY_CONDITIONS.md`)에 남겼다. 이번에 S3 데이터 선정·추론·채점은 시작하지 않았다.

**S2-B 완료 후 중단했다.**

## 산출물

- Runtime 및 사용법 (local artifact: `RUNTIME_AND_USAGE.md`)
- [Allocation·선택·Coverage 진단](S2B/ALLOCATION_AND_SELECTION.md)
- [RoPE/KV/append 검증](S2B/ROPE_KV_APPEND_AUDIT.md)
- GPU memory/latency (local artifact: `GPU_MEMORY_LATENCY.md`)
- CPU 회귀 결과 (local artifact: `CPU_REGRESSION.md`)
- S3 진입 조건 (local artifact: `S3_ENTRY_CONDITIONS.md`)
- 기존 파일 보존·한계 (local artifact: `INTEGRITY_AND_LIMITS.md`)
- `gpu_run_01/`: 실제6회 결과·trace·raw CPU evidence·실행source·token overlap
- `cpu_run_01/02/`: 초기 fixture error 및 최종132/132 기록·실행source
