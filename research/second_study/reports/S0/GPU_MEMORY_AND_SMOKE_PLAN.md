> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01/GPU_MEMORY_AND_SMOKE_PLAN.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S0 — GTX 1660 SUPER 메모리 분석과 Smoke 계획

작성: 2026-11-14. **GPU 조회·CUDA 초기화·모델 로드 없이 기존 기록과 tensor shape로 분석했다.** 아래 신규 경로의 메모리는 추정이며 실측이 아니다.

## 1. 실행 가능성 판단

**Full SmolVLM FP16/eager는 이 GPU에서 이미 실행된 기록이 있다. Layer0 pruning도 6GB에서 시도할 타당성이 높다.** 새 경로의 성공은 실제 Smoke에서 확인해야 한다. ERC/affinity/KV보다 vision eager attention의 큰 일시 tensor와 allocator reserve가 주요 peak 원인이다.

| 기존 기록 | 실제 보고 |
|---|---|
| `results/pope_pilot/full_model_report.md` | GTX1660SUPER, Full30/30, N832×26/N1088×4, OOM0, peak allocated2402.02MiB |
| `results/pope_spatial_floor_80_evaluation/full_v2_20261024_01/REPORT.md` | 80이미지160문항×5방법, OOM0, allocated2458.05MiB/reserved3230MiB |

두 번째 기록의 pruning은 **Layer0 이전 삭제**다. 새 dense Layer0 경로의 peak 실측으로 바꿔 쓰지 않는다. 모델 카드의 저메모리 안내도 실제2048-longest-edge·13/17views·eager 경로의 보장이 아니다.

## 2. shape 기반 비용

FP16은2 bytes, FP32는4 bytes, MiB=2²⁰ bytes다. T는 실제 nonvisual 수이며 실행 때 기록한다. 아래 비교에서만 설명용 T=112를 가정했다. **실측 prompt 길이가 아니다.**

| 항목 | 산식 | N832, L944 | N1088, L1200 |
|---|---|---:|---:|
| full30층 KV | 2(K/V)×30×3×64×L×2 | 20.74MiB | 26.37MiB |
| K64 이후30층 KV | 같은 산식, L'=T+64=176 | 3.87MiB | 3.87MiB |
| ERC visual H0+H1 FP16 보관 | 2×N×576×2 | 1.83MiB | 2.39MiB |
| ERC delta FP32 한 개 | N×576×4 | 1.83MiB | 2.39MiB |
| local affinity FP32 한 개 | 4(N−64)² | 2.25MiB | 4.00MiB |
| all-N 평가 affinity FP32 한 개 | 4N² | 2.64MiB | 4.52MiB |
| Layer0 attention FP16 한 개 | 9×L²×2 | 15.30MiB | 24.72MiB |
| vision attention FP16 한 개 | V×12×1024²×2 | 312MiB | 408MiB |
| view pixels FP16 | V×3×512²×2 | 19.5MiB | 25.5MiB |
| 전체 prompt logits FP16 한 개 | L×49280×2 | 88.73MiB | 112.79MiB |

Vision attention이 FP32로 올라가는 buffer 하나는 위 값의2배다. score/probability/reduction 임시 tensor가 동시에 존재할 수 있고, framework workspace/context/reserve도 있다. 표를 합해서 정확한 peak라고 제시하지 않는다. MLP/hidden도 있으나 inference에서30층 전체 activation을 동시에 보관할 필요는 없다.

로컬 `model.safetensors` 파일은513,028,808 bytes(약489MiB)다. header와 저장 dtype이 포함된 파일 크기이며 GPU resident weights나 총 VRAM과 같지 않다. 256M급 FP16 weights의 대략적인 규모를 보여주는 보조 정보다. 가중치 bytes를 읽어 parameter를 다시 집계하지 않았다.

Layer0 full KV도 gather하므로 prefill 종료 후에는 전 layer가 compact하다. append token 하나당 KV 증가량은23040bytes(22.5KiB)다. 첫 layer를 full 길이로 남기는 mixed-cache 방식은 메모리보다 정합성 문제로 제외한다.

## 3. 기본 CPU/GPU 분리

- GPU FP16/eager: vision/projector, text embeddings, dense layer0, native residual subtraction, suffix decoder, KV, greedy decode.
- CPU FP32: projector V의 명시적 handoff 후 calibrated Coverage,32-step quota selection, Laplacian/Uniform, geometry, ERC vector를 받은 최종 Cell 선택, 기록용 F(S).
- GPU→CPU: allocation용 local projector V와 ERC[N]을 복사한다. 공통 F_all 평가를 기록할 때는 global64를 포함한 all-N projector V도 CPU에 제공한다. all-N을 한 번 handoff한 뒤 local slice를 쓰는 방식도 가능하다. CPU→GPU는 final selected ordinals/keep positions다. 전체30-layer hidden·KV를 CPU로 복사하지 않는다.
- 복사·동기화·CPU selection 시간은 Selection Overhead 및 E2E TTFT에 포함한다. CPU에서 계산하므로 GPU 비용0이라고 보고하지 않는다.
- 세 방법 모두 CPU allocation backend를 고정한다. 이전 GPU FP32 selector와 near-tie 결과가 같다고 가정하지 않는다. CPU 설계가 이 연구의 명시적 실행 변형이다.

V1024×576 FP16 복사량은약1.125MiB, CPU FP32 보관은약2.25MiB다. affinity 한 개는4MiB지만 similarity·centered·gain 임시본이 함께 살아 있을 수 있다. CPU에서도 single matrix 비용을 전체 RSS라고 쓰지 않는다.

## 4. 메모리 감소 우선순위

1. batch1, inference/eval, grad 비활성, output_attentions/전체 output_hidden_states 비활성. 필요한 layer0 visual states만 관측한다.
2. methods 순차 실행 및 fresh cache. methods별 hidden/KV/GPU feature 복제본을 쌓지 않는다. hook은 요약만 보관하고 큰 payload는 즉시 해제한다.
3. affinity와 diagnostics는 CPU에 두고 GPU inference와 큰 진단 tensor의 생존 시간을 겹치지 않는다. 기능 감사와 성능 측정 run을 분리한다.
4. custom path의 lm_head는 마지막 hidden position만 처리한다. native reference는 지원되는 인자로 마지막 logits만 구할 수 있는지 설치 signature를 먼저 확인한다.
5. 그래도 OOM이면 종료 후 원인을 기록한다. 다음 protocol revision에서 view chunking vision extraction이나 chunked CPU Coverage를 검토한다. chunking은 batch/reduction 순서·near-tie 결과가 바뀔 수 있어 parity 재검증이 필요하다.

자동으로 resize/splitting/K/FP16/backend를 바꾸거나 일부 views를 버리지 않는다. 전체 코호트를 같은 정책으로 다시 잠그지 않고 메모리 fallback 결과를 같은 실험에 섞지 않는다. 새 quantization/offload 패키지 설치는 초기 범위에 없다.

## 5. 최초 GPU Smoke 최소 단위 — 향후 실행 제안

**현재 실행하지 않는다.** 다음 CPU 구현과 계약 검증이 통과한 뒤 사용자의 후속 작업 지시 범위에서 실행할 최소 단위다.

- 기존 검증 이미지 `COCO_val2014_000000376959.jpg`: N832, local3×4+global.
- 고정 질문2개: 공식 Q1153 `Is there a chair in the image?`, 기능전용 `Describe this image in one sentence.`.
- 두 번째 질문은 decode append 관측용이며 POPE Accuracy에 넣지 않는다. 질문/stop 규칙을 실제 응답을 보고 바꾸지 않는다.
- checkpoint revision/전처리/FP16/eager/batch1 고정, greedy 최대16, EOS49279/PAD2/BOS0, 매 호출 fresh cache.
- 순서: **native Full → custom Full → K=N no-op → S2-L → S2-U → S2-C**.
- 질문2개×6경로=**최대12 generation 호출, 최대192 generated tokens**. native generation prefill hook으로 reference를 관측하며 별도 warm-up/probe를 몰래 추가하지 않는다.

첫 image의 Full/custom/no-op에서 first logits·greedy IDs·EOS/길이가 맞지 않으면 L/U/C로 넘어가지 않는다. L의 geometry/ERC/gather/cache가 실패하면 U/C로 넘어가지 않는다. 모든 방법이 Full과 같은 답을 내야 한다는 gate는 아니다.

기능 비교에서는 native Full에서 관측한 projected V와 merged H0를 custom 경로에 재사용할 수 있다. 동일 입력 tensor를 비교한다는 provenance를 남기며 질문/방법별 cache는 새로 만든다. 이 feature 공유 시간은 독립 E2E 성능 값으로 사용하지 않는다.

각 pruning 방법에서 full Layer0 length L, Layer1 length T+64, selected local48/global16, all30-layer KV length T+64, 실제 append와 original logical RoPE 증가를 검사한다. 두 질문 모두 append가 없으면 decode는 `unexercised`이며 추가 질문/forced continuation 없이 중단한다.

**pilot 진입 전 두 번째 layout 검사:** 기존 image436127(N1088,4×4+global)의 고정 공식 Q1819로 위6경로의 first-token forward-only 검사6회를 수행한다. 이는 generation 성공이나 append 검사를 대신하지 않는다. 모든 runtime 구조가 첫 image와 같은지 확인하고, 다른 cache/position 경로가 발견되면 별도 decode smoke를 설계한다. 기존 smoke manifest의 정확한 원문 질문을 사용한다.

## 6. 실행 전·중단·측정 gate

향후 preflight는 모델/processor snapshot·실제 image backend·전체 설정·입력/출력 경로를 고정한다. GPU free memory 확인은 그 GPU 단계에서 처음 수행한다. 운영상 제안은 load 전 free≥4GiB, peak reserved<5GiB 목표, 최초 단위 wall limit600초다. **이는 성공 보장이 아니라 여유/중단 기준 제안**이며 실제 Windows display VRAM 점유에 따라 후속 실행 전에 확정한다.

OOM, nonfinite, view/placeholder mismatch, native/no-op mismatch, KV/mask/position 불일치, wall limit 초과에서 중단한다. 자동 retry/resume, 설정 fallback, stale cache 재사용은 없다. 기존 output이 있으면 새 run ID를 사용한다.

기능 Smoke에서는 runtime call 수·각 layer length·KV shape·position/mask·first logits/IDs·ERC/선택 좌표·allocated/reserved peak를 보존한다. timing은 debug overhead 포함 기능 기록이다. **TTFT/속도 우위는 별도 반복 측정 전 주장하지 않는다.**

정식 timing은 preprocessing/vision 포함 E2E TTFT, decoder-only TTFT, allocation/Cell/ERC/transfer 및 decoding을 나눈다. CUDA synchronize로 경계를 맞추고 load time과 steady inference를 구분한다. method별 max_memory_allocated와 max_memory_reserved, reset 시점의 baseline, 공유 feature cache 여부를 함께 기록한다.

이 계획은 외부 GPU 임대나 Qwen 실행을 요구하지 않는다. Qwen 원본의 GPU 검증 보류 상태는 그대로 유지한다.
