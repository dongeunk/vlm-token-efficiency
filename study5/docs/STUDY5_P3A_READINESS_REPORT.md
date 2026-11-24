# Study5-P3A 평가 Runner 준비 보고서

**CPU Gate PASS. P3-B 후보 평가 Runner READY.** 이 상태는 탐색적 후보 평가 실행 준비를 뜻한다. 독립 Holdout 확정은 BLOCKED다. P3-A의 실제 GPU 모델 로드·Generation·평가 Job은 각각 0회다.

## 고정 데이터와 실행 설정

- Study4 `RESERVE79.json`을 검증된 참조로 사용한다. 이미지 79장·공식 질문 474문항·Yes 237개·No 237개를 원본 POPE와 다시 대조했다. 이미지 ID·파일 경로·파일 SHA-256을 확인했고, 기록된 기존 노출 및 Dev100과 ID 중복은 없었다. CPU에서 실제 로컬 Processor로 474개 질문을 모두 처리했다. N832 432문항, N1088 42문항이며 모두 padding row가 없었다.
- 고정 순서로 이미지/질문/방법 고유 ID를 만든다. A/D/G/H/I 각각 474개, 합계 2,370개를 조성한다. 5장씩 16개 Chunk다. P1의 Prompt, tokenizer 경로, 모델 snapshot, K64, Batch1 No-Padding, SDPA FP16, Greedy, 최대 10토큰, Strict 및 Official-compatible Parser를 적용한다. Study4와 Study5의 선택 코드 및 기존 결과는 변경하지 않았다.
- Manifest, 실행 설정, Job 계획, snapshot 식별자·메타데이터 해시, 모델 파일 크기, 코드 해시는 실행 전 SQLite에 저장한다. 재개 시 모든 항목이 일치해야 한다. [공개용 출처 해시](../results/PROVENANCE.json)
- 노출 기록 밖의 이력이 남아 있다. 현재 계획은 **Holdout79 후보 탐색 평가**로 고정하며 독립 Test 결과로 보고할 수 없다. 별도의 시각 QA 단계는 추가하지 않았다. [변경하지 않은 사전 등록](STUDY5_PROTOCOL.md)

## 재개와 무결성

Study4의 생성 전 예약과 불확실한 Job을 반복하지 않는 계약을 재사용했다. Study5는 단일 SQLite에 Job별 예약과 결과를 별도 트랜잭션으로 저장한다. 마지막 Job을 저장할 때 Chunk 해시도 원자적으로 기록한다. 실제 Runner는 FULL 동기화를 사용한다. 동일 디렉터리를 재실행하면 완료된 결과는 그대로 사용하고 미시작 Job만 실행한다. 생성 전 실패는 로그를 남기고 같은 Job을 재시도할 수 있다. 생성 예약 후 실패 또는 중단으로 성공 여부가 불확실하거나, 완료된 결과가 유실된 경우에는 원본 저널을 보존하고 자동 재생성 없이 중단한다.

무결성 확인에는 DB, Job ID 재고, Chunk 구성·해시, 결과 해시, 각 방법의 474개 결과, 출력 K64·실행 프로파일·문항별 공통 Query 해시가 포함된다. 위 조건을 모두 통과한 뒤 COMPLETE를 기록하고, 전체 결과에서만 집계 파일을 발행한다. 기존 결과를 덮어쓰지 않는다.

## CPU 검증과 분석

- **154/154 PASS**. 기존 148개와 신규 6개다. 중단 지점 317/2,370에서 재개 순서를 섞고, 미시작 2,053개만 실행한 결과가 중단 없이 만든 전체 Mock 결과와 완전히 일치했다. 작업별 호출은 한 번이었다. 설정·Manifest 변경, 중복 Job, Chunk 손상, 완료 결과 유실, 미완료 COMPLETE, 생성 예약 후의 재시도를 차단했다.
- 대량 CPU Mock에서는 디스크 동기화만 껐고 동일 SQLite 원자적 트랜잭션·재개 로직을 사용했다. 작은 디스크 복구·손상 테스트와 실제 Runner에서는 FULL을 사용했다. Windows에서 Mock DB 연결이 정리 전에 닫히도록 수정한 뒤 전체 스위트를 다시 통과했다. 별도 합성 한 쌍의 비영 효과에서 paired 전환과 bootstrap 구간을 확인했다. Mock 성능 수치와 응답은 보존하지 않았다.
- 주 비교 D−G Strict Accuracy, 보조 D−H·D−I·G−H·I−H·D−A·G−A를 구현했다. 방법별 Accuracy·F1·Precision·Yes/No Recall·혼동행렬, paired 정답 전환, Cell Occupancy, Coverage를 산출한다. 통계는 동일 이미지 단위의 paired bootstrap 10,000회, seed 20261126, 선형 보간 95% 구간이다. 보조 비교는 다중 비교 보정 없이 탐색적으로 해석한다.

## P3-B 실행과 재개

고정 실행과 재개는 원래 비공개 Manifest, 감사 파일과 P3-A readiness 파일을 사용하는 로컬 환경에 한정된다. 완료된 응답은 같은 실행 디렉터리로 재개하면 재사용된다. 공개 CPU 테스트의 2,370개 작업은 합성 fixture다. [공개판 실행 조건](../README.md#검증과-재현) · [GPU 실행 진입점](../scripts/run_p3b_candidates.py) · [재개 테스트](../tests/test_p3a_resume_cpu.py)

미해결 BLOCKER는 Holdout79의 독립성 증거와 기존 Left-Padding FP16 parity다. P2의 No-Padding GPU Smoke 15/15 이력은 보존했다. P3-A에서는 새 GPU 연동을 실행하지 않았다.
