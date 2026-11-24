# Study5-P1 구현과 감사 결과

**CPU 148/148 PASS.** 기존 Study4 공개 회귀 139개와 신규 A/D/G/H/I 테스트 9개를 포함한다. 가중치가 필요한 GPU Smoke와 실제 성능 평가는 수행하지 않았다. [사전 등록](STUDY5_PROTOCOL.md) · [공개판 검증 범위](../README.md#검증과-재현)

## 구현과 회귀

- `src/study5_selection.py`에서 Study4의 준비된 공통 문제를 받아 A/D는 기존 선택 함수를 호출한다. G는 동일 raw FP32 점수의 전역 Top-25를 seed로 기존 `global_greedy`를 호출한다. H는 전역 Top-64를 선택한다. I는 D의 실제 seed tuple 뒤에 미선택 후보의 점수순 39개를 붙인다. 동률은 native ordinal 오름차순이다.
- D와 G는 같은 점수 벡터, 가중치, 유틸리티 및 원래 시퀀스 매핑을 공유한다. 동일 seed-aware Coverage 경로를 사용하며, G는 격자 셀을 고르는 과정에서 사용하지 않는다. G의 25개가 한 셀에 몰릴 수 있음을 합성 사례로 테스트했다.
- N576/832/1088에서 A/D의 기존 seed·선택 순서·원본 시퀀스 위치·Coverage trace가 정확히 일치했다. D/G의 seed 초기값과 매 단계 39개 선택을 느린 CPU 계산과 대조했다. H의 Top-64와 G/H의 Top-25 일치, D/I seed 공유, H/I에서 greedy 미호출, unique K64, 비시각 토큰 보존을 대조했다. 세 구조의 정규 매핑과 N832 4×3 전치도 검사했다.
- Study4의 핵심 구현은 복사 후 원본 SHA-256을 다시 대조했다. 복사본 중 `coverpruner_spatial_layout.py`의 기본 fixture 경로만 Study5 내부로 변경했다. A/D 회귀는 합성 CPU 결과다. [공개 소스](../src/)

## RESERVE79 감사

- 공식 POPE Random **500장·3,000문항**과 Dev Manifest의 **기존 노출 321장 + Dev100 + 예비 79장**이 정확히 일치했다. 예비 79장과 기존 421장의 이미지 ID 교집합은 0이다. 예비의 공식 문항 ID·문장·라벨, 이미지 SHA-256·크기를 재확인했다. 474문항이고 이미지당 Yes/No가 각 3개이며, 전체는 **237 Yes / 237 No**다.
- 기존 감사의 원본 기록 **51개** 해시를 다시 확인했다. COCO 421장, 예제 이미지 5개와 P4 QA에서 확인된 이미지 ID 한 개에 대응하는 기록을 대조했다. 후보와 이 대조 자료 사이 및 79장 내부에 파일 SHA·EXIF 방향 보정 RGB 픽셀 일치 또는 128비트 dHash 거리 ≤10 의심 사례가 없었다. 검토할 의심 쌍이 없어 별도 제외는 0장이다. dHash는 변형이나 크롭을 모두 검출할 수 없다.
- 로컬 CPU Processor에서 79장 모두 호환됐다. N832 3×4 **59장**, N832 4×3 **13장**, N1088 4×4 **7장**이다. 이미지별 첫 공식 질문에 대해 확인한 입력 검사이며 성능 평가는 수행하지 않았다.
- **기록된 범위에서 가용 후보는 79장이다. 독립 Holdout은 미확정이다.** 보존되지 않은 입력 이력 때문에 79장 모두의 완전한 미노출을 판정할 수 없다. Study4 Test400 Manifest 및 READY는 존재하지 않았다. 기존 QA 저장소는 읽기 전용으로 조회했고 Test400 상태는 그대로 유지했다. [공개용 집계 출처](../results/PROVENANCE.json)

이미지는 Study4 `runs/p3b0-dev-20261124/RESERVE79.json`이 가리키는 아카이브 내 COCO val2014 이미지이며, 공식 문항·라벨은 `archive/study1-2/ai_project/data/pope/coco_pope_random.json`을 읽는다. POPE 코드의 MIT 고지, COCO 주석의 CC BY 4.0, 원본 이미지에 적용되는 개별 권리는 구분한다. 이번 단계의 데이터 읽기는 로컬 검증에 한했다. [공식 POPE](https://github.com/RUCAIBox/POPE) · [COCO 이용 조건](https://cocodataset.org/dataset/termsofuse.htm)

## 다음 GPU Smoke 명령

별도 승인 후 아래 명령을 실행한다. 기존 비평가 N576 이미지에서 다섯 방법을 각 한 번, 최대 5회만 생성한다. 이 단계에서 확인한 것은 `--run-gpu`를 뺀 경로·해시 검사다. 실제 GPU 연결은 미검증 상태다.

```powershell
cd study5
python -I -B -X utf8 scripts/smoke_gpu.py --snapshot "<local-snapshot>" --image "<allowlisted-image>" --run-gpu
```

주 비교 **D−G Strict Accuracy**, 추가 대비 D−H/D−I/G−H/I−H 및 분석 정책은 사전 등록 문서에 고정했다. 독립 자격 증빙과 실제 GPU 검증이 남아 있으며 Left Padding FP16 parity는 기존 BLOCKER다. Study1–4와 GitHub에 변경을 가하지 않았다.
