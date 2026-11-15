# 2차 연구 — S²Prune-inspired SmolVLM adaptation

S0–S4-A의 연구 기록과 소형 집계 결과다. 원본 Qwen S²Prune/CoverPruner 전체 재현으로 표기하지 않는다. 기존 1차 공개 자료를 대체하지 않는다.

|방법|Semantic Accuracy|Semantic F1|Invalid/1200|
|---|---:|---:|---:|
|Full|83.333%|85.481%|78|
|S2-L|80.917%|83.470%|54|
|S2-U|80.583%|83.226%|55|
|S2-C|80.250%|82.904%|57|

주 비교 C−L F1은 −0.566%p, image-paired95%CI [−1.575,+0.432]%p다. Coverage 배분의 우위는 관측되지 않았다. 고정 POPE Random 부분집합 결과이며 공식 전체 benchmark 점수가 아니다.

- reports/: S0 설계, S1 CPU, S2-A/B 기능 검증, S3-A 고정, S3-B1/B2, S3-C, S4-A 보고서.
- config/: 경로 비식별 설정 및 ID/digest-only 코호트 view. 원본 잠금 파일의 byte-identical 대체물이 아니다.
- main/: 이미지별6문항 confusion count vectors, 지표/CI와 예산·Coverage·시간 집계. 원시 응답·질문 전문은 제외.
- figures/: 원본 사진이 없는 집계 그림2개.

## CPU에서 제공하는 재현

루트에서 `python -B tools/reproduce_summary_cpu.py`를 실행하면200개 이미지 count로 Accuracy/F1/Invalid/Strict 집계를 검산한다. `--bootstrap`은 선택적으로 기존 image-paired10000회·seed42 CI를 다시 계산한다. 이 정리 단계에서는 기본 집계 검산만 실행하고 대규모 재검증/Generation을 하지 않는다.

PyTorch/Transformers 등 모델 import와 private S²Prune helper import를 차단한다. 원시 응답을 배포하지 않으므로 Parser 재판정은 이번 공개 범위에 포함하지 않는다. NumPy 버전은 requirements-summary.txt에 기록했다.

## 제공하지 않는 실행 범위

권리/의존성 검토가 남은 src/s2_coverage_study의 helper/allocator, Smol Cell/ERC/Pruning runtime과 GPU runner는 로컬에 보존하고 이번에 배포하지 않는다. 기존 upstream helper를 stub이나 자동 다운로드로 대체하지 않았다. 전체 GPU/과거 CPU suite 실행은 공개본만으로 제공하지 않는다.

S4-A에서 C/L 예산은200장 모두 달랐으나 Coverage 임시32의 최종 보존은평균2.625개(8.20%), 정답 변화는11개 개선·19개 악화였다. 목적 전달과 객체 판별 정렬은 검증 가설이며 새 Ablation을 실행하지 않았다.

현재 공개 정책은 [release scope](../../docs/SECOND_STUDY_RELEASE_SCOPE.md), 전체 후보는 [FILE_LIST](../../docs/publication/FILE_LIST.md)를 따른다. 원본 본문/캐시/사진/권리 HOLD 상태를 바꾸지 않는다.
