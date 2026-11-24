# Study4 공개 범위와 검증 근거

## Method and layout

이 릴리스의 Coverage, 공간 선택, Query Probe, Vision 관측, Cold-KV 및 세션 체크포인트는 Study4 자체 코드에서 발췌했다. 선택 수식과 알고리즘은 [최종 보고서](STUDY4_FINAL_RESEARCH_REPORT.md)에 있다. A의 full-prefix Softmax, 원본 위치의 RoPE·마스크, 압축 후 compact Position은 SmolVLM 이식 정책이다. 논문 구현을 통째로 가져오거나 모델별 동일성을 전제하지 않았다. [CoverPruner arXiv v1](https://arxiv.org/abs/2609.03158v1)

[공개 매핑 fixture](../configs/layout_geometry.json)는 N576·N832·N1088에서 확인한 블록·연결·셀 점유율과 원본 폭/높이만 골라낸 P0 메타데이터다. 이미지 내용이나 위치를 포함하지 않는다. 명목 중심으로 셀을 배정하며 이미지가 새로 들어오면 Processor 순서와 마스크를 별도로 확인해야 한다. 원래 사설 GPU Runner의 N832 4×3 경로는 3×4 좌표를 전치하고 실제 토큰 순서를 검증했다. 공개 fixture CPU 테스트는 2×4, 3×4, 4×4 계약을 재생한다.

## Historical function checks

- P1에서 Coverage 수학·Oracle, 첫 Decoder native Query Attention, Vision 관측의 feature 불변성, 물리적 Visual64 압축과 빈 KV에서의 Decoder 실행을 검증했다.
- P2에서 N576·N832·N1088 각각 A–F GPU Smoke를 완료했다. N832·N1088에서는 Coverage Reference 12/12가 허용 범위 안이었다. P2-A의 과거 N576 Coverage 독립 계산에는 필요한 배열이 저장되지 않았다. Smoke 시간은 단일 입력의 기능 진단이다.
- Batch1 No-Padding SDPA·FP16 조건을 사용했다. Left Padding의 별도 Native 비교는 평균 logits 오차 0.015676으로 기준 0.01을 넘었다.
- 기존 환경은 Windows, NVIDIA GTX 1660 SUPER, Python 3.11.15, PyTorch 2.13.0+cu126, Transformers 5.15.1이었다. 모델 revision과 평가·파서 정책은 [공개 설정](../configs/evaluation.json)에 있다.

이 항목은 기존 실험 기록의 범위를 설명한다. 이 GitHub 공개본의 CPU 검사는 합성 입력·가중치 없는 소형 초기화 모형과 발췌된 공간 fixture를 사용한다.

## Dev100 evidence

공식 POPE Random에서 고정한 100장·600문항을 Full과 A–F가 공유했다. Pilot 420개 및 후속 3,780개로 총 4,200개 응답이었다. 두 파서의 불일치와 Strict Invalid는 모두 0건이었다. 런타임 계상 E2E는 조건당 540개 문항의 Processor, 공유 입력 준비와 방법별 wall time을 합한 값이다. 독립 반복 지연 또는 TTFT 벤치마크는 수행하지 않았다. 해시와 자료 범위는 [결과 출처](../results/PROVENANCE.json)에 있다.

## Independent Test

Test400의 후보와 로컬 검토 도구만 준비됐다. 확정된 Manifest 또는 실제 독립 Test 결과는 없다. 공개본에는 후보, 검토 파일, 연구 데이터와 사설 GPU Launcher를 포함하지 않는다.
