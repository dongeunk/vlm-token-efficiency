# Study4 · Spatially-Constrained Visual Token Pruning

SmolVLM-256M-Instruct에서 같은 K64로 전역 Coverage와 다섯 공간 대표 선택을 비교했다. 공식 POPE Random Dev100에서 **D는 A 대비 Accuracy +3.33%p, F1 +7.32%p**였고, Accuracy 차이의 95% CI는 **[+0.67, +6.17]%p**였다. F는 A보다 정확도가 낮았다. [최종 보고서](docs/STUDY4_FINAL_RESEARCH_REPORT.md) · [한국어 연구 요약](docs/STUDY4_RESEARCH_SUMMARY.md) · [결과 그림](docs/figures/STUDY4_DEV100_RESULTS.svg)

| 방법 | 토큰 선택 |
|---|---|
| Full | 전체 시각 토큰 |
| A | Global Coverage Greedy64 |
| B | Grid-Center25 + Coverage39 |
| C | Grid-Medoid25 + Coverage39 |
| D | Grid-Query Attention25 + Coverage39 |
| E | Grid-Vision Attention25 + Coverage39 |
| F | Grid-Constrained Coverage25 + Coverage39 |

B–F는 5×5 셀당 대표 하나를 고른 뒤 Seed25의 실제 Coverage 상태에서 Greedy39를 이어 간다. A는 CoverPruner-style SmolVLM 이식이다. Dev100은 탐색적이며 독립 Test400은 미실시다. Decoder Prefill 구간은 줄었고 종단간 가속은 확립되지 않았다. [검증 범위](docs/VALIDATION.md)

## 재현

저장된 비식별 이미지별 count로 Accuracy·F1·Recall 및 이미지 단위 bootstrap 10,000회 95% CI를 표준 라이브러리만으로 검산한다. 저장된 paired 정답 전환의 순변화를 확인한다. 개별 응답이 없어 두 파서의 원문 재채점과 전환 건수 각각의 독립 재구성은 이 공개본 범위 밖이다.

```console
python -I -B study4/scripts/reproduce_dev100.py --bootstrap
```

합성·tiny-model CPU 테스트는 Python 3.11과 [관측 의존성](requirements-cpu.txt)을 사용한다. PyTorch는 플랫폼에 맞는 CPU wheel을 설치한다. 테스트의 합성 모델은 다운로드한 가중치를 사용하지 않는다. 명령은 저장소 루트 기준이다. [PyTorch 설치](https://pytorch.org/get-started/locally/)

```console
python -m pip install -r study4/requirements-cpu.txt
python -I -B study4/scripts/run_cpu_tests.py
```

원래의 모델 실행에는 별도로 받은 SmolVLM-256M-Instruct 모델 및 Processor, COCO 이미지와 공식 POPE Random 주석, 연구 내 고정 Dev100 Manifest, Batch1 No-Padding SDPA·FP16 지원 환경, 사설 평가 오케스트레이션이 필요했다. [공개 설정](configs/evaluation.json)은 실행 정책과 해시를 보존한다. 이 자료만으로 고정 Dev100 GPU 평가 전체를 실행할 수는 없다. COCO와 모델은 각 원 출처에서 이용 조건을 확인한다. [모델](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) · [COCO](https://cocodataset.org/#download) · [공식 POPE](https://github.com/RUCAIBox/POPE)

## 파일과 공개 경계

- [결과 집계·출처](results/README.md) · [공개 파일 목록](PUBLIC_FILES.md)
- 원본 이미지·주석·Manifest, 응답·체크포인트, 모델·캐시, Test400 검토 데이터, Study3 및 권리 미확인 논문 파일은 공개 파일에서 제외했다. 그림에는 COCO 사진이 없다.
- 논문과 upstream 데이터·모델은 각각의 출처에 귀속된다. 이 추가 작업은 저장소 기존 [LICENSE](../LICENSE)·[NOTICE](../NOTICE) 및 과거 공개 범위 문서를 보존한다.
