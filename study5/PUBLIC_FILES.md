# Study5 공개 파일 목록

Study5 원본에서 선택 코드, 합성 검증, GPU 진입점을 복사했다. 공개 보고서의 개인 경로와 원시 입력 식별자를 정리했다. 원본 결과에서 집계 CSV/JSON과 결과 그림을 생성했다. 별도 Study4 체크아웃 없이 공개 CPU 검증을 실행할 수 있다.

원본 이미지·주석·평가 Manifest·모델 캐시·학습 가중치·원시 응답 DB·체크포인트·원시 GPU 로그·Tensor·로컬 감사 원본·권리 미확인 PDF·Study3 파일은 미포함이다.

| 파일 | 바이트 |
|---|---:|
| [.gitattributes](.gitattributes) | 375 |
| [configs/layout_geometry.json](configs/layout_geometry.json) | 4527 |
| [docs/figures/STUDY5_CANDIDATE79_RESULTS.png](docs/figures/STUDY5_CANDIDATE79_RESULTS.png) | 77132 |
| [docs/STUDY5_FINAL_RESEARCH_REPORT.md](docs/STUDY5_FINAL_RESEARCH_REPORT.md) | 14543 |
| [docs/STUDY5_P1_REPORT.md](docs/STUDY5_P1_REPORT.md) | 4496 |
| [docs/STUDY5_P2_GPU_SMOKE_REPORT.md](docs/STUDY5_P2_GPU_SMOKE_REPORT.md) | 5787 |
| [docs/STUDY5_P3A_READINESS_REPORT.md](docs/STUDY5_P3A_READINESS_REPORT.md) | 4750 |
| [docs/STUDY5_P3C_RESULTS_REPORT.md](docs/STUDY5_P3C_RESULTS_REPORT.md) | 9309 |
| [docs/STUDY5_PROTOCOL.md](docs/STUDY5_PROTOCOL.md) | 6492 |
| [docs/STUDY5_RESEARCH_SUMMARY.md](docs/STUDY5_RESEARCH_SUMMARY.md) | 2502 |
| [README.md](README.md) | 4637 |
| [requirements-cpu.txt](requirements-cpu.txt) | 141 |
| [results/candidate_summary.json](results/candidate_summary.json) | 13285 |
| [results/cluster_bootstrap.csv](results/cluster_bootstrap.csv) | 2632 |
| [results/image_cluster_counts.csv](results/image_cluster_counts.csv) | 6720 |
| [results/performance.csv](results/performance.csv) | 627 |
| [results/PROVENANCE.json](results/PROVENANCE.json) | 936 |
| [scripts/reproduce_candidate79.py](scripts/reproduce_candidate79.py) | 5314 |
| [scripts/run_cpu_tests.py](scripts/run_cpu_tests.py) | 1573 |
| [scripts/run_p3b_candidates.py](scripts/run_p3b_candidates.py) | 4033 |
| [scripts/smoke_gpu.py](scripts/smoke_gpu.py) | 11008 |
| [src/coverpruner_attention_probe.py](src/coverpruner_attention_probe.py) | 11982 |
| [src/coverpruner_checkpoint.py](src/coverpruner_checkpoint.py) | 14083 |
| [src/coverpruner_checkpointed_runner.py](src/coverpruner_checkpointed_runner.py) | 2180 |
| [src/coverpruner_coverage.py](src/coverpruner_coverage.py) | 11437 |
| [src/coverpruner_decoder.py](src/coverpruner_decoder.py) | 10978 |
| [src/coverpruner_execution_profile.py](src/coverpruner_execution_profile.py) | 984 |
| [src/coverpruner_native_validation.py](src/coverpruner_native_validation.py) | 3422 |
| [src/coverpruner_spatial_layout.py](src/coverpruner_spatial_layout.py) | 4806 |
| [src/coverpruner_spatial_selection.py](src/coverpruner_spatial_selection.py) | 10564 |
| [src/coverpruner_vision_attention.py](src/coverpruner_vision_attention.py) | 12153 |
| [src/study5_analysis.py](src/study5_analysis.py) | 4042 |
| [src/study5_eval.py](src/study5_eval.py) | 11173 |
| [src/study5_eval_gpu.py](src/study5_eval_gpu.py) | 4643 |
| [src/study5_layout.py](src/study5_layout.py) | 3910 |
| [src/study5_scoring.py](src/study5_scoring.py) | 1402 |
| [src/study5_selection.py](src/study5_selection.py) | 2625 |
| [tests/test_attention_probe_cpu.py](tests/test_attention_probe_cpu.py) | 11010 |
| [tests/test_checkpoint_recovery_cpu.py](tests/test_checkpoint_recovery_cpu.py) | 13955 |
| [tests/test_coverage_cpu.py](tests/test_coverage_cpu.py) | 19793 |
| [tests/test_decoder_cpu.py](tests/test_decoder_cpu.py) | 10766 |
| [tests/test_p3a_resume_cpu.py](tests/test_p3a_resume_cpu.py) | 6423 |
| [tests/test_spatial_selection_cpu.py](tests/test_spatial_selection_cpu.py) | 19402 |
| [tests/test_study5_selection_cpu.py](tests/test_study5_selection_cpu.py) | 6884 |
| [tests/test_vision_attention_cpu.py](tests/test_vision_attention_cpu.py) | 11876 |

목록 문서 자체와 저장소 루트 README 및 .gitignore를 제외한 45개 파일의 작업 트리 크기는 391,312바이트다.
