> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/pope_spatial_floor_80_evaluation/full_v2_20261024_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Spatial Floor POPE80 × 5방법 — 최종 실행 및 해석 보고서

## 1. 실행 및 무결성 검증 요약

**800개 고유 key를 모두 완료했다. Smoke 20개를 그대로 재사용했고 신규 generation은 정확히 780회다. 오류·OOM·audit 실패·누락은 0건이다.**

- 80이미지 × 공식 Yes/No 2문항 × 5방법. 네 그룹 각각 20이미지/200결과.
- SmolVLM-256M-Instruct, FP16/eager, K64/G16, batch1, greedy16/beam1, fresh cache.
- cached checkpoint/processor/template/config/source SHA, v2 manifest와 이미지 SHA, Smoke 20개 key 및 audit를 실행 전후에 확인했다.
- 실제 embedding `[49280,576]`, generation pad/BOS/EOS=`2/0/49279`를 유지했다. `pad_token_id=128002` 경고는 config 표시 과정의 기본 config 생성에서 발생했으며 실제 generation pad에는 쓰이지 않았다. 경고를 숨기거나 설정을 수정하지 않았다.
- 800개 모두 정확한 K, unique/range/order, 실제 visual placeholder 삭제, nonvisual token/embedding/mask 보존, 실제 prefill/RoPE/causal mask/fresh KV-cache 및 generation 중 vision/connector 재실행 0을 통과했다.
- 기존 파일 769개의 크기/SHA가 모두 동일하다. 기존 30문항 결과, v1/v2, Smoke, selector/runner/parser를 수정하지 않았다.
- 모델 revision: `7e3e67edbbed1bf9888184d9df282b700a323964`.
- 실험 fingerprint: `74e81a1a7b87f45b89c5ba4f58dfadf15cec6d1b6121a9388de9d1c080c1aaf7`.
- runner 전체 1566.48초(약 26분 6초), 신규 generation 합 606.66초, 주 경로 78이미지 feature 추출 합 767.20초. 관찰/검증/I/O를 포함하며 speedup 주장에 사용하지 않는다.
- PyTorch peak allocated 2458.05 MiB / peak reserved 3230 MiB. GPU 시작 직전 free 3984 MiB였고, 다른 데스크톱/브라우저/클라이언트 프로세스 목록도 OBSERVATION.json에 보존했다. WDDM의 프로세스별 VRAM은 N/A였다.
- F(S)를 위해 기존 Smoke 두 이미지의 feature만 복구해 원래 tensor SHA와 완전 일치함을 확인했다. 답변은 재생성하지 않았다. 실제 새 780회에서도 이미지별 feature/selection을 공유하고 질문·방법별 입력/cache는 독립적으로 만들었다.

전체 audit 검증 (local artifact: `analysis/VALIDATION.json`), 실행 요약 (local artifact: `run/summary.json`), 실행 config/SHA (local artifact: `run/config.json`), 관찰 기록 (local artifact: `OBSERVATION.json`).

## 2. 전체 다섯 방법의 정확도 비교

| 방법 | Semantic 정답 | Semantic Accuracy | Strict 정답 | Strict Accuracy | Semantic invalid |
|---|---:|---:|---:|---:|---:|
| Random | 124/160 | 77.500% | 92/160 | 57.500% | 9/160 |
| Pure GC | 129/160 | 80.625% | 90/160 | 56.250% | 6/160 |
| Ours | 131/160 | 81.875% | 94/160 | 58.750% | 4/160 |
| A: Local-only with Floor | 131/160 | 81.875% | 109/160 | 68.125% | 3/160 |
| B: Laplacian Quota with Floor | 125/160 | 78.125% | 106/160 | 66.250% | 4/160 |

Ours−Pure GC는 +2문항/+1.25pp, 이미지 paired bootstrap 95% CI는 **[-1.875,+4.375]pp**다. Ours와 A의 점추정은 같지만 개별 답변까지 같다는 뜻이나 동등성의 증명은 아니다.

Ours−A는 0pp [-5,+5], A−B는 +3.75pp [-0.625,+8.125]다. 모두 탐색적 pointwise CI다. A/B는 **Free Allocation vs Laplacian Quota**이며 Equal vs Adaptive 또는 S²Prune 전체 재현이 아니다.

모든 Accuracy 분모에 invalid를 포함했다. Strict/Semantic Yes/No별 Accuracy, Precision/Recall/F1, prediction ratio, confusion matrix는 전체·그룹 지표 (local artifact: `analysis/metrics.json`), CSV (local artifact: `analysis/group_metrics.csv`)에 있다.

## 3. 네 이미지 그룹별 성능 비교

각 셀은 Semantic 정답/40이다. 각 그룹의 독립 이미지 수는 20장이다.

| 그룹 | Random | Pure GC | Ours | A | B | Ours−GC |
|---|---:|---:|---:|---:|---:|---:|
| Low C / Concentrated | 27 | 34 | 32 | 34 | 31 | −5.0pp |
| Low C / Dispersed | 34 | 31 | 32 | 33 | 32 | +2.5pp |
| High C / Concentrated | 31 | 30 | 33 | 30 | 28 | +7.5pp |
| High C / Dispersed | 32 | 34 | 34 | 34 | 34 | 0.0pp |

Group comparison (local artifact: `analysis/figures/group_accuracy.png`)

## 4. Primary Interaction 분석 및 신뢰구간

사전 지정대로 이미지 안의 Yes/No를 함께 묶고, 그룹별 이미지 20장을 10,000회 paired bootstrap했다. seed42, 95% percentile CI이며 interaction은 같은 반복의 HD−LC다. 그룹/질문/parser/Primary를 결과에 맞춰 바꾸지 않았다.

| 추정량 | 점추정 | 95% CI |
|---|---:|---:|
| Δ_HD: High C/Dispersed의 Ours−GC | 0.0pp | [0.0,0.0]pp |
| Δ_LC: Low C/Concentrated의 Ours−GC | −5.0pp | [−12.5,0.0]pp |
| Interaction = Δ_HD−Δ_LC | +5.0pp | [0.0,12.5]pp |

**양의 interaction은 HD의 성능 이득이 아니라 LC의 손실 점추정에서 나왔다.** HD에서는 Ours/GC의 40개 semantic 판정이 모두 동일했다(공동 정답34/공동 오답6). 20개 이미지 차이가 전부0이라 경험적 bootstrap CI가 [0,0]으로 퇴화했으며, 모집단 동등성이나 향후 차이 부재를 증명하지 않는다.

LC의 손실과 interaction CI 모두 0을 포함한다. 따라서 세 주장은 구분해야 한다.

1. 복잡·분산 그룹의 절대 이득: 이번 표본에서 관측하지 못했다.
2. 단순·집중 그룹의 손실: −5pp 점추정은 있으나 불확실하다.
3. 두 그룹의 효과 차이: +5pp 점추정은 있으나 불확실하다.

여기서 interaction은 사전 정의한 두 코너의 효과 차이로, 완전한 2×2 모형의 C×D interaction 계수와 구분한다. 분포·추정값 (local artifact: `analysis/primary.json`), 재표집 인덱스/분포 (local artifact: `analysis/primary_bootstrap.npz`).

Primary estimates (local artifact: `analysis/figures/primary_interaction.png`)

## 5. Ours가 유리한 사례와 불리한 사례

전체 160문항에서 Ours만 정답5, GC만 정답3, 둘 다 정답126, 둘 다 오답26이었다. 아래는 **모든** 정답 전환이며 사례 설명을 위한 사후 분석이지 표본/그룹 재선정이 아니다.

| QID | 그룹 | 질문 대상 / GT | Ours | GC | 전환 유형 |
|---|---|---|---|---|---|
| 870 | Low C/Dispersed | stop sign / No | No·정답 | invalid | GC의 미해결 응답→명확한 부정 |
| 1290 | High C/Concentrated | baseball glove / No | No·정답 | invalid | 같은 유형 |
| 2109 | Low C/Dispersed | wine glass / Yes | Yes·정답 | No·오답 | 유효한 Yes/No 판정 변화 |
| 2650 | High C/Concentrated | parking meter / No | No·정답 | invalid | GC의 미해결 응답→명확한 부정 |
| 2882 | High C/Concentrated | cat / No | No·정답 | invalid | 같은 유형 |
| 1141 | Low C/Dispersed | pizza / Yes | invalid | Yes·정답 | Ours의 미해결 응답 발생 |
| 2585 | Low C/Concentrated | tie / Yes | No·오답 | Yes·정답 | 유효한 Yes/No 판정 변화 |
| 2934 | Low C/Concentrated | baseball glove / No | invalid | No·정답 | Ours의 미해결 응답 발생 |

특히 **High C/Concentrated의 +3문항은 전부 GC semantic invalid가 명확한 No로 바뀐 사례**다. 이를 객체 존재 인식의 개선이라고 단정할 수 없다. 전체적으로 GC/Ours의 의미상 유효한 오답은 각각25개로 같고, semantic 미해결은6→4개였다. 순 정답 +2는 이 미해결 수 감소와 대응한다. 개별 유효 판정 변화도 wine glass와 tie에서 반대 방향으로 각각 관측됐다.

semantic invalid 해결은 단순히 Strict의 선행 Yes/No 형식을 맞춘 것과도 구분한다. 예를 들어 “no mention ... cannot determine”에서 명확한 시각적 부정으로 응답 내용이 바뀌었다. 모델 내부 원인은 미검증이다.

원문·두 parser·정오·지역 count/복잡도 heatmap은 상세 사례 보고서 (local artifact: `analysis/REPORT.md#5-ours가-유리불리한-사례`), 모든 질문 전환 (local artifact: `analysis/question_transitions.csv`), 오류 유형 집계 (local artifact: `analysis/error_taxonomy.json`)에 있다. 이미지의 실제 객체 증거가 특정 token에 담겼다고 주장하지 않는다.

## 6. Local/global 및 Spatial Coverage 분석

각 이미지를 한 번씩 집계한 80개 selection의 평균이다. global view는 지역 통계에서 제외했다.

| 방법 | local / global 평균 | zero-region 평균 | Coverage F(S) 평균 |
|---|---:|---:|---:|
| Random | 60.25 / 3.75 | 0.8750 | 137.5186 |
| Pure GC | 58.7625 / 5.2375 | 0.1625 | 154.4561 |
| Ours | 60.10 / 3.90 | 0 | 152.9021 |
| A | 64 / 0 | 0 | 152.4285 |
| B | 64 / 0 | 0 | 148.5395 |

- GC는 68/80이미지에서 이미 모든16지역을 덮었고, 12이미지는 zero-region이 있었다. Ours는 모두0이다.
- GC zero>0인12이미지에서는 Ours/GC 정답 차이가 없었다. 모든8문항의 정답 전환은 GC가 이미16지역을 덮은 이미지에서 발생했다. 따라서 이번 정답 변화가 빈 지역을 채운 효과라는 증거는 없다.
- Ours의 F(S)는 80/80이미지에서 GC보다 낮았지만 평균 Semantic Accuracy는 +1.25pp였다. 목적함수와 VQA Accuracy를 동일시할 수 없다. 역상관의 일반 법칙을 입증하는 결과도 아니다.
- minimum-one은 지역 보장이고 정답 보장은 아니다. zero-region 역시 global view와 encoder 비국소 정보 때문에 그 영역 정보의 완전 소실을 뜻하지 않는다.
- density는 지역별 선택 수/실제 candidate capacity로 계산했다. 동적 crop layout, 원래 view 순서와 global 경계를 사용했다. 원본 픽셀 위에 임의 token overlay를 만들지 않았다.

이미지·방법별 count/density/F (local artifact: `analysis/per_image_method.csv`), 그룹별 4×4 평균 (local artifact: `analysis/region_group_means.json`), 요약 통계 (local artifact: `analysis/spatial_and_coverage.json`).

## 7. C·D·H와 성능 차이의 관계

C는 긴 변672 전처리의 전체 Laplacian variance, D는 전체 COCO 주석 객체 중심 분산, H는 원본 이미지 지역 complexity의 std/(mean+epsilon)이다. **높은 H와 높은 C, 단순 배경과 낮은 C, 질문 관련 객체 분산과 전체 D를 같은 분류로 취급하지 않았다.**

Ours−GC 이미지 평균 정답 차이와의 전체 Spearman은 C=+0.188, D=−0.037, H=+0.072였다. 비영(非零) 이미지 차이가8/80개뿐인 탐색적 기술 상관으로 어떤 지표가 원인을 더 잘 설명한다고 결론 내리지 않는다.

지원 Yes72문항에서 질문 대상 개수·native 위치·segmentation/bbox 크기·분산도도 분석했다. 주석 없는 Yes8개는 위치 필드를 NA로 유지하고 정답 평가에는 포함했다. 이 Yes 분석에서 정답 전환은3문항뿐이므로 위치/크기 상관을 일반화하지 않는다. 주석 없는 영역을 배경으로 만들거나 object-hit token을 추정하지 않았다.

C/D/H 및 모든 방법 대비 상관 (local artifact: `analysis/descriptor_associations.json`), 질문 대상 분석 (local artifact: `analysis/yes_target_annotation_analysis.csv`), 위치·크기 상관 (local artifact: `analysis/target_associations.json`).

## 8. 기존 30문항 탐색 결과와 비교

기존 K64에서는 GC28/30, Ours25/30이었고, K128에서는 각각27/30이었다. 이번 신규160/K64에서는 GC129/160, Ours131/160이다. 서로 다른 표본·추출 설계이며 이번 네 그룹은 의도적으로 균등하게 구성됐다. 기존 H rank 3그룹과 이번 C×D 4그룹도 동일하지 않다. 이를 paired한 전체 성능 개선이나 공식 POPE 결과로 비교하지 않는다.

기존 Full26/30은 저장된 기존30문항의 기준이다. 신규160문항 Full Model은 실행하지 않았으므로 shared 오답이 원래 모델 한계인지 pruning 효과인지 분리할 수 없다. 기존 결과 대조 (local artifact: `analysis/historical_30_comparison.json`).

## 9. 원래 가설을 지지하는 근거와 반대되는 근거

- **제한적 방향 일치:** LC 손실과 양의 두 코너 효과 차이의 점추정은 원래 기대 방향이다. 두 CI는0을 포함한다.
- **핵심 미확인:** 가장 중요한 HD에서 절대 이득은 없었다. “복잡·분산이면 Floor가 도움”을 이번 표본으로 입증하지 못했다.
- **해석상 반대/제한:** 더 큰 점추정 이득은 High C/Concentrated에서 나왔고 모두 semantic invalid 해소였다. 빈 지역을 채운12이미지에서는 정답 변화가 없었다.
- **기술적으로 확인:** floor의 공간 보장은 성립한다. 이는 실제 객체 인식 향상이나 배경 영역 token 낭비의 원인 증명이 아니다.

결론은 **가설을 확립하지 못한 탐색적 결과**다. 가설이 틀렸다고 확정하거나 동등성이 입증됐다고도 말하지 않는다.

## 10. 한계 및 다음 연구 질문

1. 독립 이미지 수는 그룹20/총80이다. 160문항 또는800개 방법 결과를 독립 이미지 수로 취급하지 않았다. 의도적 균등 표본을 공식 POPE 전체로 일반화하지 않는다.
2. CI는 pointwise이며 다중 비교 보정이 없다. 특히 HD [0,0]은 표본 내 discordance가 없는 경험적 bootstrap의 퇴화다.
3. C/D/H는 배경 비율이나 질문 관련 증거 분포의 완전한 대리변수가 아니다. category/질문 난이도와 주석 누락도 고려해야 한다.
4. 많은 차이가 의미상 미해결 응답과 관련됐다. 인식·답변 불확실성·언어 표현 변화의 원인을 구분하려면 별도 진단이 필요하다.
5. 새 cohort의 저장 Full baseline이 없고 Random은 기존 seed42 한 개만 평가했다. 다른 모델/seed/budget으로 일반화하지 않는다.
6. F(S), spatial coverage, VQA Accuracy는 서로 다른 측정값이다. floor 이후 local/global 차이의 기술 상관을 primary 인과 분해로 해석하지 않는다.

## 11. 다음 단계에서 검토할 알고리즘 개선 후보 — 제안만

- 우선 원인 분리: 같은 cohort의 Full baseline 및 미해결 응답 사례 검토가 필요한지 결정한다. 현재 추가 inference는 실행하지 않았다.
- 같은 local-only 후보/demand/K/affinity에서 floor 유무만 다른 최소 통제군으로 view 배분과 local 내부 배분을 분리하는 기존 계획을 검토한다.
- GC가 이미 모든 지역을 덮은 경우의 floor 필요성과 greedy 경로/응답 변화를 연구 질문으로 삼는다. 결과에 맞춰 조건부 floor 임계값을 최적화하지 않았다.
- 질문 대상 증거의 위치·개수와 regional floor의 관계, Equal/Adaptive quota ablation은 별도 단계로 유지한다. A/B를 그 ablation으로 재명명하지 않는다.

**새 알고리즘 구현·추가 실험은 하지 않았다. 승인된 800개 완성과 저장 결과 분석을 마치고 중단한다.**

## 산출물 안내

- 상세 11절 보고서 및 그림 (local artifact: `analysis/REPORT.md`)
- 800개 raw predictions/생성 IDs/audit (local artifact: `run/predictions.jsonl`)
- 질문별 결과·시간·메모리 (local artifact: `analysis/question_results.csv`)
- 최종 검증 (local artifact: `analysis/VALIDATION.json`)
- 원래 Smoke timing을 포함한 provenance 연결 (local artifact: `analysis/PROVENANCE_LINKS.json`)
- 실행 관찰 및 source-local 재현 코드 (local artifact: `OBSERVATION.json`): observe_full.py, analyze_full.py, analysis_helpers.py, make_analysis_report.py가 같은 폴더에 있다.
- run과 analysis 각각의 FILES_SHA256.json을 유지하고, 전체 결과 폴더는 마지막 RESULT_FILES_SHA256.json으로 별도 seal한다.
