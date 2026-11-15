> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/pope_local_floor_paired/full_v1_20261025_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Fresh Paired320 — Local GC No-Floor vs A 최종 보고서

**완료: 80이미지·160공식 질문·2방법의 고유320개 결과를 검증했다. 성공한 Fresh Paired Smoke8개를 변경 없이 재사용하고 신규312회만 generation했다. 추가 방법/K/G/Full Model generation은 없었으며 GPU 프로세스 종료 후 저장 결과만 CPU로 분석했다.**

**연구 지위:** 기존800회 결과와 후속 CPU 진단을 본 뒤 설계한 탐색적 후속 비교다. 기존 Primary Hypothesis/HD−LC interaction·선정 그룹·parser·평가 기준을 소급 변경하거나 이 결과를 사전 Primary 가설의 재검증이라고 표현하지 않는다.

## 1. 실행·데이터·GPU 재현성 무결성

- 기존 v2의 동일80이미지와 이미지별 공식 Yes/No 각1문항(총160)을 사용했다. 네 C×D 그룹은 각20장으로 유지했다. 질문/정답/category/이미지/전처리/입력 IDs SHA를 검증했다.

- 총320 expected key와 실제 key가 일치하며 duplicate/미완료/pending/누락0이다. 모든 record envelope의 SHA, raw 사본312개, 재사용 Smoke8개와 원본의 bit identity를 확인했다.

- 같은 이미지에서 두 방법은 동일 FP16 projected tensor와 같은 FP32 affinity 객체를 공유했다. 전체 N개를 unit-weight Coverage demand로 유지하고 Local candidate만 사용했다. K64·global0·unique/range/sort, No-Floor empty init, A minimum-one을 확인했다.

- 실제 시퀀스에서 비선택 placeholder를 삭제하고 nonvisual ID/embedding/mask를 보존했다. 모든320 record의 실제 prefill·position/RoPE·eager causal mask·독립 fresh KV-cache·vision/projector 재실행0을 frozen audit로 확인했다.

실제 N832인70이미지는 원래 길이878–881에서 prefill110–113으로 정확히768토큰이 줄었다. N1088인10이미지는1143–1145에서119–121로 정확히1024토큰이 줄었다. 두 방법 모두 같은 질문의 길이는 같고 실제 retained visual placeholder는64개다.

**기존 A 재현성:** 전체80이미지의 새 GPU A indices·medoid Floor·completion 순서·affinity SHA가 과거 GPU 결과와 exact 일치했다. 이전 CPU final-set 불일치18장 모두 이번 GPU 검증에 포함됐고18/18 exact 재현했다. A의160문항 raw/generated IDs/Strict·Semantic도 과거와 동일하다. 새 비교의 A는 예전800 응답을 재사용한 것이 아니라 새 paired A160(그 중4개는 이번 연구 Smoke에서 생성)을 기준으로 한다.

CPU/GPU 차이를 tolerance로 무시하거나 FP64/epsilon/tie-breaking으로 맞추지 않았다. 과거 CPU에서는 FP32 near-tie score와 최종indices 차이가 있었고, 이번 같은 GPU backend에서는 전부 재현됐다. 이는 이전 CPU 선택을 과거 GPU 선택과 동일하다고 간주해도 된다는 뜻은 아니다. 원본 GPU affinity 전체가 과거에 저장되지 않았으므로 CPU/GPU 차이의 모든 수치적 경로를 증명한 것은 아니다.

checkpoint revision `7e3e67edbbed1bf9888184d9df282b700a323964`, FP16/eager, greedy16/num_beams1/batch1을 유지했다. GPU 시작: NVIDIA GeForce GTX 1660 SUPER, 6144 MiB, 2038 MiB, 3929 MiB, 617.42. 실제 vocabulary49280·dimension576·generation pad/BOS/EOS=2/0/49279를 검증했다. pad128002 기본 config 경고는 숨기지 않고 GPU_STDERR.log에 보존했으며 실제 generation config는 수정하지 않았다.

## 2. 전체 Semantic / Strict 성능

| 방법 | Semantic 정답/160 | Semantic Acc | Strict 정답/160 | Strict Acc | Semantic invalid | Strict invalid | Yes/No Semantic Acc |
|---|---|---|---|---|---|---|---|
| Local GC No-Floor | 131 | 81.875% | 104 | 65.000% | 3/160 (1.875%) | 32/160 (20.000%) | 71.25% / 92.50% |
| A: Local-only + Floor | 131 | 81.875% | 109 | 68.125% | 3/160 (1.875%) | 28/160 (17.500%) | 71.25% / 92.50% |

| 방법 | Parser | Precision | Recall | F1 | Yes prediction ratio | GT Yes → Yes/No/invalid | GT No → Yes/No/invalid |
|---|---|---|---|---|---|---|---|
| Local GC No-Floor | strict | 0.9344 | 0.7125 | 0.8085 | 38.125% | [57, 20, 3] | [4, 47, 29] |
| Local GC No-Floor | semantic | 0.9344 | 0.7125 | 0.8085 | 38.125% | [57, 22, 1] | [4, 74, 2] |
| A: Local-only + Floor | strict | 0.9344 | 0.7125 | 0.8085 | 38.125% | [57, 19, 4] | [4, 52, 24] |
| A: Local-only + Floor | semantic | 0.9344 | 0.7125 | 0.8085 | 38.125% | [57, 22, 1] | [4, 74, 2] |

모든 Accuracy의 분모160에 invalid를 포함하고 오답으로 처리했다. invalid를 No로 바꾸지 않고 frozen parser 판정만 사용했다. 원문과 저장 판정은 변경하지 않았다.

두 방법의 Semantic 정답 수와 aggregate confusion matrix가 같아도 문항별 응답은 같지 않다. Semantic prediction은10문항에서 다르고, 이 중8문항은 유효한 Yes/No 주장의 반전으로 정오가 바뀌었다. 나머지 Q1345·Q1373은 invalid와 유효한 No 간 전환이지만 둘 다 GT Yes에 대한 오답이다. 두 방법은 서로 다른3문항에서 invalid를 기록했다.

## 3. A−No-Floor paired 차이와 이미지 단위95% CI

| 범위 | Semantic A−No-Floor [95% CI] | Strict A−No-Floor [95% CI] | Semantic A만 정답 / No-Floor만 정답 / 둘 다 정답 / 둘 다 오답 |
|---|---|---|---|
| 전체 | +0.000 pp [-3.750, +3.125] | +3.125 pp [-1.250, +7.500] | [4, 4, 127, 25] |
| Low C / Concentrated | +2.500 pp [+0.000, +7.500] | +7.500 pp [+0.000, +15.000] | [1, 0, 33, 6] |
| Low C / Dispersed | +0.000 pp [-7.500, +7.500] | +5.000 pp [-5.000, +15.000] | [1, 1, 32, 6] |
| High C / Concentrated | -2.500 pp [-10.000, +5.000] | +0.000 pp [-10.000, +10.000] | [1, 2, 29, 8] |
| High C / Dispersed | +0.000 pp [-7.500, +7.500] | +0.000 pp [-10.000, +10.000] | [1, 1, 33, 5] |

두 질문을 하나의 이미지 cluster에 묶어 각 그룹20이미지를 복원추출했다. 그룹 내 numeric image_id 순서, NumPy PCG64/seed42/10,000회, replicate→group→image-position draw 순서를 고정했다. 전체는 네 그룹의 동일 가중 평균이다. CI는 linear percentile의 pointwise95%이며 다중비교 보정은 하지 않았다. 질문160개를 독립 이미지160개로 취급하지 않았다. 그룹에서 관측 차이가 모두0인 경우의 CI[0,0]은 모집단 동등성 증거가 아니다.

Paired CI (local artifact: `analysis/figures/paired_CI.png`)

## 4. 네 고정 C×D 그룹

| 그룹 | No-Floor Semantic/40 | A Semantic/40 | No-Floor Strict/40 | A Strict/40 | No-Floor/A Semantic invalid |
|---|---|---|---|---|---|
| Low C / Concentrated | 33 | 34 | 27 | 30 | 1/2 |
| Low C / Dispersed | 33 | 33 | 27 | 29 | 0/0 |
| High C / Concentrated | 31 | 30 | 24 | 24 | 1/1 |
| High C / Dispersed | 34 | 34 | 26 | 26 | 1/0 |

C는 장변672 Laplacian 구조 proxy, D는 COCO 전체 주석 객체 중심 분산, H는 원본 지역 complexity의 상대 편차다. 서로 같은 개념으로 분류하지 않았다. frozen C Q1/Q3 및 D 중앙값으로 정한 그룹을 유지하며 결과에 유리한 새 H/복잡도 경계는 만들지 않았다. C·D·H 상관은80이미지 기술통계이고 인과적 설명력 비교가 아니다.

High C/Dispersed는 두 방법 모두34/40이지만 Q598은 No-Floor만 정답, Q2099는 A만 정답이다. Q1345도 invalid/No 응답이 다르되 둘 다 오답이다. 따라서 이 그룹의 paired 순차이0pp를 문항별 동일 판정으로 해석하지 않는다. 이는 기존 Ours/GC의 해당40문항 정오가 모두 같았던 관측과 다르다.

Group accuracy (local artifact: `analysis/figures/group_accuracy.png`)

## 5. Invalid·Strict/Semantic·길이 제한과 정답 전환

| 방법 | 유효한 Semantic 오답 | Strict/Semantic 판정 불일치 | Strict invalid→Semantic valid | Strict valid→Semantic invalid | 종료 유형 | invalid 종료 유형 |
|---|---|---|---|---|---|---|
| Local GC No-Floor | 26 | 31 | 30 | 1 | {'eos_before_cap': 159, 'cap_without_eos': 1} | {'cap_without_eos': 1, 'eos_before_cap': 2} |
| A: Local-only + Floor | 26 | 27 | 26 | 1 | {'eos_before_cap': 159, 'cap_without_eos': 1} | {'cap_without_eos': 1, 'eos_before_cap': 2} |

Semantic 정오 전환은 총8문항이고 유형은 {'opposite_valid_assertions': 8}다. Strict 정오 전환은17문항이다. 아래 전수 표에서 원문을 함께 비교한다. invalid/유효 응답 전환과 양쪽의 유효한 존재·부재 주장의 반전을 구분하며, 어느 토큰이 객체 정보를 보존/제거했는지 원인으로 단정하지 않는다.

| QID/image/그룹 | 공식 질문·정답 | No-Floor raw / Semantic | A raw / Semantic | 유형 | No-Floor zero / 공통 token |
|---|---|---|---|---|---|
| 598/105732/High C / Dispersed | Is there a snowboard in the image? GT=no | ` No, there is no snowboard present in the image.` / no | ` Yes, there is a snowboard in the image.` / yes | opposite_valid_assertions | 0 / 45 |
| 903/430052/Low C / Concentrated | Is there a bird in the image? GT=yes | ` No, there is no bird present in the image.` / no | ` Yes, there is a bird in the image.` / yes | opposite_valid_assertions | 2 / 34 |
| 1027/153300/Low C / Dispersed | Is there a spoon in the image? GT=yes | ` Yes, there is a spoon in the image.` / yes | ` No, there is no spoon present in the image.` / no | opposite_valid_assertions | 0 / 41 |
| 1217/112664/High C / Concentrated | Is there a suitcase in the image? GT=yes | ` Yes, there is a suitcase in the image.` / yes | ` No, there is no suitcase present in the image.` / no | opposite_valid_assertions | 0 / 46 |
| 2099/114941/High C / Dispersed | Is there a car in the image? GT=yes | ` No, there is no car present in the image.` / no | ` Yes, there is a car in the image.` / yes | opposite_valid_assertions | 0 / 43 |
| 2400/288576/High C / Concentrated | Is there a skateboard in the image? GT=no | ` Yes, there is a skateboard in the image.` / yes | ` No, there is no skateboard present in the image.` / no | opposite_valid_assertions | 0 / 32 |
| 2813/396068/Low C / Dispersed | Is there a remote in the image? GT=yes | ` No, there is no remote present in the image.` / no | ` Yes, there is a remote in the image.` / yes | opposite_valid_assertions | 0 / 36 |
| 2883/69863/High C / Concentrated | Is there a vase in the image? GT=yes | ` Yes, there is a vase in the image.` / yes | ` No, there is no vase present in the image.` / no | opposite_valid_assertions | 0 / 42 |

There is no ... in the image처럼 완결된 시각적 부정 진술을 Strict가 invalid로 처리하는 차이와, no mention/cannot determine처럼 대상 부재를 단정하지 않는 응답을 Semantic이 보류하는 차이를 구분했다. 16토큰·EOS 없음은 길이 상한 도달과 일치하지만 더 길게 생성하면 정답이 됐을 것이라고 추론하지 않았다. EOS로 끝나도 지원 문법 밖의 표현을 frozen parser가 invalid로 처리할 수 있다. 새 parser로 점수를 재평가하지 않았다.

전체320응답의 raw·IDs·frozen 판정 이유·EOS/cap 분류는 all_response_diagnostics.csv, Semantic invalid 원문은 semantic_invalid_responses.csv, Strict 정오 차이는 strict_discordant_questions.json에 보존했다.

Strict에서 Semantic으로의 정오 변화는 No-Floor28개 개선/1개 악화, A23개 개선/1개 악화다. 그러므로 A의 Strict 정답5개 우위가 Semantic 우위로 이어지지 않는다. 기존 Ours/GC 정오 전환8개 중6개가 invalid/유효 응답 사이의 변화였던 것과 달리, 이번 정오 전환8개는 모두 유효한 반대 주장이었다. 특정 시각 정보가 어느 토큰에 의해 달라졌는지는 여전히 확인하지 못했다.

## 6. 공간 배분·token overlap·Coverage F(S)

| 지표 | No-Floor | A |
|---|---|---|
| Local/Global selected | 모든 이미지64/0 | 모든 이미지64/0 |
| zero-region이 있는 이미지 | 10/80 | 0/80 |
| 전체 zero-region 수 | 11 | 0 |
| 이미지별 평균 region CV | 0.43875 | 0.39654 |
| F(S) 평균 | 153.876474 | 152.428513 |
| F(S)/N 평균 | 0.178458 | 0.176768 |

선택 token overlap은 평균 **60.879%** (범위 37.500%–75.000%)다. A_F−NoFloor_F 평균 **-1.447961**, 범위 [-2.273346, -0.493866]다. A F가 낮은 이미지는80/80이다. 이는 이 표본의 실제 GPU 선택에 대한 관측이며 greedy가 모든 constrained set보다 항상 높다는 정리는 아니다.

| 그룹 | No-Floor zero 이미지/20 | No-Floor/A zero-region 평균 | overlap 평균 | A−No-Floor F 평균 |
|---|---|---|---|---|
| Low C / Concentrated | 4 | 0.250/0.000 | 57.344% | -1.280478 |
| Low C / Dispersed | 4 | 0.200/0.000 | 59.688% | -1.455103 |
| High C / Concentrated | 1 | 0.050/0.000 | 61.875% | -1.536912 |
| High C / Dispersed | 1 | 0.050/0.000 | 64.609% | -1.519350 |

### 공간 공백 집합과 이미 덮인 집합

| 집합 | 이미지/문항 | A/No-Floor 정답 | paired 전환 | 새 Floor: 빈 지역 / 기존 덮인 지역 |
|---|---|---|---|---|
| NoFloor_has_zero | 10/20 | 18/17 | {'both_correct': 17, 'both_incorrect': 2, 'A_only_correct': 1} | 11/131 |
| NoFloor_fully_covered | 70/140 | 113/114 | {'both_correct': 110, 'both_incorrect': 23, 'NoFloor_only_correct': 4, 'A_only_correct': 3} | 0/954 |

A의 medoid Floor1,280개 중 No-Floor에 없던 토큰은1096개다. 이 중 No-Floor의 빈 지역에 해당한 것은11개, 이미 덮인 지역에 해당한 것은1085개다. A는 공백만 보충하는 절차가 아니며, 이미 덮인 지역의 대표 토큰을 강제 선택하고 후속 greedy 경로도 바꾼다. 공간 보장과 대표 선택/초기화 효과를 완전히 분리할 수 없다.

공백이 있던10이미지의 유일한 정오 전환은 Q903/image430052의 bird Yes로 A만 정답이다. 이 이미지의 No-Floor 공백은2개지만 frozen v2에서 bird의 COCO 위치 주석은 NA다. 따라서 빈 지역에서 새 토큰이 bird 정보를 회복했다고 주장할 수 없다. 나머지7정오 전환은 No-Floor도 이미 모든 지역을 덮은 이미지에서 발생했다.

양쪽의 전체 calibrated affinity와 completion demand는 같다. A의 Floor 선택은 기존대로 각 지역 A[P,P] demand sum의 medoid이며 이 지역 대표성 목적 자체가 비교 중인 초기화 절차의 일부다. No-Floor는 지역 점수 없이 all-N marginal gain에서 시작한다. 같은 후보 범위와 local/global64/0을 고정했으므로 이전 all-view 비교의 global 후보/개수 차이 confound를 제거했지만, Floor의 어느 구성요소가 VQA를 바꿨는지는 별도 식별하지 않는다.

같은 이미지에서는 같은 affinity로 F를 계산했고, 다른 이미지끼리는 N832/1088 수 차이를 고려해 F/N도 함께 보존했다. F가 높은 방법이 항상 정답이라는 목적함수-정확도 등치를 하지 않는다. C/D/H/F/overlap와 이미지별 paired Accuracy 상관은 descriptor_associations.json에 있으며, 적은 전환에 민감한 탐색적 기술통계다. 객체 bbox나 attention 관측 없이 특정 token이 질문 객체를 보존했다고 단정하지 않았다.

Spatial allocation (local artifact: `analysis/figures/spatial_allocation.png`)

Coverage and overlap (local artifact: `analysis/figures/coverage_overlap_accuracy.png`)

Descriptors (local artifact: `analysis/figures/descriptors_vs_accuracy_delta.png`)

## 7. 기존 Ours−Pure GC와의 관계

| 범위 | 기존 Ours−Pure GC Semantic [95% CI] | 신규 A−No-Floor Semantic [95% CI] |
|---|---|---|
| 전체 | +1.250 pp [-1.875, +4.375] | +0.000 pp [-3.750, +3.125] |
| Low C / Concentrated | -5.000 pp [-12.500, +0.000] | +2.500 pp [+0.000, +7.500] |
| Low C / Dispersed | +2.500 pp [-5.000, +10.000] | +0.000 pp [-7.500, +7.500] |
| High C / Concentrated | +7.500 pp [+0.000, +15.000] | -2.500 pp [-10.000, +5.000] |
| High C / Dispersed | +0.000 pp [+0.000, +0.000] | +0.000 pp [-7.500, +7.500] |

기존 비교는 전체 local+global 후보에서 Pure GC empty init과 Ours medoid Floor를 비교했다. 신규 비교는 두 방법의 후보를 local-only로 고정하고 global도 포함한 전체 N demand를 유지했다. 따라서 기존 +1.25pp와 새 차이는 같은 개입의 독립 반복값이 아니며 하나로 합치거나 차이의 차이를 순수 global token 효과로 해석하지 않는다.

같은160 공식 질문을 사용했고 fresh A가 기존 A의160응답과 완전히 재현돼 새 대조의 변화가 기존 A drift에서 온 증거는 없다. 그러나 전체80이미지와 기존800결과를 본 뒤 설계한 후속 연구이므로 독립 holdout 검증이 아니다. 원래 “복잡·분산 장면에서 Floor가 유리하다”는 사전 가설은 기존 all-view 비교에서 지지되지 않았다는 기록을 유지한다. 새 후보 조건의 결과를 이용해 과거 결론을 소급 변경하지 않는다.

문항별 기존 Ours/GC와 신규 A/No-Floor의 원문·정오, 양쪽 대조의 전환 교차표는 historical_Ours_GC_relation.csv/historical_relation.json에 보존했다. 같은 문항에서 승패가 일치하는지 기술적으로 비교했으며 원인 증거로 삼지 않았다.

두 비교에서 정오가 바뀐 문항은 각각8개지만 교집합은0개다. 신규 local-only 통제 결과가 과거 all-view 비교와 같은 실패 문항을 재현한 것은 아니다. 두 비교의 후보 조건과 초기 선택 경로가 달라 이 차이를 단일 원인에 귀속시키지 않는다.

Historical comparison (local artifact: `analysis/figures/historical_effect_comparison.png`)

## 8. 자원·오류·provenance

이번 full attempt는 **989.42초 (16.49분)**다. 사전 파일/SHA 검사를 제외하고 모델 로드·78이미지 처리·selector 검증·신규312 generation을 포함한다. 신규 generation 시간 합 192.35초, 신규78이미지 feature 추출 합 724.96초다. 재사용 Smoke8개의 generation 5.40초·feature2개 22.53초는 과거 별도 실행 비용으로 구분해 보존했다.

모든 관측 단계/재사용 Smoke를 포함한 PyTorch allocated 최고 **2404.57MiB**, reserved 최고 **3230.00MiB**다. helper가 단계마다 peak를 reset하므로 연속 전체-run profiler 최고값이라고 표현하지 않는다. 이번 주 경로 vision/projector는 각각78회, Smoke에서 각각2회로 총80이미지 한 번씩의 feature를 사용했다. generation 중 vision/projector 재실행0이며 OOM/CUDA/generation 오류0이다.

Feature/affinity는 이미지별 공유 비용이다. 기존 A API가 B completion도 내부 계산하지만 B generation은 없다. 해당 envelope는 비가산으로 기록하고 A floor/완성 및 unused B 시간과 중복 합산하지 않았다. 단일 실행의 warmup/검증/audit hook을 포함하므로 방법별 end-to-end speedup을 주장하지 않는다.

재사용8개 record는 원본 그대로라 상대 timing 참조가 원래 Smoke run을 기준으로 한다. REUSED_PROVENANCE_LINKS.json에 원래 record/image diagnostic/resource ledger와 SHA를 연결했다. frozen runner의 resource_ledger에 남은 CPU_preparation_mismatch_scope는 Smoke 전용 설명 문자열이며 full에서는 실제 범위와 맞지 않는다. 원문을 수정하지 않고 VALIDATION.json에 이 주석의 한계를 기록했으며 실제78 새 diagnostic+2 Smoke diagnostic, 320 record를 근거로 범위를 검증했다.

## 9. 결론과 해석 범위

**전체 Semantic A−No-Floor: +0.000 pp [-3.750, +3.125].** 전체 paired 차이의95% CI가0을 포함한다. 이 표본에서 효과 방향과 크기가 불확실하며, Floor 효과가 없거나 두 방법이 동등하다는 증거로 해석하지 않는다.

Spatial Floor가 지역 minimum-one을 보장하는 것은 코드/선택 counts로 확인했지만, 이 보장과 VQA 정확도 효과는 별개다. 같은 demand/candidate/K를 고정한 새 통제로 global 선택 개수 차이를 제거했어도 medoid 대표 선택과 greedy 초기화/경로가 함께 바뀐다. valid polarity 오류와 응답 불확실성/형식 변화 역시 관측만으로 완전히 분해할 수 없다.

80장은 기존30장을 제외한 사전 선정 C 극단×D 균형 탐색 코호트이며 공식 POPE 전체 대표 표본이 아니다. 객체 존재 여부 질문은 다지역 관계/공간 추론을 반드시 요구하지 않는다. C는 구조 proxy, D는 전체 COCO 주석 중심 분산, H는 지역 상대 편차로 각각 한계가 있고 질문 난이도/대상 정보의 공간 분포와 동일하지 않다. 소수 전환과 그룹·상관 분석으로 인과적 결론을 내리지 않는다.

이 작업은 결과 해석까지 완료하고 중단했다. 새 알고리즘·다른K/G·추가 generation·Git/Notion/GitHub 업데이트·패키지/데이터 다운로드는 진행하지 않았다.

## 산출물

- run/predictions.jsonl (local artifact: `run/predictions.jsonl`): 고유320개 원본 결과, 선택/응답/IDs/audit/시간/메모리/입력 SHA.

- analysis/VALIDATION.json (local artifact: `analysis/VALIDATION.json`): 총320 무결성·GPU A80/80 및 이전 CPU18/18 재현·A160응답 동일성.

- analysis/group_metrics.csv (local artifact: `analysis/group_metrics.csv`), paired_bootstrap_CI.json (local artifact: `analysis/paired_bootstrap_CI.json`), paired_bootstrap.npz: 전체/그룹/두 parser 지표·CI·실제 resample indices/분포.

- analysis/per_image_analysis.csv (local artifact: `analysis/per_image_analysis.csv`), regional_counts.csv (local artifact: `analysis/regional_counts.csv`): 80이미지/1,280지역의 counts·zero·overlap·F·medoid 정보.

- analysis/question_paired_results.csv (local artifact: `analysis/question_paired_results.csv`), semantic_discordant_questions.json (local artifact: `analysis/semantic_discordant_questions.json`): 전체160 paired 질문과 전수 Semantic 정오 전환.

- analysis/parser_and_termination.json (local artifact: `analysis/parser_and_termination.json`), all_response_diagnostics.csv/semantic_invalid_responses.csv: invalid·종료·응답 원문.

- analysis/GPU_A_reproduction.csv (local artifact: `analysis/GPU_A_reproduction.csv`), historical_relation.json (local artifact: `analysis/historical_relation.json`): 재현성과 이전 Ours/GC와의 관계.

- analysis/INTERPRETATION_DETAILS.json (local artifact: `analysis/INTERPRETATION_DETAILS.json`): 전체 응답 차이10개, 정오 전환8개, 주석 지원 및 실제 sequence 감소의 독립 재확인. interpret_saved_results.py는 저장 결과만 읽는다.

- analysis/resources.json (local artifact: `analysis/resources.json`), REUSED_PROVENANCE_LINKS.json (local artifact: `analysis/REUSED_PROVENANCE_LINKS.json`): 자원/재사용 출처와 원본 상대 참조의 해석.

- analyze_full.py/make_report.py: 저장된 결과만 읽는 CPU 재현 스크립트. source_snapshots: 실제 실행한 paired code3파일 사본. PROTECTION_BASELINE/INTEGRITY_VERIFICATION과 RESULT_FILES_SHA256: 기존/신규 결과 전수 크기·SHA.

CPU 분석 재현은 analyze_full.py --output <이 폴더 아래 새 경로>로 가능하며 기존 출력이 있으면 덮어쓰지 않는다. 결과 해시는 기존 실험의 원본 SHA나 실행 commit에 소급 적용하지 않았다. 과거 계획의 승인 대기 문구는 당시의 역사적 기록으로 보존했고 이번 승인은 새 GPU_EXECUTION_COMMAND/PRE_EXECUTION_CHECKS에 기록했다.
