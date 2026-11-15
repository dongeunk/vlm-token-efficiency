> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3b2/phase_s3b2_20261114_01/CONFIG_AND_EXECUTION_PROTOCOL.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# S3-B2 고정 파일럿 실행·판정 계약

범위는 S3-A에 고정된10이미지·60문항의 Full/L/U/C 네 방법, 최대240 generation이다. main200은 실행·이미지로드·질문/정답분석 대상이 아니다. 실제 완료/지표는 최종 REPORT와 run ledger에 기록한다. 이 문서는 실행 중 결과에 따른 설정 변경을 허용하지 않는다.

## 데이터와 알고리즘

pilot_images/pilot_questions/pilot_expected_run_keys와 기존 protocol/selection lock의 SHA를 S3-A FILES_SHA256 index와 대조했다. pilot10/60·Yes30/No30·각이미지3yes/3no·qid/원문/정답/source record SHA·old110 제외를 확인했다. main disjoint는 변경 없는 pilot/selection hash와 기존 S3-A validation에 근거한다. main manifest 내용을 새로 파싱해 확인한 것이 아니다.

`PilotIOBoundary`는 main_images/main_questions/main_expected_run_keys, 전체 POPE annotation, 파일럿10장 외 모든 COCO JPEG/PNG 읽기를 차단한다. main manifest는 보호용 size/mtime stat만 확인한다. selection lock은 SHA만 계산하고 main ID 배열을 decode하지 않는다. 따라서 전체 annotation에서 본 평가 정답을 읽는 기존 generic validator/CLI를 호출하지 않는다. 새 pilot-only loader를 사용한다.

checkpoint는cached `HuggingFaceTB/SmolVLM-256M-Instruct` revision `7e3e67edbbed1bf9888184d9df282b700a323964`다. FP16/eager/batch1, greedy16, beams1, EOS49279/PAD2/BOS0, 동일processor/chat template를 유지한다. Full은 사용자지정 Native HF 정상경로다. L/U/C는 기존S3-B1 `run_pruned_generation`을 그대로 호출한다. 기존 runtime/allocator/Cell/ERC/parser/metric을 수정하지 않는다.

local48/global16/K64,4×4 region/base1+extra32, 동일recursiveCell·nativeFP16 subtraction→FP32 ERC·Layer0 이후gather·모든nonvisual보존·original1D RoPE·동일KV/decoder다. C만all-visual demand/local candidates/query-independent calibrated Coverage로예산을배분한다. query weight는없다. 각방법이JPEG/processor부터같은입력을다시만들고CPU Gate A tensor SHA와대조한다. 선택·allocation cache를이파일럿에서재사용하지않는다.

`CONFIG_LOCK.json`에는frozen설정·기존algorithm source SHA·pilot/source hash·명시적Native Full·Gate B 선택·timing 정의가있다. configuration_hash는canonical JSON SHA256이다. 변경된config와기존completed key를혼합하지않는다.

## Gate A와 B 문항 선정

실제CPU processor로60입력을확인했다. 실제832가9이미지,1088이1이미지이며비시각T는46~56이다. 각이미지L/U/C의합성features/ERC와기존S1 Cell을검증했다. CUDA/model/generation0 상태의증거는GATE_A와cpu_run_01에있다.

첫이미지는고정image554002다.6문항의**BPE/T 길이는모두46**이었다. 초기새orchestrator는서로다른T를과도하게요구해CPU에서중단했다. 모델/GPU generation은0회였고, 기존알고리즘이나질문을바꾸지않았다. 사용자지시의서로다른질문길이를원문문자길이로충족하도록, 고정6문항중가장짧은qid2902와가장긴qid2900을선택했다(qid tie-break). 이기준을실제GPU 결과전에config에기록했다.

따라서Gate B는원문질문길이가다른두문항을사용하지만실제BPE 길이변화검증은아니다. 이한계를명시한다. 실제다른T는고정파일럿의다른문항에서관측하며새질문을만들거나코호트를교체하지않는다.

Gate B는2문항×4방법=8회다.구조·예산·ERC·비시각보존·실제30층KV/gather/append·NaN/OOM·EOS/길이종료를검사한다. 기존full RuntimeObserver/PruningObserver의강한관측을사용한다. raw응답의format parser는정답없이호출할수있으나, 정확도로설정을조정하거나실행을결정하지않는다. 정상Gate B 결과8개를최종240에포함한다.

## 실행순서·중복방지·중단/재개

두Gate B qid를먼저처리하고, 나머지는고정manifest 순서다. 각문항은Full→L→U→C, 모델인스턴스는가능한1회로드로순차사용한다. warm-up/추가probe/추가질문generation은없다.

durable key는 `(image_id,question_id,method,configuration_hash)`다. Generation 전에append+fsync한start event를쓰고, 결과JSON과scientific evidence를저장한뒤completion event와SHA를쓴다. `run/EVENTS.jsonl`이기준이다. 최대240 unique reservations, Gate B8개completion 전에는Gate C를예약할수없다.

재개는기존completed key를검증하고재사용한다. 결과파일은있으나completion event만없는경우에는같은config의정상결과를검증해completion을복구한다. start만있고정상결과가없는모호한generation은자동재시도하지않는다. raw tensor checkpoint도보존하여후속CPU 기록복구를돕는다. 해결불가능한mapping/cache/decode/OOM/NaN은전체중단하며문항/이미지를빼거나재추출하지않는다. 기존검증알고리즘변경이필요하면멈추고수정안을보고한다.

## 채점·통계

240개가모두정상일때만기존S3-A `score_complete`와`paired_bootstrap`을호출한다. raw응답은그대로보존한다. frozen visual_semantic_yes_no_v1이주판정이며strict도함께저장한다. invalid는별도confusion column으로남기고Accuracy에서오답, invalid_yes는Recall/F1의FN에포함한다. invalid_no를TN으로보지않으며positive F1이이를직접벌점화하지않으므로Accuracy/Invalid/Strict를함께보고한다.

주지표는Semantic F1, 주비교는C−L이다. 공식original POPE Random의고정60문항부분집합이며공식전체benchmark 점수가아니다. image-cluster6문항을함께resample,10000회PCG64 seed42,95%percentileCI/linear quantile, confusion 합산후micro F1재계산을사용한다. pilot10장의CI는불안정할수있으며유의성/순위로algorithm·K·parser·생성길이·본평가코호트를선택하지않는다.

EOS없이16개이면possible truncation으로기록하고문항을제외하지않는다.16번째EOS는truncation으로세지않는다. length민감도실험이나길이확장은별도사전등록/승인범위이며이번에는수행하지않는다.

## 진단과시간의해석

read-only module hooks와sys.setprofile로실제mergedH0 준비·첫greedyargmax·Layer0/suffix·Coverage calibration/greedy CPU구간을관측한다. 기존function을대체하거나runtime/선택로직을수정하지않는다. TTFT E2E는각route JPEG open부터첫token argmax 완료, decoder TTFT는inputs_merger return부터같은끝점이다. selection overhead는CPU handoff+allocation+partition+ERC norm/선택+gather로정의하고dense Layer0 계산을중복가산하지않는다.

Gate B8개는full audit, 나머지는가벼운shape/cache/prefix audit다. instrumentation tag를기록하고timing을분리해해석한다. 모든시간은동기화·checks·CPU전송·profiler의부하가있는파일럿진단이며안정적인TTFT 우위가아니다. GPU peak allocated/reserved를각route reset후기록하지만reserved는같은프로세스의allocator cache를포함한다.

최종Feature Coverage는공통full-N FP32 calibrated cosine affinity에서 `F(S)=sum_i max_{j in S} A_ij`로CPU에서계산한다. query별ERC-final64를사용하며임시Coverage32와혼동하지않는다. 서로다른N 비교에는F/N과Full 대비ratio도기록한다. image별C−L Coverage/Accuracy 관계는descriptive이며인과·유의성·알고리즘선정을주장하지않는다.
