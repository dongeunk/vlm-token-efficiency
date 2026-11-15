# 원격 최신본 기반 공개 후보 재구성 보고서

## 위치와 기준

- E드라이브는 정상 NTFS 고정 디스크, 생성 전 가용약851GB였다. 별도 새 공개 worktree를 만들었으며 기존 프로젝트/디스크를 이동하지 않았다.
- 기준: origin/main `66bbfaf44770b99473fbe94fdde2554f5c94f368`. detached HEAD이므로 원래 main branch를 checkout/reset/rebase하지 않았다.
- C 작업 폴더의 로컬 변경, 원래 파일과201개 준비 후보는 그대로 보존했다. 새 구성은 이전201개 안을 대체하는 별도 공개 검토안이다.
- Commit/Push와 실제 두 worktree의 index staging은 하지 않았다. Git diff 확인에만 별도 임시 index를 사용한다.

## 1. 기존 원격에서 보존한 자료

원격72개 경로를 모두 유지했다. LICENSE, NOTICE, .gitignore, .gitattributes, 기존 문서·공개 manifests·코드·tests·1차 집계/그림·vlm.py는 변경하지 않는다. README만 새 연구 안내를 앞에 추가하며 원래 본문을 정확히 남긴다. 기존 원격의 허용/비공개 경계도 보존한다.

## 2. 새로 추가할 자료

- 후속1차80이미지800결과 및 Floor/No-Floor320결과의 보고서·소형 지표/CI. 기존30이미지 공개 자료를 대체하지 않는다.
- 2차 S0–S4-A 보고서·경로 비식별 설정·ID/digest-only 코호트·본 평가200개 이미지 count vectors·결과 CSV/기존CI/예산·Coverage·시간 집계·photo-free 그림2개.
- 독자 작성 CPU evaluation의5개 파일과 집계 전용 공개 wrapper만 포함한다. 연구 코드의 AST 본문은 그대로 두고 새 공개 복사본에 라이선스 주석만 추가했다.
- 원시 모델 응답은 기존 원격의 비공개 정책을 유지해 제외한다. 현재 도구는 Parser 재판정 대신 저장 count를 통한 지표 검산을 제공한다. 원래 C의5920응답 재현 도구는 바꾸지 않았다.

## 3. HOLD10개와 추가 보류

[FILE_LIST.md](FILE_LIST.md)에10개 전체 경로를 기록했다. 권리 HOLD인 src/pruning/s2_style_allocation.py, dependency-HOLD9개가 전부 후보에서 제외된다. 기존 mixed-provenance test를 포함한 원격11개 HOLD 전체도 없다. 원본 파일들은 C 작업 폴더에 그대로 있다.

또한 S²Prune allocator/reference, Smol mapping/cell/ERC/offline/Pruning runtime/GPU runner와 관련 test의 검토가 남은 경로를 보수적으로 보류했다. 이는 모든 파일이 외부 코드를 차용했다는 판정이 아니다. 공개되는 CPU 평가 subset에는 그 helper가 필요하지 않다. 원본 helper를 stub/다운로드/재명명으로 대체하지 않았다.

## 4. MMTok 고지3개 처리

coverage.py, spatial_floor_coverage.py, matched_allocation.py는 원격 버전을 그대로 유지한다. 기존 Zoom copyright와 SPDX 고지가 보존되고 로컬 연구 본문과AST가 일치함을 확인했다. 따라서 무고지 로컬 복사본으로 덮어쓸 필요가 없다.

pinned upstream revision8e2686836ed5bbe05d064f1d1c20e50c73c70b94의 selector 저작권/Apache-2.0 표기와 비교했다. LICENSE는 선행빈줄·부록 copyright 예시만 달랐고 작동 조항은 일치했다. 기존 LICENSE/NOTICE를 수정하지 않으며, Zoom 고지는NOTICE와3개 소스에 그대로 남아 있다. 원본 upstream source는 배포하지 않는다.

## 5. 파일 수·용량과 diff

전체 후보 **138개 경로 = 기존72 + 신규66개**. 예상 변경은 **README1개 수정 + 신규66개**, 삭제0개다. 경로 전체는 FILE_LIST.md, 파일 hash는 UPLOAD_MANIFEST.json에 있다. 최종 후보 파일 합계는 **3,116,539 bytes (3.117 MB)**이며, Git diff·최대 파일·검증 결과는 VALIDATION.json에 기록한다. 원격 전송량은 미측정이다.

## 6. 남은 한계

S²Prune 원본과 code-informed 참조 구현의 재배포 권한은 계속 HOLD다. 모델·사진·annotation·raw tensor/응답도 공개하지 않는다. 새 데이터 view는 원본 실행 manifest/lock의 hash 동일 대체물이 아니다. Full GPU 및 전체 역사적 suite 재실행은 이번 공개본으로 제공하지 않는다.

기존1차 Apache-2.0 grant와NOTICE를 보존하고 새 독자 CPU/문서 기여는 기존 프로젝트 original-contribution 방식으로 분류했다. 외부 원본의 권한을 확대하거나 불명확한 코드를 라이선스로 덮지 않았다. 자세한 범위는 [SECOND_STUDY_RELEASE_SCOPE](../SECOND_STUDY_RELEASE_SCOPE.md)를 따른다.

## 7. Commit/Push 가능 여부

최종 VALIDATION이PASS이면 **이 E worktree의 축소·검토된 후보에 한해** 다음 승인 단계에서 commit을 검토할 수 있다. 이전 C 작업 폴더의201개 목록을 그대로 올려서는 안 된다. 실제 공개 직전에는 원격 tip과 후보 diff를 다시 확인하고 non-force push만 사용해야 한다. 현재는 사용자의 지시에 따라 commit/push하지 않고 준비·검증에서 중단한다.
