# 2단계 · DB에 저장하고 변경 사항 확인

리뷰 DB와, 새 CSV가 들어왔을 때 **형식 확인(B) → 기존 DB와 비교 후 신규·수정만 반영(C)** 하는 과정의 결과입니다.

## 운영 DB: PostgreSQL 17 (2026-10-18부터)

| 항목 | 값 |
| --- | --- |
| 접속 주소 | `postgresql+psycopg://bibi@127.0.0.1:5432/bibi_reviews` (`configs/review_pipeline/pipeline.json`) |
| 서버 위치 | `%USERPROFILE%\.bibi-postgres` (무설치판. 프로그램 `pgsql\`, 데이터 `data\`, 로그 `server.log`) |
| 같은 서버의 다른 DB | `bibi_labeling`(Groq 호출 장부·캐시), `bibi_opinion_pilot`(AI 후보 의견 시범) — 운영 DB와 분리 |
| 비밀번호 | 프로젝트에 없음. Windows 사용자 전용 `%APPDATA%\postgresql\pgpass.conf`에서 자동으로 읽음 |
| 접속 범위 | 이 PC 안에서만(127.0.0.1) |
| 다른 PC·사내 서버 | 환경변수 `BIBI_DATABASE_URL`에 주소를 넣으면 코드 수정 없이 바뀜 |

DB 안에는 성격이 다른 자료를 표로 나눠 둡니다. 원본은 덮어쓰지 않습니다.

```
bibi_reviews
├─ 원본 층      products, reviews, review_history, upload_jobs, upload_reviews
├─ 분석 결과 층  analysis_* (모델 판정), opinion_predictions·opinion_evidence·opinion_reasons (의견·근거·이유), model_versions
├─ 사람 정답 층  opinions (검수 의견), human_corrections (화면에서 고친 내용), reason_candidates (새 이유 후보)
└─ 분류 기준     ontology_* (분류 기준 버전별 기록)
```

**서버 켜기·끄기·백업** (PC를 재부팅하면 다시 켜야 합니다. Windows 서비스로 등록하지 않았습니다):

```powershell
powershell -File scripts/postgres_server.ps1 start    # 켜기
powershell -File scripts/postgres_server.ps1 status   # 상태
powershell -File scripts/postgres_server.ps1 backup   # db/backups/에 백업 파일(.dump) 생성
powershell -File scripts/postgres_server.ps1 stop     # 끄기
```

백업에서 되살리기: `pg_restore -h 127.0.0.1 -U bibi -d <새 DB 이름> --no-owner <백업 파일>`

## 폴더 내용

| 경로 | 내용 |
| --- | --- |
| `db/backups/` | PostgreSQL 백업. 첫 백업 `bibi_reviews_20261018-215430.dump`(약 89MB, 25개 테이블) |
| `db/uploads/` | 웹 업로드로 들어온 CSV가 저장되는 곳(업로드 시 자동 생성) |
| `verification/2026-10-18-postgres-migration.json` | 이전 검증: 7개 테이블 행 수·전체 내용 해시 일치 |
| `verification/2026-10-16/` | 최초 적재·재반영 검증(동일 CSV 재반영 시 신규·수정 0건) |

- 판정: 신규 / 수정(텍스트 변경 → 재분석) / 별점만 변경(분석 재사용) / 미변경 / 블라인드(본문 삭제)
- 코드: `src/bibi_reviews/backbone.py`(검증·증분 반영), `src/bibi_reviews/api.py`(업로드·인증)
- CSV 직접 반영: `python -m bibi_reviews.backbone import --csv <파일.csv>`
- 웹 업로드: `BIBI_PASSWORD=... uvicorn --factory bibi_reviews.service.app:app_from_env` 후 `/api/uploads`
