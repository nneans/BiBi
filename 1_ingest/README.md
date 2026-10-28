# 1단계 · 수집·저장 (1_ingest)

스마트스토어에서 받은 리뷰 CSV를 **리뷰 DB(원본 장부)**에 넣습니다. 새 리뷰도, 수정·블라인드도 모두 여기서 한 번만 판정하고, 뒤 단계(2_preprocessing → 분석 → 대시보드)는 이 DB를 읽습니다.

```powershell
powershell -File scripts/postgres_server.ps1 start   # DB 서버 켜기 (PC를 켤 때마다)
python 1_ingest/ingest.py add <새.csv>               # 새 CSV: 검사 → DB 반영 → raw/에 보관 → 목록 기록 → 2_preprocessing 다시 만들기
python 1_ingest/check_db.py                          # DB가 받은 CSV와 정확히 같은지 확인 → verification/db-vs-raw.json
```

## 새 리뷰를 넣을 때

새 기간만 담긴 CSV든, 처음부터 다시 뽑은 전체 CSV든 같은 명령입니다. 나중에 기업이 대시보드에서 올리는 웹 업로드도 같은 DB 반영 코드(`backbone.ReviewStore.import_csv`)를 씁니다.

**중복은 날짜가 아니라 리뷰글번호로 막습니다.** 같은 날에도 리뷰가 여럿이고, 옛 리뷰가 나중에 수정·블라인드될 수 있어서입니다.

| 같은 리뷰글번호가 다시 오면 | 처리 |
| --- | --- |
| 내용이 똑같음 (겹치는 기간) | 건너뜀 |
| 최종수정일(없으면 등록일)이 더 최근 | 새 버전으로 교체, 이전 버전은 `review_history`에 보관 |
| 더 오래된 버전 | 무시 |
| 시각은 같은데 내용이 다름 | 나중에 받은 CSV를 따름 (건수 기록) |
| 블라인드로 바뀜 | 본문을 지우고 분석에서 빠짐 |
| 새 CSV에 없음 | **지우지 않음** (새 기간 CSV에는 옛 리뷰가 없는 게 당연함) |

- 형식이 다르면(컬럼, 별점, 날짜, 리뷰글번호 중복) DB를 건드리기 전에 멈춥니다.
- 같은 파일을 두 번 넣으면 거부합니다(파일 해시).
- 받은 CSV는 `raw/`에 그대로 보관하고 `raw/deliveries.json`에 파일 이름·해시·행 수·리뷰 기간·DB 반영 건수(신규·수정·변화 없음·옛 버전·블라인드)를 남깁니다.
- 2026-10-28 확인(`check_db.py`): DB의 리뷰 649,524건이 받은 CSV와 모든 컬럼(행 해시 포함)까지 같습니다. CSV는 `ingest.py`나 웹 업로드로만 DB에 들어갑니다(예전 `python -m bibi_reviews.backbone import` 명령은 목록 기록을 건너뛰어서 없앴습니다).

## 리뷰 DB: PostgreSQL 17

| 항목 | 값 |
| --- | --- |
| 접속 주소 | `postgresql+psycopg://bibi@127.0.0.1:5432/bibi_reviews` (`configs/review_pipeline/pipeline.json`) |
| 서버 위치 | `%USERPROFILE%\.bibi-postgres` (무설치판. 프로그램 `pgsql\`, 데이터 `data\`, 로그 `server.log`) |
| 같은 서버의 다른 DB | `bibi_labeling`, `bibi_opinion_pilot` (예전 시범, 리뷰 DB와 분리) |
| 비밀번호 | 프로젝트에 없음. Windows 사용자 전용 `%APPDATA%\postgresql\pgpass.conf`에서 자동으로 읽음 |
| 접속 범위 | 이 PC 안에서만(127.0.0.1) |
| 다른 PC·서버 | 환경변수 `BIBI_DATABASE_URL`에 주소를 넣으면 코드 수정 없이 바뀜 |

```
bibi_reviews
├─ 원본 층      products, reviews, review_history, upload_jobs, upload_reviews
├─ 분석 결과 층  analysis_*, opinion_*, model_versions
├─ 사람 정답 층  opinions, human_corrections, reason_candidates
└─ 분류 기준     ontology_*
```

```powershell
powershell -File scripts/postgres_server.ps1 status   # 상태
powershell -File scripts/postgres_server.ps1 backup   # db/backups/에 백업 파일(.dump) 생성
powershell -File scripts/postgres_server.ps1 stop     # 끄기
```

백업에서 되살리기: `pg_restore -h 127.0.0.1 -U bibi -d <새 DB 이름> --no-owner <백업 파일>`

## 폴더

| 경로 | 내용 |
| --- | --- |
| `ingest.py` | 새 CSV 넣기 (`add`) |
| `check_db.py` | DB 검증: 받은 CSV 전부를 같은 반영 코드로 임시 DB에 넣은 결과와 실제 DB를 컬럼마다 비교, 끝나지 않은 업로드 작업 표시 |
| [raw/](raw/README.md) | 받은 CSV들(수정 금지), `deliveries.json`, `audit/`(원본을 처음 점검한 기록) |
| `db/backups/` | PostgreSQL 백업 |
| `db/uploads/` | 대시보드 웹 업로드로 들어온 CSV (자동 생성, 저장소에 올리지 않음) |
| `verification/db-vs-raw.json` | `check_db.py`의 최근 결과 (예전 2026-10-16·10-18 적재 기록은 만든 코드가 없어 로컬 `archive/ingest/`로 옮김) |

- DB가 없는 사람도 분석할 수 있습니다: 2_preprocessing의 결과 파일이 저장소에 있고, `python 2_preprocessing/preprocess.py --from-raw`는 raw/의 CSV를 같은 반영 코드로 임시 SQLite DB에 넣어 다시 만듭니다.
- 아직 연결하지 않은 것: 대시보드 웹 업로드(`api.py`)는 DB 반영까지만 하고 `deliveries.json` 기록과 2_preprocessing은 부르지 않습니다. 대시보드를 새 흐름에 붙일 때 같이 연결합니다.
- 코드: `src/bibi_reviews/backbone.py`(검사·증분 반영), `src/bibi_reviews/api.py`(웹 업로드·인증)
