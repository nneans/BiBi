# 비비수산 AI 리뷰 분석

상품(`product_id`)마다 고객이 **무엇을 칭찬하고 무엇에 불만인지, 왜 그런지**를 근거 문장과 함께 찾아 집계합니다. 상품은 ID 하나씩 따로 분석하고 서로 합치지 않습니다. 보고서는 '맛있어요' 같은 일반 칭찬보다 '치즈가 안 늘어난다', '기름에 튀겨야 맛있다' 같은 구체적인 불만·의견을 앞에 둡니다.

## 폴더 지도 (흐름 순서)

| 단계 | 폴더 | 하는 일 |
| --- | --- | --- |
| 1 | [1_ingest](1_ingest/README.md) | 받은 리뷰 CSV → 리뷰 DB(PostgreSQL, 원본 장부): 리뷰글번호로 중복 제거, 수정·블라인드 반영, 받은 CSV 보관(`ingest.py add`), 백업 |
| 2 | [2_preprocessing](2_preprocessing/README.md) | 리뷰 DB → 명령 하나(`preprocess.py`)로 정상 리뷰, 상품 ID별 대표 이름, 분석용 텍스트(줄바꿈·반복 문자 정리), 구매 번호(원 리뷰 + 한달사용 후기 = 한 구매), 바뀐 리뷰 목록 |
| 3 | [3_사후학습](3_사후학습/README.md) | KcELECTRA를 우리 리뷰 49.5만 건으로 추가 학습(식품 리뷰 적응), 1에폭 결과 `frozen-encoder/` 사용 |
| 4 | [4_정답데이터](4_정답데이터/README.md) | 사람이 확정한 정답 300리뷰·666의견, 추가 단위 정답 800개, 같은 뜻 판정 148쌍 |
| 5 | [5_감성분류](5_감성분류/README.md) | Kiwi로 문장·절 단위 나누기, 단위 감성 분류기 교차검증 |
| 6 | [6_이유분석](6_이유분석/README.md) | 30개 상품 분석: 감성 판정 → KURE-v1 임베딩 → BERTopic 묶기·합치기 → 상품별 보고서 |
| 7 | [7_보고](7_보고/README.md) | 기초 집계, 흐름도 그림, 대시보드 인계, [리뷰 한 건 따라가기 노트북](7_보고/노트북/리뷰_한_건_따라가기.ipynb) |

코드와 설정은 단계 폴더와 따로 있습니다: `src/bibi_reviews/`(공용 코드, 폴더 위치는 [paths.py](src/bibi_reviews/paths.py) 한 곳에서 관리), `configs/`, `scripts/`, `tests/`, `docs/`.

## 분석 흐름 (v3, 2026-10-26)

```
원문 리뷰 ─ ① 전처리(normalize_text) ─ ② Kiwi 문장·절 단위 ─ ③ KcELECTRA 단위 감성(긍정/부정/평가 아님, 확신도 0.8 미만은 재확인)
         ─ ④ KURE-v1 임베딩 ─ ⑤ BERTopic(UMAP → HDBSCAN → c-TF-IDF) → 같은 뜻 묶음 합치기 → 묶음 중심 기준 소속 정리
         ─ ⑥ 상품별 보고서: 구체적 불만과 같은 주제 칭찬 짝짓기, 최근 3개월 vs 이전(리뷰 100건당)
```

- 감성 분류기: 사후학습 인코더에서 출발해 정답 단위 725개 + 추가 단위 799개로 미세조정(정답 교차검증 정확도 0.862, 실제 상품 단위의 '부정' 정확도 0.76).
- 결과: [요약](6_이유분석/results-v3/summary.md) · [상품별 구체적 불만·칭찬](6_이유분석/results-v3/insights.md) · [리뷰별 판정](6_이유분석/results-v3/review-samples.md)
- 그림: [분석 흐름](7_보고/그림/pipeline-flow.png) · [전처리 흐름](7_보고/그림/preprocess-flow.png) · [고등어 묶음 지도](6_이유분석/results-v3/300868056/topics-map-NEG.png)

## 팀원 시작하기

이 저장소에는 **데이터와 학습된 모델까지 전부** 들어 있습니다(private, 고객 리뷰 원본 포함이므로 외부 공유 금지). 큰 파일은 Git LFS로 관리합니다(약 2GB).

```powershell
git lfs install
git clone https://github.com/nneans/BiBi.git
cd BiBi
git lfs pull                                    # 모델·데이터 실제 파일 받기 (clone이 자동으로 받았으면 생략 가능)

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121    # NVIDIA GPU용 torch
pip install -r requirements.txt
pip install -e .                                # src/bibi_reviews 패키지

python scripts/setup_models.py                  # 외부 모델 2개(KURE-v1, KcELECTRA 원본)를 고정 버전으로 받기
python -m pytest tests 3_사후학습/test_stage1_core.py    # 69 passed, 2 skipped 이면 준비 완료 (DB 서버를 켜 두면 71 passed)
```

- 필요 환경: Python 3.11, NVIDIA GPU(CUDA). 분석 PC는 Windows 11 + RTX 4080 SUPER입니다.
- 바로 볼 것: [리뷰 한 건 따라가기 노트북](7_보고/노트북/리뷰_한_건_따라가기.ipynb), [상품별 구체적 불만·칭찬](6_이유분석/results-v3/insights.md)
- 새 리뷰를 넣는 사람만 DB(PostgreSQL)가 필요합니다. 분석(3단계부터)은 `2_preprocessing/output/`의 parquet을 바로 읽어서 DB 없이도 돌아갑니다. DB가 필요하면 [1_ingest](1_ingest/README.md)의 백업 파일로 복원하세요.

| 저장소에 들어 있는 큰 파일 (LFS) | 위치 | 크기 |
| --- | --- | ---: |
| 원본 리뷰 CSV (64.9만 행) | `1_ingest/raw/` | 225MB |
| 정리된 리뷰 parquet 4개 | `2_preprocessing/output/` | 192MB |
| DB 백업 | `1_ingest/db/backups/` | 93MB |
| 사후학습 KcELECTRA 인코더 | `3_사후학습/frozen-encoder/` | 509MB |
| 단위 감성 분류기 v3 | `6_이유분석/models/unit-sentiment-v3/` | 511MB |
| 30개 상품 단위·묶음 결과 | `6_이유분석/results-v3/<상품ID>/` | 약 200MB |

외부 모델 `beomi/KcELECTRA-base-v2022`(사후학습 출발점)와 `nlpai-lab/KURE-v1`(임베딩)은 저장소에 넣지 않고 `setup_models.py`가 Hugging Face에서 받습니다. 위치와 버전은 [paths.py](src/bibi_reviews/paths.py)에 있습니다.

## 자주 쓰는 명령

```powershell
pip install -e .                                                 # src/bibi_reviews 패키지 설치
python 1_ingest/ingest.py add <새.csv>                           # 새로 받은 리뷰 CSV 넣기 (DB 반영: 리뷰글번호로 중복 제거, 최신 버전 유지 → 전처리까지)
python 2_preprocessing/preprocess.py                              # DB에서 전처리 다시 만들기 (--from-raw: DB 없이 원본 CSV에서)
powershell -File scripts/postgres_server.ps1 start                # PC를 켠 뒤 DB 서버 켜기
bash 6_이유분석/run_all.sh                                        # 30개 상품 분석 (products.json)
python 6_이유분석/summarize_all.py v3                             # 요약·인사이트 보고서
python -m pytest tests 3_사후학습/test_stage1_core.py             # 69 passed, 2 skipped (DB 서버가 켜져 있으면 71 passed, 2026-10-28)
```

**분석 PC 주의:** 분석 PC의 i9-14900KF는 성능 코어가 불안정해서(2026-10-25 확인) `bibi_reviews`를 불러오면 효율 코어에만 고정됩니다([cpu.py](src/bibi_reviews/cpu.py)). i9-13900/14900이 아닌 PC에서는 자동으로 꺼지고, `BIBI_ALL_CORES=1`로 끌 수도 있습니다.

정리 기록: [docs/cleanup-2026-10-23.md](docs/cleanup-2026-10-23.md)
