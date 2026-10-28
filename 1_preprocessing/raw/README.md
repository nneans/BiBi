# 1단계 · 전처리 / raw (원본)

네이버 스마트스토어에서 내려받은 리뷰 원본들과, 원본을 처음 점검한 기록입니다. **원본 CSV는 내용을 수정하지 않습니다**(2026-10-28 폴더 이름만 `1_전처리/원본` → `1_preprocessing/raw`로 바꿈, 파일 이름은 그대로).

| 경로 | 내용 |
| --- | --- |
| `리뷰데이터(2025.01.01~2026.10.13).csv` | 원본 649,524행(정상 642,172 · 블라인드 7,352 · 539상품), Git LFS |
| `deliveries.json` | 받은 CSV 목록(받은 순서, 파일 해시, 행 수, 리뷰 기간). 새 CSV는 `python 1_preprocessing/preprocess.py add <새.csv>`로 넣으면 여기에 복사되고 기록됨 |
| `audit/data-audit.md` | 컬럼 의미·건수·결측 등 데이터 진단 설명 |
| `audit/profile/` | 데이터 진단 결과 |
| `audit/data_quality/` | 품질 점검: 길이 분포, 중복 텍스트, 별점과 부정 표현의 관계 |

- `audit/data_quality/` 다시 만들기: `python -m bibi_reviews.review_pipeline.data_quality` (DB 읽기 전용)
