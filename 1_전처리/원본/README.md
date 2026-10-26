# 1단계 · 전처리 / 원본

네이버 스마트스토어에서 내려받은 리뷰 원본과, 원본을 처음 점검한 결과입니다. **원본 CSV는 수정하거나 옮기지 않습니다.**

| 경로 | 내용 |
| --- | --- |
| `리뷰데이터(2025.01.01~2026.10.13).csv` | 원본 649,524행(정상 642,172 · 블라인드 7,352 · 539상품) |
| `profile/` | 컬럼 의미·건수 등 데이터 진단 결과([설명](data-audit.md)) |
| `data_quality/` | 품질 점검: 길이 분포, 중복 텍스트, 별점과 부정 표현의 관계 |

- 코드: `src/bibi_reviews/review_pipeline/data_quality.py`
- 다시 실행: `python -m bibi_reviews.review_pipeline.data_quality` (DB 읽기 전용)
