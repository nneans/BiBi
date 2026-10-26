# 1단계 · 전처리 / 결과

상품 ID 기준으로 묶고, 상품명을 대표 이름 하나로 통일하고, 블라인드 리뷰를 뺀 분석용 데이터입니다.

| 경로 | 내용 |
| --- | --- |
| `processed/reviews_normal.parquet` | 정상 리뷰 642,172건 |
| `processed/canonical_products/` | 대표 상품명 적용본, ID별 대표 이름 매핑, 기존 이름 → 대표 이름 대응표 |
| `product_name_mapping/` | 상품명 정리 근거 표와 검증 |
| `product_distribution.*` | 상품별 리뷰 분포 |

- 상품 식별·조인은 항상 `product_id`로 합니다. 대표 이름은 화면 표시용입니다.
- 코드: `src/bibi_reviews/canonicalize_products.py`
- 다시 실행: `python -m bibi_reviews.canonicalize_products --output-dir <새 폴더>`
