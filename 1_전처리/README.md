# 1단계 · 전처리

원본 CSV를 분석에 쓸 수 있는 형태로 정리합니다. 원본은 고치지 않습니다.

| 폴더 | 내용 |
| --- | --- |
| [원본/](원본/README.md) | 원본 CSV 649,524행, 데이터 진단([data-audit.md](원본/data-audit.md)), 품질 점검 |
| [결과/](결과/README.md) | 정상 리뷰 642,172건, 상품 ID별 대표 이름 적용본(`processed/canonical_products/`) |

- 정상 리뷰만 사용합니다(블라인드 제외). 쓰는 컬럼은 상품 ID·상품명·별점·작성일·리뷰 본문 등 일부입니다.
- 상품 ID → 이름은 하나로 대응하지만, 같은 이름을 쓰는 다른 ID가 있어 항상 `product_id`로 묶습니다.
- 본문의 `&hellip;` 같은 HTML 기호는 분석 직전 `normalize_text`에서 원래 문자로 되돌립니다.
- 흐름 그림: [preprocess-flow.png](../7_보고/그림/preprocess-flow.png)
