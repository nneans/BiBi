# 7단계 · 보고 / 집계

상품별 리뷰 수·별점·월별 변화 같은 통계와 그래프입니다.

| 경로 | 내용 |
| --- | --- |
| `basic_statistics/2026-10-16/` | 539상품 별점 분포, 월별·계절별 통계(한글 CSV 포함) |
| `figures/` | 시각화 PNG·HTML(기초통계, 상품명 정리 흐름 등) |

- 지금 집계는 **CSV 원자료(별점·건수)** 기준입니다. 속성·이유별 집계는 대시보드 API가 DB에서 바로 계산합니다(`../대시보드/`).
- 코드: `scripts/analyze_basic_statistics.py`, `src/bibi_reviews/service/analytics.py`, `src/bibi_reviews/opinion_graph/queries.py`
