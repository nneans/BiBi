# 6단계 · 이유 분석

상품 하나의 리뷰를 단위로 나누고 감성을 붙인 뒤, 비슷한 의견끼리 묶어 "무엇이 왜 좋고 나쁜지"를 보여 줍니다.

```powershell
bash 6_이유분석/run_all.sh                       # products.json의 30개 상품 전부(끝난 상품은 건너뜀)
python 6_이유분석/run_product.py <product_id>   # 상품 하나
python 6_이유분석/summarize_all.py v3           # results-v3/summary.md(한 장 요약), insights.md(상품별 구체적 불만·칭찬)
```

- 흐름: `normalize_text` → Kiwi 단위 → 감성 분류기 → 확신도 0.8 미만은 재확인 목록 → KURE-v1 임베딩 → BERTopic(감성별)
- 묶음마다 최근 3개월과 그 이전의 리뷰 100건당 건수를 비교합니다. 근거는 단위 문장 그대로입니다.

- 보고서는 '맛있어요', '배송 빨라요', '그냥 그래요' 같은 일반 묶음을 앞에 두지 않고, 치즈가 안 늘어난다·기름에 튀겨야 맛있다처럼 구체적인 불만·칭찬을 같은 주제끼리 짝지어 보여 줍니다(`insights.md`).

| 결과 파일 (`results-v3/<상품ID>/`) | 내용 |
| --- | --- |
| `report.json` | 묶음별 키워드·대표 문장·건수·기간 비교, 점검 결과 |
| `units.jsonl` | 단위마다 감성·확신도 |
| `review-queue.jsonl` | 사람이 다시 볼 단위(확신도 0.8 미만) |
| `groups-NEG.json`, `groups-POS.json` | 묶음 단계 중간 저장(PC가 꺼져도 이어서 진행) |

- 감성 분류기는 `models/unit-sentiment-v3/`에 처음 실행 때 자동으로 학습해 둡니다(정답 300건 단위 + 추가 단위 799개, 사람 확인 완료).
- 이 PC는 성능 코어가 불안정해서(2026-10-25 확인) 분석 프로그램이 스스로 효율 코어에만 고정됩니다(`src/bibi_reviews/cpu.py`). `run_all.sh`는 그래도 꺼지면 최대 5번 다시 띄웁니다.
