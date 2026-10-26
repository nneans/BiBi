# 7단계 · 보고 / 대시보드

화면 담당자에게 넘기는 문서입니다. 화면 구현은 다른 담당자가 진행합니다.

| 문서 | 내용 |
| --- | --- |
| [review-pipeline-v2-api.md](review-pipeline-v2-api.md) | `/api/v2` 엔드포인트, 화면 표시 규칙, 실제 응답 예시(의견 트리·근거 리뷰 포함) |
| [dashboard-handoff.md](dashboard-handoff.md) | 화면에 담을 정보와 쉬운 표현 기준 |

- 서버 실행: `BIBI_PASSWORD=<12자 이상> uvicorn --factory bibi_reviews.service.app:app_from_env --host 127.0.0.1 --port 8000`
- 운영 모델이 아직 승인되지 않아, AI 통계 API는 "모델 없음"(409)을 돌려줍니다. CSV 기반 통계는 바로 나옵니다.
