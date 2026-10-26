# 대시보드 API 인계 — `/api/v2`

> 2026-10-23 폴더 재편 전 문서입니다. 안의 경로(01_~08_ 등)와 방식은 예전 기준이고, 지금 흐름은 맨 위 README.md를 보세요.

대상: 대시보드 화면을 구현하는 프론트엔드 담당자. 기준일 2026-10-17.

## 접속

- 실행: `BIBI_PASSWORD=<12자 이상> uvicorn --factory bibi_reviews.service.app:app_from_env`
- 인증: 모든 `/api/*`는 HTTP Basic(사용자 `bibi` 또는 `BIBI_USERNAME`). 실패 시 401.
- 스키마: `GET /api/schema`(인증 필요)가 OpenAPI JSON을 돌려준다. 공개 `/docs`는 꺼져 있다.
- 기간 파라미터 `start`, `end`는 `YYYY-MM-DD` 형식이다. `end`는 **포함하지 않으며**, 시간대는 Asia/Seoul이다.

## 화면에 반드시 반영할 규칙

1. **AI 통계는 모델 예측이다.** 모든 AI 응답의 `meta`에 `result_kind: "model_prediction"`, `human_validated: false`가 들어 있다. `production_validated`가 `false`이면 화면에 “검증 전 결과”를 표시한다. 현재는 활성 모델이 없어 AI 엔드포인트가 **409**를 돌려준다. 이때 CSV 기반 정보(개요·리뷰 원문·업로드 현황)만 보여 준다.
2. **분모를 함께 표시한다.**
   - `positive_share_of_analyzed`·`negative_share_of_analyzed`의 분모는 기간 내 분석 완료 리뷰 수(`coverage.analyzed`)다.
   - `negative_share_of_mentioned`의 분모는 그 속성에 긍정·부정 예측이 있는 리뷰 수다.
   - 한 리뷰에 여러 속성이 있을 수 있어 속성별 비율의 합은 100%가 아니다.
3. **`null`은 0이 아니다.** `model_supported`가 false인 칸은 모델이 그 감성을 학습하지 못한 것이다. “데이터 없음/평가 불가”로 표시한다. 예측이 없다는 것도 “고객 의견 없음”의 증거가 아니다.
4. **한 화면에는 한 모델 버전만 쓴다.** 기간 비교 차트에서 서로 다른 `model_version`의 결과를 섞지 않는다.
5. **세부 이유(`reason_candidates`)는 모두 미검증 후보다**(`reason_status: "unverified_candidate"`). 집계 지표로 쓰지 말고 근거 옆 참고 태그로만 표시한다.
6. 블라인드로 바뀐 리뷰는 모든 응답에서 빠진다. 단건 조회 시 404다.

## 엔드포인트

| 메서드·경로 | 용도 | 주요 파라미터 |
| --- | --- | --- |
| `GET /api/summary` (기존) | 전체 리뷰 수·평균 별점 | |
| `GET /api/v2/products` | 상품 목록(리뷰 수·평균 별점) | `q`(상품명 일부 또는 ID) |
| `GET /api/v2/products/{id}/overview` | 상품 기본 현황(CSV 기반) | `start`, `end` |
| `GET /api/v2/products/{id}/aspects` | 속성별 긍정·부정 리뷰 수와 비율, 분석 현황 | `start`, `end`, `model_version` |
| `GET /api/v2/products/{id}/trends` | 기간별 속성·감성 변화 | `granularity=month\|week`, `aspect` |
| `GET /api/v2/products/{id}/emerging` | 최근 증가한 부정 속성 **후보** | `as_of`, `recent_days=28`, `baseline_days=84` |
| `GET /api/v2/products/{id}/reviews` | 근거 문장이 붙은 실제 리뷰 목록 | `aspect`, `sentiment=positive\|negative`, `rating`, `status`, `start`, `end`, `limit≤200`, `offset` |
| `GET /api/v2/reviews/{review_id}` | 리뷰 원문 전체와 예측 | |
| `GET /api/v2/processing/status` | 업로드 성공·실패, 신규·수정·미변경, 분석 대기·완료·실패, 검토 대기, 적용 모델 | |
| `GET /api/v2/review-queue` | 사람 검토가 필요한 리뷰 | `product_id`, `limit`, `offset` |
| `POST /api/v2/reviews/{review_id}/corrections` | 검토자 수정 저장. 예측을 덮어쓰지 않고 별도 정답 자료로 쌓는다 | 본문 아래 참조 |
| `GET /api/v2/models` | 활성 모델·검증 여부·이력 | |
| `POST /api/uploads` (기존) | CSV 업로드 → 202와 `job_id` | multipart `file` |
| `GET /api/uploads/{job_id}` (기존) | 반영 상태와 건수 | |

## 응답 예시

아래 예시는 격리 테스트 환경에서 **테스트 전용 스모크 모델**로 만든 실제 응답이다. 수치는 형식 예시일 뿐이다.

`GET /api/v2/products/7826600725/aspects`

```json
{
  "meta": {"model_version": "review-clf-klue_roberta-2ec97b8a0652", "production_validated": false,
           "test_only_model": true, "result_kind": "model_prediction", "human_validated": false,
           "note": "모델 예측 결과입니다. 사람 검수 정답이나 확정 통계가 아닙니다."},
  "product": {"product_id": "7826600725", "product_name": "명태포 제수용 명태 슬라이스 700g+700g 동태전 동태포"},
  "period": {"start": null, "end": null, "end_exclusive": true},
  "coverage": {"normal_reviews": 7372, "analyzed": 7372, "unclassified": 1552, "review_required": 3925, "failed": 0, "pending": 0},
  "aspects": [
    {"aspect": "taste", "positive_reviews": 2849, "negative_reviews": 32, "mixed_reviews": 19, "mentioned_reviews": 2862,
     "positive_share_of_analyzed": 0.3865, "negative_share_of_analyzed": 0.0043, "negative_share_of_mentioned": 0.0112,
     "model_supported": {"positive": true, "negative": true}},
    {"aspect": "delivery", "positive_reviews": 1647, "negative_reviews": null, "mixed_reviews": null, "mentioned_reviews": null,
     "negative_share_of_analyzed": null, "model_supported": {"positive": true, "negative": false}}
  ],
  "denominators": {"share_of_analyzed": "analyzed normal reviews of this product in the period (coverage.analyzed)",
                   "share_of_mentioned": "reviews with a positive or negative prediction for that aspect"},
  "counting": "distinct reviews; shares across aspects do not sum to 100%"
}
```

`GET /api/v2/products/7826600725/reviews?sentiment=negative` 의 항목 하나:

```json
{"review_id": "5082704110", "product_id": "7826600725", "rating": 1, "created_at": "2026-10-13T16:18:25+09:00",
 "review_type": "한달사용", "review_text": "두께랑 사이즈가 너무 들쭉날쭉해서 사용하기 불편했어요",
 "analysis_status": "review_required", "queue_status": "open",
 "predictions": [{"aspect": "quantity", "sentiment": "negative", "score": 0.9284, "threshold": 0.5,
   "evidence_text": "두께랑 사이즈가 너무 들쭉날쭉해서 사용하기 불편했어요", "evidence_start": 0, "evidence_end": 29,
   "reason_candidates": ["whole_size", "thickness"], "reason_status": "unverified_candidate"}]}
```

- `evidence_start`/`evidence_end`는 `review_text`의 Unicode 문자 위치이고 끝은 포함하지 않는다. 즉 `review_text.slice(start, end)`가 근거 문장이다. JS의 `slice`는 UTF-16 단위이므로 이모지가 있으면 `Array.from(text)`로 잘라야 한다.
- `score`는 보정된 확률이 아니다. 백분율 “정확도”로 표시하지 않는다.

`POST /api/v2/reviews/{review_id}/corrections` 본문:

```json
{"annotator": "홍길동", "complete": true,
 "opinions": [{"target_scope": "current_product", "statement_type": "evaluation", "aspect": "packaging",
               "sentiment": "negative", "reason_code": "leak_damage", "reason_text": "터져서", "evidence_text": "포장이 터져서 왔어요"}]}
```

- `target_scope`: `current_product | other_product | general_experience | unclear`
- `statement_type`: `evaluation | fact | deferred | uninterpretable`
- `sentiment`: `positive | negative | mixed | neutral | ambiguous | not_applicable`
- `aspect` 코드와 이름은 `configs/review_pipeline/taxonomy_v2.json`을 따른다.
- `annotator`가 비어 있으면 422다. `complete=true`는 리뷰 전체를 확인했다는 뜻이며, 이때만 표시하지 않은 속성이 “의견 없음”으로 학습된다.

## 상태 값

| 값 | 의미 |
| --- | --- |
| `analysis_status=classified` | 하나 이상의 속성·감성이 임계값을 넘음 |
| `unclassified` | 모델이 아무것도 찾지 못함. **불만 없음이 아니다** |
| `review_required` | 임계값 근처 점수가 있어 사람 확인 필요(검토 대기 목록에 포함) |
| `null`(분석 없음) | 분석 대기 또는 실패. `processing/status`의 `pending`·`failed`로 구분 |
| 업로드 `analysis_status` | `completed`, `completed_with_failures`, `pending_no_active_model`(검증된 모델 없음), `model_load_failed`, `failed`, `running` |
| HTTP 409 | 활성 운영 모델 없음. AI 통계를 표시하지 않음 |

## 집계 기준 요약

- 리뷰 단위로 센다. 한 리뷰에서 같은 속성·감성이 여러 번 나와도 1건이다.
- `mixed_reviews`는 같은 속성에 긍정과 부정이 함께 예측된 리뷰다.
- `emerging`은 최근 28일 부정 비율이 직전 84일의 2배 이상이고 최근 건수가 5건 이상일 때 `increase_candidate=true`다. 확인 전 신호이므로 리뷰 목록과 함께 보여 준다.
- `overview`는 업로드된 CSV 자료만 쓴다(`source: "uploaded CSV data (not AI)"`).

## 의견 그래프 API (온톨로지 v1, 2026-10-17 추가)

기존 `/api/v2` 엔드포인트는 그대로 동작합니다. 아래 엔드포인트는 같은 인증을 쓰며, **개별 의견(Opinion)** 단위로 "상품 → 속성 묶음 → 속성 → 감성 → 세부 이유 → 실제 근거"를 제공합니다. 설계는 [프로젝트 개요 4장](../08_실무활용/project-overview.md#4-온톨로지와-의견-관계-구조)에 있습니다.

### 공통 파라미터와 규칙

- `tier`(기본 `prediction`)
  - `prediction`: 활성 운영 모델의 예측입니다. 활성 모델이 없으면 기존 API처럼 409를 반환합니다.
  - `gold`: 사람 승인 의견입니다.
  - `silver`: AI 후보 의견입니다(검수 전).
  - 한 응답은 하나의 tier만 사용하며 합산하지 않습니다. `meta.label_tier`, `meta.human_validated`, `meta.note`를 화면에 표시해 주세요.
- `source`: silver/gold의 출처 필터입니다(예: `assistant_readthrough`, `local_llm_pilot:qwen3.5:9b`). silver는 출처별로 따로 조회하는 것을 권장합니다.
- `model_version`: 생략하면 활성 모델입니다. 기간 비교는 같은 모델·같은 `meta.ontology_version`에서만 합니다.
- **리뷰 수(`reviews`)와 의견 수(`opinions`)를 구분해 표시**해 주세요. 한 리뷰는 여러 의견을 가집니다. 묶음 합계는 자식 합이 아니라 고유 리뷰 수입니다.
- `reasons`에는 그 속성 소유의 이유만 들어 있습니다. `cross_aspect_reason_hints`는 다른 속성의 이유 단서입니다(예: 품질 부정 의견 안의 "너무 두껍거나 얇음"). 합계에 더하지 말고 "관련 단서"로만 보여 주세요.
- 이유의 `confirmed_reviews`는 사람 승인(gold)에서만 생깁니다. `candidate_reviews`는 AI·어휘 단서 후보입니다. **확정 통계와 후보를 시각적으로 구분**해 주세요.
- `without_specific_reason_reviews`: 해당 감성은 있으나 세부 이유가 연결되지 않은 리뷰 수입니다.

### 엔드포인트

| 메서드·경로 | 용도 |
| --- | --- |
| `GET /api/v2/ontology` | 속성 묶음·속성·이유 트리, 극성별 표시명, 버전·해시 |
| `GET /api/v2/products/{id}/opinion-tree?tier&source&model_version&start&end` | 계층형 통계(위 규칙) |
| `GET /api/v2/products/{id}/opinions?tier&aspect&sentiment&reason&target_scope&start&end&limit&offset` | 의견 인스턴스 + 원문 + 근거 위치 + 이유. 세부 이유 클릭 시 사용 |
| `GET /api/v2/products/{id}/reasons?tier&sentiment=negative&min_reviews` | 반복 불만(또는 칭찬) 이유 순위 |
| `GET /api/v2/products/{id}/opinion-cooccurrence?tier&level=reason\|aspect&min_reviews` | 같은 리뷰에 함께 등장하는 의견 쌍 |
| `GET /api/v2/reasons/{code}?tier&product_id&sentiment&limit` | 이유의 연관 속성(설계상 vs 실제 분포)과 근거 리뷰 추적 |
| `GET /api/v2/reason-candidates?product_id&status=pending_review` | 체계 밖 이유 후보(검토 대기). 자동 등록되지 않음 |
| `GET /api/v2/opinion-graph/status` | tier·출처·모델별 의견 수, 온톨로지 버전, 후보 수 |

### 응답 예시 (실데이터 격리 실행, 검증되지 않은 테스트용 모델)

`05_분석결과저장/e2e_isolated/20261017-220844/e2e_log.json`에서 발췌했습니다. 수치는 파이프라인 동작 확인용이며 품질을 뜻하지 않습니다.

```json
{"meta":{"label_tier":"prediction","model_version":"review-clf-klue_roberta-2ec97b8a0652","ontology_version":"bibi-opinion-ontology-v1-draft",
         "human_validated":false,"note":"자체 모델 예측입니다. 이유 연결은 어휘 단서 후보이며 확정 이유가 아닙니다."},
 "denominators":{"normal_reviews_in_period":7372,"reviews_with_opinions_in_tier":5406},
 "counts":{"opinions":10000,"reviews":5406},
 "groups":[{"code":"grp_quality","name":"품질·상태","reviews":1401,"aspects":[{"code":"quality","sentiments":{"negative":{
   "reviews":519,"review_required_opinions":402,"without_specific_reason_reviews":221,
   "reasons":[{"reason_code":"bones","name":"가시·뼈 잔존","reviews":202,"confirmed_reviews":0,"candidate_reviews":202}],
   "cross_aspect_reason_hints":[{"reason_code":"thickness","name":"너무 두껍거나 얇음","reviews":79,"reason_primary_aspect":"quantity"}]}}}]}]}
```

`/opinions?aspect=quality&sentiment=negative&reason=bones`의 한 항목은 다음과 같습니다. `review_text[start_offset:end_offset] == evidence_text`(Unicode 코드 포인트, end 미포함)로 강조 표시하면 됩니다.

```json
{"opinion_id":"p:ac28c258c37e49ebc53e0d181ba76fa1","review_id":"5082712982","aspect":"quality","sentiment":"negative","status":"review_required",
 "evidence":[{"start_offset":79,"end_offset":90,"evidence_text":"생각보다 가시가 있네","evidence_kind":"sentence"}],
 "reasons":[{"reason_code":"bones","name":"가시·뼈 잔존","role":"primary","link_status":"candidate","method":"lexicon"}]}
```

`evidence_kind=sentence`는 모델이 고른 근거 문장이라는 뜻이며 정확한 구절이 아닙니다. 사람 승인 의견은 `exact_span`입니다. `status=review_required`는 모델 점수가 임계값 근처라는 뜻입니다.
