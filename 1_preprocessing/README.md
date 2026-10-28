# 1단계 · 전처리 (1_preprocessing)

네이버 스마트스토어에서 받은 리뷰 원본 CSV 하나에서, 뒤 단계가 읽는 파일을 **명령 하나로** 만듭니다. 원본은 고치지 않습니다.

```powershell
python 1_preprocessing/preprocess.py                            # 원본 CSV → output/ (같은 CSV면 다시 만들어 비교만 하고 그대로 둠)
python 1_preprocessing/preprocess.py --csv <새.csv> --replace   # 새 CSV로 결과를 바꿀 때
python 1_preprocessing/check_clean.py                           # 분석용 텍스트 정리 규칙 점검 → output/checks/
```

## 폴더

| 경로 | 내용 |
| --- | --- |
| `preprocess.py` | 전처리 전체 (아래 흐름 ①~⑤) |
| `check_clean.py` | 분석용 텍스트 정리 규칙 점검: 규칙별 해당 건수·예시, 줄바꿈 판정 표본, 정답 300리뷰 단위 점수 |
| [raw/](raw/README.md) | 원본 CSV(수정 금지)와 `audit/`(처음 점검한 기록: 데이터 진단, 품질 점검) |
| `output/` | `preprocess.py`가 만드는 파일 (아래 표), `checks/`(`check_clean.py` 결과) |

## 흐름

```
원본 CSV 649,524행
 ① 읽기·검사      DB 반영과 같은 규칙(backbone.ReviewStore.prepare): 필수 컬럼, 리뷰글번호 중복 없음, 별점 1~5,
                  날짜는 한국 시간, 본문 normalize_text(HTML 기호 되돌리기·NFC·공백 정리), text_hash
 ② 정상 리뷰      블라인드 7,352건 제외 → 642,172건 · 상품 539개
                  review_group_id = 'review:' + (관련리뷰글번호 또는 리뷰글번호): 원 리뷰와 한달사용 후기가 같은 값
                  → output/reviews_normal.parquet
 ③ 상품 이름      product_id마다 가장 많이 쓰인 이름 하나(동률이면 이름순). 상품은 product_id로만 구분하고 합치지 않음
                  → output/canonical_products/
 ④ 분석용 텍스트  text_clean v2(아래 표) + purchase_id(한 고객의 한 구매) → output/analysis_text.parquet
 ⑤ 기록           입력 해시, 단계별 건수, 규칙, 출력 해시 → output/preprocess-manifest.json
```

## 출력 (`output/`)

| 파일 | 내용 | 쓰는 곳 |
| --- | --- | --- |
| `reviews_normal.parquet` | 정상 리뷰 642,172건 (원래 상품명) | 상품명 정리 입력 |
| `canonical_products/reviews_normal_canonical.parquet` | 위 + 대표 상품명(`product_name`), 원래 이름(`product_name_original`) | 사후학습, 정답 만들기, 이유 분석 |
| `canonical_products/product_name_mapping.csv` | product_id → 대표 이름, 이름 수, 리뷰 수 (분석할 상품 목록) | 이유 분석, 보고서 |
| `canonical_products/product_name_aliases.csv` | product_id × 예전 이름 → 리뷰 수 | 상품 이름만 쓴 단위 걸러내기 |
| `analysis_text.parquet` | review_id → `analysis_text`(v2), `purchase_id` | 이유 분석(단위 나누기, 구매 기준 집계) |
| `preprocess-manifest.json` | 입력 해시, 건수, 규칙, 출력 해시 | 재현 확인 |

2026-10-28 원본 CSV에서 다시 만들어 그 전의 파일 4개와 내용이 완전히 같음을 확인했습니다(그 전에는 ①②를 만든 코드가 저장소에 없었음). 규칙은 `tests/test_preprocess.py`가 지킵니다.

## 두 가지 텍스트

| 컬럼 | 규칙 | 왜 따로 두나 |
| --- | --- | --- |
| `review_text` | `normalize_text` | DB 텍스트·해시(새 CSV의 신규·수정 판정), 정답 300리뷰의 근거 위치, 사후학습 데이터가 여기에 묶여 있어 바꾸지 않음 |
| `analysis_text` | `text_clean.clean_review` v2 | 단위 나누기·모델·보고서용 |

v2 정리 규칙(642,172건 기준, `output/checks/clean-report.json`):

| 규칙 | 해당 리뷰 | 예 |
| --- | ---: | --- |
| 줄바꿈 유지 (문장 끝 뒤의 줄바꿈 = 단위 경계, 문장 중간 = 이어 붙임, `kiwi_units`) | 98,432 | '괜츈해요⏎많이 파세요' → 두 단위 / '떼어내느라⏎고생했어요' → 한 단위 |
| 같은 문자 3번 이상 → 2번 (숫자 제외, KcELECTRA 원작 데이터 방식) | 30,462 | '좋아요~~~~~~~~' → '좋아요~~' |
| 옛 한글 점 'ᆢ' → '..', 'ᆞ' → '.' | 614 | '불편하긴한데ᆢ좀녹여서' |
| 보이지 않는 문자 제거 | 138 | 폭 없는 공백, BOM |
| 숫자 10자리 이상 제거 (글자 수 채우기, 전화번호) | 129 | '1234567890' (날짜 20250211은 남김) |
| 띄어 쓴 점 → '..' | 110 | '굿 . . .' |
| 블로그 주소 제거 | 15 | 'https://m.blog.naver.com/…' |

- 일부러 남기는 것: 이모지·♡·ㅠㅠ·ㅡㅡ·^^(KcELECTRA가 학습한 기호: 이모지 벡터가 한글만큼 학습돼 있고 😭 옆에 😢·ㅠㅠ가 있음), 맞춤법·띄어쓰기(근거 문장은 고객의 말 그대로).
- 정답 300리뷰로 확인: 정답 의견을 단위 하나로 찾는 비율 84.2% → 84.8%, 의견이 섞인 단위 70 → 68개.
- 아직 못 잡는 것: 명사로 끝나는 줄('굿굿⏎너무 맛있어요')은 경계로 보지 않습니다(예전과 같음). 정리 후 빈 리뷰 101건(숫자·점만 있던 리뷰)은 단위가 없습니다.

## 구매 기준 (`purchase_id`)

한 고객의 같은 의견은 한 번만 셉니다(2026-10-28 결정). 리뷰 수는 그대로 모두 셉니다.

| 묶는 것 | 규모 |
| --- | ---: |
| 원 리뷰 + 한달사용 후기 (같은 review_group_id) | 한달사용 231,475건 중 원 리뷰가 데이터에 있는 것 181,685건 |
| 같은 상품에 다시 붙여 넣은 40자 이상 리뷰 → 첫 리뷰의 구매 | 2,773건 |
| 결과 | 리뷰 642,172건 → 구매 458,990건 |

## 참고

- 원본에는 카테고리 컬럼이 없습니다(상품주문번호, 마스킹된 등록자 nr****는 있음). 상품군은 비비수산 스토어의 카테고리를 받아 정할 예정이고, 그 전의 임시 초안은 공식이 아니어서 저장소에 올리지 않고 로컬 `archive/`에 둡니다.
- 흐름 그림: [preprocess-flow.png](../7_보고/그림/preprocess-flow.png)
