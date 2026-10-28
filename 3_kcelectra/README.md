# 3단계 · KcELECTRA 사후학습 (3_kcelectra)

`beomi/KcELECTRA-base-v2022`(한국어 댓글로 사전학습된 모델)를 우리 리뷰로 한 번 더 학습해 식품 리뷰 말투와 어휘에 익숙하게 만듭니다. 결과 `frozen-encoder/`는 뒤 단계(감성·속성 LoRA)가 모두 불러 쓰는 본체이고, 그 뒤로는 바뀌지 않습니다.

```powershell
cd 3_kcelectra
python prepare_data.py     # 2_preprocessing의 리뷰 → train/validation/test.jsonl, sampling-report.json
python train.py            # 사후학습 3에폭 → frozen-encoder/ (1,000스텝마다 이어하기 저장)
python plot_training.py    # 학습 곡선 → training-curve.png
```

## 무엇을 했나

| 항목 | 내용 |
| --- | --- |
| 데이터 | 정상 리뷰 642,172건 중 같은 본문은 하나만(139,908건 제외), 정답 300리뷰·예전 실험 100리뷰와 연결된 리뷰 제외(1,833건) → 학습 494,941건 / 검증 2,000건 / 시험 2,000건. 원 리뷰와 한달사용 후기는 같은 묶음으로 같은 쪽에만 들어감 |
| 입력 | 리뷰 전체를 자르지 않고 넣음 (512토큰 넘는 18건 제외), 본문은 `review_text`(normalize_text) |
| 방식 | ELECTRA 바꿔치기 판별(RTD, [google-research/electra](https://github.com/google-research/electra)): 작은 생성기가 단어 15%를 바꾸고 본체가 어느 단어가 바뀌었는지 맞힘. 원래 KcELECTRA의 생성기는 공개되지 않아 새로 만든 작은 생성기(256차원, 12층)를 1에폭 먼저 학습한 뒤 함께 학습 |
| 설정 | 본체 학습률 1e-5, 생성기 1e-4, 판별 손실 ×50, 3에폭, 웜업 5% (`training-config.json`) |
| 고른 모델 | 고정 검증 문제에서 판별 손실이 가장 낮은 에폭을 내보냄 → **3에폭** (0.270 → 0.223 → 0.216 → 0.214) |
| 시험 세트 | 판별 손실 0.275 → 0.214, 교체 토큰 검출 F1 0.452 → 0.613 (`results.json`, `training-curve.png`) |
| 뒤 단계에서 | 감성 분류 교차검증 정확도 원래 모델 0.793 → 사후학습 0.845 (5_감성분류) |

**에폭 기록 바로잡기 (2026-10-28):** 그동안 문서와 기록에 'frozen-encoder = 1에폭'이라고 적혀 있었지만, `frozen-encoder/model.safetensors`의 해시가 `results.json`에 적힌 학습 끝 내보내기(3에폭)와 같고, 파일 시각도 그 내보내기와 1초 차이입니다. 지금까지의 모든 뒤 단계 결과(감성 분류 비교 'v3e1' 포함)는 이 3에폭 인코더로 나온 것입니다. 1~3에폭의 감성 점수가 같았다는 판단은 그대로 유효하고, 1·2에폭 체크포인트는 2026-10-23 정리 때 지워졌습니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `prepare_data.py` | 데이터 나누기. 2026-10-28 다시 돌려 학습·검증·시험 파일이 기록된 해시와 바이트까지 같음을 확인(`train.jsonl`은 크기 때문에 저장소에 두지 않고 필요할 때 이것으로 다시 만듦) |
| `train.py` | 사후학습 (생성기 준비 → 함께 학습 → 고정 검증으로 에폭 선택 → `frozen-encoder/` 내보내기) |
| `rtd_core.py`, `test_rtd_core.py` | 데이터 격리(묶음·같은 본문은 한 쪽에만)와 RTD 부품, 그 테스트 |
| `plot_training.py` | 학습 곡선 그림 |
| `frozen-encoder/` | 사후학습한 KcELECTRA 본체 (Git LFS, 509MB) |
| `prior-100-dataset.json` | 예전 실험 100리뷰 (학습에서 빼기 위한 입력) |
| `validation.jsonl`, `test.jsonl`, `sampling-report.json` | 검증·시험 리뷰와 나누기 기록 |
| `validation-corruptions.jsonl`, `test-corruptions.jsonl` | 모든 에폭을 같은 문제로 비교하려고 고정해 둔 바꿔치기 문제 |
| `training-config.json`, `history.json`, `best.json`, `results.json`, `training-audit.json`, `baseline-validation.json`, `generator-bootstrap-*.json` | 학습 설정, 에폭별 점수, 고른 에폭, 최종 결과, 기울기·변화량 점검, 학습 전 점수, 생성기 준비 기록 |
| `*-predictions.npz` | 학습 전·1에폭 검증, 학습 전·3에폭 시험의 판별 점수 원본 |

## 다시 학습할 때

새 리뷰가 많이 쌓였거나 표현 변화 신호가 커졌을 때 다시 합니다. 그때는 분석용 텍스트(v2)와 새 리뷰를 쓰는 게 맞습니다(지금 모델은 v1 텍스트로 학습했지만, v1과 v2의 토큰 차이는 1.0%뿐이라 다시 할 이유가 없었음). 본체가 바뀌면 감성·속성 LoRA와 재확인 기준도 다시 학습·계산해야 합니다.
