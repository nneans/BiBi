"""Build a hand-editable gold-label JSON from the 300-review human_eval bundle (no AI drafts, no ratings)."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / 'gold-300.json'
from bibi_reviews.backbone import normalize_text  # noqa: E402
SEED = 20261022

ASPECTS = {
    '맛': '맛·냄새 (맛있다, 짜다, 밀가루 맛, 이상한 냄새)',
    '식감': '바삭, 눅눅, 쫀득, 안 늘어남, 딱딱',
    '재료구성': '치즈가 적다·많다, 튀김옷이 두껍다 (재료 비율)',
    '양': '양·크기 (양이 많다, 크기가 작다)',
    '품질': '녹아서 옴, 부서짐, 내용물 하자',
    '신선도': '"신선하다", "상했다"처럼 직접 말한 것만',
    '배송': '빠르다, 늦다 ("잘 받았어요"는 사실)',
    '포장': '꼼꼼한 포장, 보냉·아이스팩, 터짐',
    '가격': '저렴, 가성비, 비싸다',
    '조리편의': '간편하다, 번거롭다',
    '조리결과': '에어프라이어로 하니 터짐 등 조리 후 결과',
    '설명일치': '안내·설명과 실제가 다름',
    '종합만족': '무엇이 좋은지 없이 "좋아요", "아이들이 잘 먹어요", 용도 만족',
    '종합불만': '무엇이 나쁜지 없이 "별로예요", 원료 불신 같은 애매한 불만',
    '재구매': '"또 살게요", "다시는 안 사요"',
    '사실': '평가 없는 사실 ("간식으로 샀어요", "어제 받았어요") — 감성은 "해당없음"',
    '모름': '속성을 정말 모르겠을 때',
}
SENTIMENTS = ['긍정', '부정', '중립', '혼합', '애매', '해당없음']
TARGETS = ['이 상품', '다른 상품', '일반 경험', '불명확']


def main():
    bundle = json.loads((HERE / 'review_bundle.json').read_text(encoding='utf-8'))
    from bibi_reviews.paths import CANONICAL_REVIEWS
    source = CANONICAL_REVIEWS
    table = pq.read_table(source, columns=['review_id', 'product_name', 'review_text_raw']).to_pylist()
    names = {r['review_id']: r['product_name'] for r in table}
    raw = {r['review_id']: r['review_text_raw'] for r in table}
    reviews = sorted(bundle['reviews'], key=lambda r: hashlib.sha256(f'{SEED}:{r["review_id"]}'.encode()).hexdigest())
    items = []
    for no, r in enumerate(reviews, 1):
        # Shown text is the bundle's normalized text; the evaluator re-aligns evidence offsets to raw text.
        assert r['original_text'] == normalize_text(raw[r['review_id']]), r['review_id']
        items.append({'no': no, 'review_id': r['review_id'], 'product': names[r['review_id']],
                      'text': r['original_text'], 'opinions': [], 'done': False, 'note': ''})
    doc = {
        '_guide': {
            '할 일': '리뷰마다 opinions에 의견을 추가하고, 다 봤으면 done을 true로 바꾼다. 의견이 하나도 없으면 opinions는 []로 두고 done만 true.',
            '의견 형식': {'evidence': '원문 구절을 그대로 복사 (띄어쓰기·맞춤법 고치지 않음)', 'aspect': '아래 속성 중 하나',
                       'sentiment': '아래 감성 중 하나', 'target': '기본값 "이 상품". 다른 상품 얘기면 "다른 상품"'},
            '의견 나누기': '속성이나 감성이 다르면 의견을 따로 만든다. 같은 구절이 두 속성에 걸치면 의견 2개로 같은 구절을 써도 된다.',
            '속성': ASPECTS,
            '감성': SENTIMENTS,
            '대상': TARGETS,
            '예시': {
                'text': '치즈가 고소하고 맛있는데 잘 안 늘어나요. 배송도 빨라요',
                'opinions': [
                    {'evidence': '치즈가 고소하고 맛있는데', 'aspect': '맛', 'sentiment': '긍정', 'target': '이 상품'},
                    {'evidence': '잘 안 늘어나요', 'aspect': '식감', 'sentiment': '부정', 'target': '이 상품'},
                    {'evidence': '배송도 빨라요', 'aspect': '배송', 'sentiment': '긍정', 'target': '이 상품'}]},
            '애매할 때': 'note에 이유를 적어둔다. 별점은 일부러 뺐다 (별점은 정답이 아님).',
            '자세한 기준': '04_AI분석/검수기준_한장요약.md',
        },
        'meta': {'source_bundle': 'review_bundle.json', 'source_bundle_sha256': hashlib.sha256((HERE / 'review_bundle.json').read_bytes()).hexdigest(),
                 'reviews': len(items), 'text': 'bibi_reviews.backbone.normalize_text(review_text_raw)', 'order': f'sha256({SEED}:review_id)', 'ai_drafts': False, 'ratings_shown': False},
        'reviews': items,
    }
    if OUT.exists():
        raise SystemExit(f'{OUT} already exists; not overwriting labels')
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(OUT, len(items))


if __name__ == '__main__':
    main()
