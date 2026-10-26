"""Turn the external-AI draft (labeling.json) into a human verification file, riskiest reviews first.

Ratings are used only to order reviews (AI sentiment vs rating disagreement) and are never shown.
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE / 'labeling.json'
OUT = HERE / 'verify-300.json'
FUZZY_ASPECTS = {'종합만족', '종합불만', '모름', '사실'}
UNSURE_SENTIMENTS = {'애매', '혼합', '중립'}


def check_points(review):
    text, ops = review['text'], review['opinions']
    points, score = [], 0
    if review['note']:
        points.append('AI 메모 있음: ' + review['note'])
        score += 3
    if any(o['sentiment'] in UNSURE_SENTIMENTS for o in ops):
        points.append('감성이 애매/혼합/중립인 의견 있음')
        score += 3
    if any(o.get('target', '이 상품') != '이 상품' for o in ops):
        points.append('다른 상품/일반 경험 대상 의견 있음')
        score += 2
    if any(text.count(o['evidence']) > 1 for o in ops):
        points.append('같은 구절이 원문에 2번 이상 나옴 — 어느 쪽인지 note에 적기')
        score += 2
    spans = [(text.find(o['evidence']), text.find(o['evidence']) + len(o['evidence'])) for o in ops]
    if any(a[0] < b[1] and b[0] < a[1] for i, a in enumerate(spans) for b in spans[i + 1:]):
        points.append('구절끼리 겹침')
        score += 2
    if any(len(o['evidence'].strip()) <= 3 and o['aspect'] not in FUZZY_ASPECTS for o in ops):
        points.append('아주 짧은 구절 — 감성 단어가 빠지지 않았는지')
        score += 1
    if len(ops) >= 4:
        points.append(f'의견 {len(ops)}개 — 빠지거나 과하게 나눈 것 없는지')
        score += 1
    if not ops:
        points.append('의견 없음으로 처리됨')
        score += 2
    if sum(o['aspect'] in FUZZY_ASPECTS for o in ops) and any(o['aspect'] not in FUZZY_ASPECTS for o in ops):
        score += 1
    return points, score


def rating_disagreement(review, rating):
    pols = {o['sentiment'] for o in review['opinions']}
    if rating is None:
        return 0
    if rating <= 2 and '부정' not in pols:
        return 3
    if rating >= 4 and '부정' in pols and '긍정' not in pols:
        return 3
    return 0


def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} already exists; not overwriting verification work')
    draft = json.loads(DRAFT.read_text(encoding='utf-8'))
    bundle = json.loads((HERE / 'review_bundle.json').read_text(encoding='utf-8'))
    ratings = {r['review_id']: int(r['rating']) if str(r.get('rating', '')).isdigit() else None for r in bundle['reviews']}
    rows = []
    for review in draft['reviews']:
        points, score = check_points(review)
        score += rating_disagreement(review, ratings.get(review['review_id']))
        item = {'no': review['no'], 'review_id': review['review_id'], 'product': review['product'], 'text': review['text'],
                '확인포인트': points, 'opinions': copy.deepcopy(review['opinions']), 'done': False,
                'note': '', 'ai_note': review['note']}
        rows.append((-score, review['no'], item))
    rows.sort(key=lambda r: (r[0], r[1]))
    reviews = []
    for order, (_, _, item) in enumerate(rows, 1):
        reviews.append({'order': order, **item})
    guide = copy.deepcopy(draft['_guide'])
    guide['할 일'] = ('opinions에는 AI 초안이 들어 있다. 맞으면 그대로, 틀리면 고치고, 빠진 의견은 추가하고, 없는 의견은 지운다. '
                    '확인이 끝나면 done을 true로. 확인포인트는 참고용이며 정답이 아니다. 위쪽일수록 틀렸을 가능성이 높다.')
    guide['note'] = '내가 고친 이유나 애매한 점을 적는다. ai_note는 AI가 남긴 메모(참고용).'
    doc = {'_guide': guide,
           'meta': {**draft['meta'], 'ai_drafts': True, 'draft_source': 'GPT (user-reported 2026-10-22)',
                    'draft_file': DRAFT.name, 'draft_sha256': hashlib.sha256(DRAFT.read_bytes()).hexdigest(),
                    'order': 'risk score desc (notes, unsure sentiment, overlaps, rating disagreement), then original no',
                    'ratings_shown': False},
           'reviews': reviews}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    flagged = sum(1 for r in reviews if r['확인포인트'])
    print(OUT, len(reviews), 'with check points:', flagged)


if __name__ == '__main__':
    main()
