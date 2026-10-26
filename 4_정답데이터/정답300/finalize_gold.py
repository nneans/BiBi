"""Apply Claude's review of the external-AI draft and write the evaluation gold file.

Provenance: external AI draft (labeling.json) -> Claude full read-through with the edits below
-> user decides only the items in USER_CHECKS (defaults applied until answered).
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE / 'labeling.json'
OUT = HERE / 'gold-300-final.json'
LOG = HERE / 'review-changes.json'


def set_(no, k, why, **fields):
    return {'no': no, 'op': 'set', 'k': k, 'fields': fields, 'why': why}


def add(no, why, **opinion):
    return {'no': no, 'op': 'add', 'opinion': {'target': '이 상품', **opinion}, 'why': why}


def drop(no, k, why):
    return {'no': no, 'op': 'drop', 'k': k, 'why': why}


FACT = {'aspect': '사실', 'sentiment': '해당없음'}
EDITS = [
    set_(1, 1, '모래 씹힘은 이물(내용물 하자) → 품질', aspect='품질'),
    set_(10, 1, '두 의견 구절이 겹쳐 맛 근거를 분리', evidence='맛없는 홍합살'),
    set_(12, 2, '"맛이나"만으로는 감성 단어가 빠짐', evidence='맛이나 식감 다 별로에요'),
    set_(12, 3, '맛·식감이 같은 술어를 공유', evidence='맛이나 식감 다 별로에요'),
    set_(15, 4, '"맛있어요."가 원문에 2번 — 두 번째 위치로 고정', evidence='바싹 맛있어요.'),
    set_(17, 5, '조리 안내만 있음 → 사실 (기준: 조리 온도·시간 안내만 → 사실)', **FACT),
    set_(18, 2, '조리 팁 → 사실', **FACT),
    set_(33, 2, '조건부 재구매("맛있으면")는 평가가 아님 → 사실', **FACT),
    set_(35, 1, '원산지 확인 못한 후회 → 애매한 불만(2026-10-19 정책: 원료 불신 → 종합불만)', sentiment='부정'),
    set_(38, 2, '"연육함량 높아서 일반맛살과는 틀림" → 더 낫다는 뜻', aspect='종합만족', sentiment='긍정'),
    set_(42, 2, '"맛이 별루 … 예상했던 맛" → 예상대로 별로', sentiment='부정'),
    set_(44, 3, '재냉동 의심은 불만 표현', sentiment='부정'),
    set_(58, 3, '"짝퉁 같다" → 원료 불신(종합불만)', aspect='종합불만'),
    set_(70, 3, '"요리할 때 역했다" → 냄새', aspect='맛'),
    set_(72, 1, '행사 구매 사실 + 인사만 있음 → 사실', **FACT),
    set_(80, 3, '원산지 탓하는 불만', sentiment='부정'),
    set_(84, 4, '냄새가 안 난 것은 좋은 점', sentiment='긍정'),
    set_(97, 3, '흠집 없이 도착은 포장 결과', aspect='포장'),
    set_(108, 1, '"배송상태"는 속도가 아니라 도착 상태 → 포장', aspect='포장'),
    set_(115, 4, '양념장 양이 적음 → 재료구성 (겹친 의견 정리)', aspect='재료구성'),
    drop(115, 5, '4번과 같은 내용의 겹친 구절'),
    add(116, '반어적 혹평 누락', evidence='장어낚시 미끼로나 써야겠어요.', aspect='종합불만', sentiment='부정'),
    set_(117, 1, '"여전히 … 느낌이 납니다" → 계속되는 불만', sentiment='부정'),
    add(133, '외관 긍정 누락', evidence='생긴건 정말 맛있어보이는데..', aspect='품질', sentiment='긍정'),
    set_(138, 2, '"새우 9개임"은 사실', **FACT),
    set_(153, 2, '"맛있었어요"가 원문에 2번 — 첫 위치로 고정', evidence='하니 맛있었어요'),
    set_(180, 2, '세일 요청은 평가가 아님 → 사실', **FACT),
    set_(206, 1, '"매번" 한 단어는 근거로 부족', evidence='매번 저렴하게 잘 구매하고있어요'),
    set_(209, 1, '"가격도"만으로는 감성 단어가 빠짐', evidence='가격도 맛도 최고에요.'),
    set_(209, 2, '가격·맛이 같은 술어를 공유', evidence='가격도 맛도 최고에요.'),
    set_(212, 2, '아이스팩이 얼어 있었다는 사실', **FACT),
    set_(223, 5, '"전 대구가 좋아요" → 다른 상품에 대한 선호(긍정)', sentiment='긍정'),
    set_(244, 1, '손바닥 크기가 좋은지 나쁜지 알 수 없음', sentiment='애매'),
    set_(259, 2, '포장 불량 의심 + 물이 떨어짐 → 부정', sentiment='부정'),
]

# Defaults applied now; the user can override by answer number.
USER_CHECKS = [
    {'id': 1, 'no': 84, 'k': 5, 'evidence': '아이스팩도 세개 다 반절정도 얼음이 있었는데..',
     'question': '보냉 불만인가, 단순 사실인가', 'default': FACT, 'alternative': {'aspect': '포장', 'sentiment': '부정'}},
    {'id': 2, 'no': [35, 80, 203], 'evidence': '중국산인줄 모르고 샀네요 / 원산지 따라 차이가 큰거 같네요 / 야채가 모두 중국산인거 인가봐요',
     'question': '원산지 언급 불만을 종합불만·부정으로 둘지', 'default': {'aspect': '종합불만', 'sentiment': '부정'}, 'alternative': {'aspect': '종합불만', 'sentiment': '애매'}},
    {'id': 3, 'no': 29, 'k': 1, 'evidence': '생각보다 그냥그래요',
     'question': '중립인가, 기대 이하(부정)인가', 'default': {'sentiment': '중립'}, 'alternative': {'sentiment': '부정'}},
    {'id': 4, 'no': 243, 'k': 1, 'evidence': '버터구이는 역시나에요',
     'question': '"역시나" = 역시 맛있다(긍정)인가, 판단 불가인가', 'default': {'sentiment': '애매'}, 'alternative': {'sentiment': '긍정'}},
    {'id': 5, 'no': 128, 'k': 3, 'evidence': '담엔 뼈없는걸로 사보려구여.',
     'question': '이 상품을 다시 안 산다(부정)인가', 'default': {'sentiment': '애매'}, 'alternative': {'sentiment': '부정'}},
    {'id': 6, 'no': 244, 'k': 1, 'evidence': '꼬막이 손바닥만해요.',
     'question': '크다는 칭찬인가, 너무 커서 불만인가, 판단 불가인가', 'default': {'sentiment': '애매'}, 'alternative': {'sentiment': '긍정'}},
]

# User answers (2026-10-22): 1 포장·부정, 2 부정(default), 3 부정, 4 긍정, 5 부정, 6 긍정.
USER_ANSWERS = [
    set_(84, 5, 'user check 1: 보냉 불만', aspect='포장', sentiment='부정'),
    set_(29, 1, 'user check 3: 기대 이하', sentiment='부정'),
    set_(243, 1, 'user check 4: 역시 맛있다', sentiment='긍정'),
    set_(128, 3, 'user check 5: 이 상품 재구매 안 함', sentiment='부정'),
    set_(244, 1, 'user check 6: 크다는 칭찬', sentiment='긍정'),
]


def main():
    draft = json.loads(DRAFT.read_text(encoding='utf-8'))
    reviews = {r['no']: copy.deepcopy(r) for r in draft['reviews']}
    log = []
    drops = []
    for e in EDITS + USER_ANSWERS:
        r = reviews[e['no']]
        if e['op'] == 'set':
            before = dict(r['opinions'][e['k'] - 1])
            r['opinions'][e['k'] - 1].update(e['fields'])
            log.append({**e, 'before': before, 'after': r['opinions'][e['k'] - 1]})
        elif e['op'] == 'add':
            r['opinions'].append(e['opinion'])
            log.append(e)
        else:
            drops.append(e)
    for e in sorted(drops, key=lambda e: -e['k']):
        before = reviews[e['no']]['opinions'].pop(e['k'] - 1)
        log.append({**e, 'before': before})
    for r in reviews.values():
        for o in r['opinions']:
            assert o['evidence'] in r['text'], (r['no'], o['evidence'])
            assert r['text'].count(o['evidence']) == 1, (r['no'], o['evidence'])
        r['label_source'] = 'external_ai_draft+claude_review'
        if any(e['no'] == r['no'] for e in USER_ANSWERS) or r['no'] in (35, 80, 203):
            r['label_source'] += '+user_decided'
        r['user_check_ids'] = [c['id'] for c in USER_CHECKS if r['no'] in (c['no'] if isinstance(c['no'], list) else [c['no']])]
        r['done'] = True
    doc = {'_guide': draft['_guide'],
           'meta': {**draft['meta'], 'ai_drafts': True,
                    'provenance': 'external AI draft (GPT, user-reported 2026-10-22) -> Claude read all 300 and edited -> user decided USER_CHECKS (6 items)',
                    'not_fully_human_verified': True, 'draft_sha256': hashlib.sha256(DRAFT.read_bytes()).hexdigest(),
                    'edits': len(EDITS), 'user_checks': USER_CHECKS, 'user_checks_resolved': True, 'user_answers': [e['why'] for e in USER_ANSWERS]},
           'reviews': [reviews[n] for n in sorted(reviews)]}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ops_before = sum(len(r['opinions']) for r in draft['reviews'])
    ops_after = sum(len(r['opinions']) for r in doc['reviews'])
    touched = len({e['no'] for e in EDITS})
    print(f'edits {len(EDITS)} on {touched} reviews; opinions {ops_before} -> {ops_after}')


if __name__ == '__main__':
    main()
