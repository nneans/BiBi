"""Collect every item marked check=true in candidates*.json into one file for the user: 확인필요.json.

  python build_check_file.py [--force] -> 확인필요.json (items to look at, with the rules on top)
  python build_check_file.py apply    -> copies the labels from 확인필요.json back into candidates*.json
                                         and marks those items check=false, checked_by=user
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
CHECK = OUT / '확인필요.json'
LABELS = ['긍정', '부정', '평가 아님', '제외']  # 제외 = too ambiguous to learn from; kept in the file, skipped in training

GUIDE = {
    '할 일': "아래 items에서 label이 틀렸으면 고치고, 맞으면 그대로 두세요. label은 '긍정' / '부정' / '평가 아님' 중 하나. "
            "고민되면 note에 한 줄 남겨도 됩니다. 다 보면 알려 주세요(제가 원래 파일에 반영합니다).",
    '무엇을 보나': "unit(리뷰에서 잘라낸 한 조각)만 판단합니다. review는 그 조각이 무슨 이야기인지 확인하는 맥락용입니다. "
              "리뷰 전체가 칭찬이어도 이 조각이 아쉬움이면 부정입니다.",
    '부정': ["이 상품의 단점·불만·아쉬움. '다만', '~지만', '~긴 한데'가 붙어도 부정 (예: '살짝 녹아서 왔지만', '좀 짜긴 하지만')",
           "미지근한 평가도 부정 (예: '평범해요', '그냥 그래요', '생각보다 보통이에요')",
           "판매자·배송·포장에 대한 불만도 부정 (예: '연락도 없어요', '아이스박스가 깨져 왔네요')"],
    '긍정': ["이 상품에 대한 칭찬. 돌려 말한 칭찬도 긍정 (예: '마트 초밥 살 필요 없어요', '다른 맛살 못 드시겠대요')",
           "재구매·단골 표현도 긍정 (예: '몇 번째 주문인지 모르겠어요', '냉동실에 없으면 불안해요')"],
    '평가 아님': ["다른 상품·다른 가게 이야기 (예: '마트에선 비싼데', '다른 데서 산 건 비려서 못 먹었는데')",
              "산 이유·상황 설명 (예: '샐러드 매번 자르는 거 귀찮고', '선물용으로 샀는데')",
              "평가 없는 조리법·사용 팁, 사실 설명 (예: '구이로 드실 분은 소금 하세요', '유통기한 4월 23일', '700g 2개로 왔어요')",
              "인사·토막 (예: '잘 받았습니다', '번창하세요', '있습니다.', '싶어요!')",
              "본인 실수 (예: '잘못 시켰어요', '1키로인 줄 알았는데 잘못 봤는지')"],
    '자주 헷갈리는 것': ["맵기·크기 설명: 불만으로 말하면 부정('너무 매워요', '작아서 아쉬워요'), 단순 설명·주의면 평가 아님"
                   "('맵찔이는 못 먹을 듯', '약간 매운데 맛있어요')",
                   "팁 + 결과: 결과가 이 상품 칭찬이면 긍정('기름에 튀겨 먹으니 고점이 높네요'), 방법만이면 평가 아님"],
    'reason': "item마다 제가 왜 애매하다고 봤는지 한 줄 적었습니다.",
}

REASONS = [  # (pattern on the unit, why it is a judgement call), first match wins
    (r'맵|매콤|매워|매운', '맵기 설명: 불만인지 단순 설명인지'),
    (r'평범|보통|그냥|쏘쏘|그저|평이', '미지근한 평가: 부정으로 볼지'),
    (r'재구매|주문|구매|시킬|몇번째|정착', '구매·재구매 이야기: 칭찬으로 볼지'),
    (r'크|작|사이즈|두께|두꺼|얇', '크기·두께 설명: 불만인지 단순 설명인지'),
    (r'하세요|드세요|해야|넣으|돌리|튀기|구워|굽|조리|해동|끓이', '조리·사용 팁: 평가가 들어 있는지'),
    (r'가격|비싸|저렴|싸', '가격 이야기: 이 상품 평가인지'),
    (r'녹|아이스|드라이', '배송 상태: 불만인지 사실인지'),
]


def reason(unit):
    for pattern, why in REASONS:
        if re.search(pattern, unit):
            return why
    return '맥락에 따라 달라짐'


def context(review, unit, width=80):
    review = review.replace('\n', ' ')
    i = review.find(unit)
    if i < 0 or len(review) <= 2 * width + len(unit):
        return review
    s, e = max(0, i - width), min(len(review), i + len(unit) + width)
    return ('…' if s else '') + review[s:e] + ('…' if e < len(review) else '')


def build():
    # 확인필요.json carries hand-written reasons (2026-10-25) and the user's edits; never overwrite it silently
    if CHECK.exists() and '--force' not in sys.argv:
        sys.exit(f'{CHECK.name} exists; use --force to rebuild (hand-written reasons and edits would be lost)')
    items = []
    for f in sorted(OUT.glob('candidates*.json')):
        for it in json.loads(f.read_text(encoding='utf-8'))['items']:
            if it.get('check'):
                items.append({'key': f'{f.stem}#{it["id"]}', 'product': it['product'], 'unit': it['unit'],
                              'label': it['label'], 'reason': reason(it['unit']), 'note': '',
                              'review': context(it['review'], it['unit']), 'model': it['model']})
    CHECK.write_text(json.dumps({'guide': GUIDE, 'count': len(items), 'items': items}, ensure_ascii=False, indent=1) + '\n',
                     encoding='utf-8')
    print(f'{len(items)} items -> {CHECK.name}')


def apply():
    checked = {it['key']: it for it in json.loads(CHECK.read_text(encoding='utf-8'))['items']}
    bad = [k for k, it in checked.items() if it['label'] not in LABELS]
    if bad:
        sys.exit(f'label must be one of {LABELS}: {bad}')
    changed = 0
    for f in sorted(OUT.glob('candidates*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        for it in d['items']:
            c = checked.get(f'{f.stem}#{it["id"]}')
            if c:
                changed += it['label'] != c['label']
                it.update(label=c['label'], check=False, checked_by='user', note=c['note'], exclude=c['label'] == '제외')
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{len(checked)} items applied, {changed} labels changed by the user')


if __name__ == '__main__':
    apply() if sys.argv[1:2] == ["apply"] else build()
