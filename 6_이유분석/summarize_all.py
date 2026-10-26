"""One table over every finished product in results/: label shares, top complaint/praise groups, rising complaints,
plus insights.md: per product, the specific complaints and praise side by side.

Generic groups ('맛있어요', '배송 빨라요', '그냥 그래요') say nothing a product page or a CS answer can use, so they
are kept in the counts but never lead the report: a group is specific when one of its top keywords is neither a
generic word nor part of the product name. Complaint and praise groups that share such keywords are paired
(e.g. 치즈 늘어남: praise 402 vs complaint 601).
Groups that say the same thing are merged first (merge_groups.py, threshold calibrated on the 148 judged pairs).
Usage: python summarize_all.py [run]   -> results[-run]/summary.json, summary.md, insights.md
"""
import json
import sys
from pathlib import Path

from bibi_reviews.cpu import pin_to_efficiency_cores  # noqa: E402
pin_to_efficiency_cores()  # before numba/torch load: this PC's P-cores are unstable (see bibi_reviews/cpu.py)
from kiwipiepy import Kiwi

OUT = Path(__file__).resolve().parent
RESULTS = OUT / ('results-' + sys.argv[1] if len(sys.argv) > 1 else 'results')  # python summarize_all.py v2
TOP = 3
# A complaint group counts as rising when it has at least this many units and its recent rate per 100 reviews
# is this many points above the earlier rate.
RISING_MIN_UNITS, RISING_MIN_DELTA = 20, 1.0
GENERIC = set('''맛 맛있다 맛나다 맛없다 너무 진짜 정말 아주 완전 좀 조금 약간 살짝 많이 더 다 잘 먹다 좋다 괜찮다 굿 최고 짱
만족 만족하다 마음 들다 배송 빠르다 포장 오다 받다 구매 주문 또 재구매 항상 늘 계속 시키다 사다 저렴하다 저렴 가격 싸다
가성비 대비 별로 최악 그냥 그냥저냥 그럭저럭 그저 쏘쏘 쏘다 평범 평범하다 실망 아쉽다 아쉬움 기대 생각 입맛 모르다 제품
상품 퀄리티 하다 되다 같다 있다 없다 안 개인 추천 비교 비 전체 부분 쫌 감사 감사하다 역시 제일 보이다 이번 처음 아직
번창 수고 후기 리뷰 느낌 편 정도 거 것 수
조아요 자주 금방 다음 갑 드시다 장 착하다 빨리 매우 넘 전 아 보다 의사 구입 구입하다 나다 물건 도착 인기 아끼다
상태 꼼꼼 꼼꼼하다 요리 사용 집 강추 대박 꿀맛 존맛 맛집 사먹다'''.split())  # second line: from the 2026-10-25 leak check
SPECIFIC_KEYWORDS = 4  # how many top keywords decide whether a group is specific
KEEP_TAGS = {'NNG', 'NNP', 'VV', 'VA', 'XR'}
kiwi = Kiwi()


def first_text(group):
    """Text nearest the group centre (report v2); older reports fall back to the most frequent readable text."""
    if group.get('representative'):
        return group['representative'][0]
    texts = [t.rsplit(' (', 1)[0] for t in group['top_texts']]
    readable = [t for t in texts if len(t.replace(' ', '')) >= 8]
    return (readable or texts or [''])[0]


def group_label(group):
    return f"[{'·'.join(group['keywords'][:3])}] {first_text(group)}"


def informative(word, name):
    return word not in GENERIC and not (len(word) >= 2 and word in name)


def text_words(text):
    return [t.form + ('다' if t.tag in ('VV', 'VA') else '') for t in kiwi.tokenize(text) if t.tag in KEEP_TAGS]


def is_specific(group, name):
    """The group's central sentence carries a non-generic word, or two of its top keywords do. Keywords alone missed groups like
    [맛·좀·같다] '조금 많이 달긴 한데' (달다) and [좀·안·좋다] '당근이 좀 많은 느낌' (당근)."""
    if group['topic'] == -1:
        return False
    # one stray keyword is not enough: '[굿·좋다·내용물·최고] 좋아요....' passed on '내용물' alone
    in_sentence = any(informative(k, name) for k in text_words(first_text(group)))
    in_keywords = len({k for k in group['keywords'][:SPECIFIC_KEYWORDS] if informative(k, name)})
    return in_sentence or in_keywords >= 2


def brief(x, name):
    return {'text': first_text(x), 'label': group_label(x), 'units': x['units'], 'share': x['share'],
            'keywords': x['keywords'][:5],
            # for pairing, product-name words count too: '에어프라이어' is in the cheese-stick title yet is the subject
            'content': list(dict.fromkeys(k for k in x['keywords'][:5] + text_words(first_text(x)) if k not in GENERIC)),
            'name_words': [k for k in x['keywords'][:5] + text_words(first_text(x)) if len(k) >= 2 and k in name],
            'recent': x['per_100_reviews_recent'], 'earlier': x['per_100_reviews_earlier']}


def merged_report(report, encode):
    """Report with same-meaning groups merged per sentiment; specific and generic groups stay apart.
    Reports from run_product v3 are merged already."""
    if report.get('merged'):
        return {**report, 'merges': {k: report['groups'][k].get('merges', []) for k in ['NEG', 'POS']}}
    from merge_groups import merge
    name = report['product_name']
    report = {**report, 'groups': {k: dict(v) for k, v in report['groups'].items()}, 'merges': {}}
    for label in ['NEG', 'POS']:
        groups, log = merge(report['groups'][label]['groups'], encode, kind=lambda g: is_specific(g, name))
        report['groups'][label]['groups'] = groups
        report['groups'][label]['topics'] = len([g for g in groups if g['topic'] != -1])
        report['merges'][label] = log
    return report


def summarize(product, report):
    labels = report['unit_labels']
    total = report['units']
    name = report['product_name']
    out = {'id': product['id'], 'name': product['name'], 'category': product['category'],
           'reviews': report['reviews'], 'units': total,
           'share': {k: round(labels.get(k, 0) / total, 3) for k in ['POS', 'NEG', 'NONE', 'REVIEW']},
           'period': report['period'], 'recent_months': report['recent_months'], 'recent_reviews': report['recent_reviews']}
    for label in ['NEG', 'POS']:
        g = report['groups'][label]
        specific = [x for x in g['groups'] if is_specific(x, name)]
        out[label] = {'units': g['units'], 'topics': g['topics'], 'outlier_share': g['outlier_share'],
                      'specific_topics': len(specific),
                      'specific_share': round(sum(x['units'] for x in specific) / max(g['units'], 1), 3),
                      'top': [brief(x, name) for x in specific[:TOP]],
                      'specific': [brief(x, name) for x in specific]}
    neg = [x for x in report['groups']['NEG']['groups'] if is_specific(x, name) and x['units'] >= RISING_MIN_UNITS]
    out['merges'] = {k: len(v) for k, v in report.get('merges', {}).items()}
    out['rising_NEG'] = sorted(({'text': first_text(x), 'label': group_label(x), 'recent': x['per_100_reviews_recent'], 'earlier': x['per_100_reviews_earlier']}
                                for x in neg if x['per_100_reviews_recent'] - x['per_100_reviews_earlier'] >= RISING_MIN_DELTA),
                               key=lambda x: x['earlier'] - x['recent'])
    return out


def insights(rows, n_neg=8, n_pos=5):
    """Per product: specific complaints with the praise group on the same subject, then the remaining specific praise."""
    lines = ['# 상품별 구체적 불만·칭찬', '',
             "일반 칭찬·불만('맛있어요', '배송 빨라요', '그냥 그래요')은 건수에는 남기고 여기서는 뺐습니다.",
             '숫자는 단위 수, "이전 → 최근"은 최근 3개월 전과 최근 3개월의 리뷰 100건당 건수. '
             '"같은 주제 칭찬"은 키워드가 2개 이상 겹치는 칭찬 묶음입니다.', '']
    for r in rows:
        neg, pos = r['NEG']['specific'], r['POS']['specific']
        used = set()
        lines += [f"## {r['name']} ({r['category']}, 리뷰 {r['reviews']:,})", '',
                  f"구체적 불만은 불만 단위의 {100 * r['NEG']['specific_share']:.0f}%, "
                  f"구체적 칭찬은 칭찬 단위의 {100 * r['POS']['specific_share']:.0f}%.", '',
                  '| 불만 | 건수 | 이전 → 최근 | 같은 주제 칭찬 | 건수 |', '| --- | ---: | --- | --- | ---: |']
        for x in neg[:n_neg]:
            overlap = lambda p: set(p['content']) & set(x['content'])
            # a pair needs 2 shared words, and not only title words ('치즈'+'스틱' alone would pair anything)
            ok = lambda p: len(overlap(p)) >= 3 or (len(overlap(p)) == 2 and overlap(p) - set(x['name_words']))
            match = max((p for p in pos if ok(p)), key=lambda p: (len(overlap(p)), p['units']), default=None)
            if match:
                used.add(match['label'])
                pair = f"{match['text']} | {match['units']:,}"
            else:
                pair = '- | -'
            lines.append(f"| [{'·'.join(x['keywords'][:3])}] {x['text']} | {x['units']:,} | {x['earlier']} → {x['recent']} | {pair} |")
        rest = [p for p in pos if p['label'] not in used][:n_pos]
        if rest:
            lines += ['', '그 밖의 구체적 칭찬: ' + ' / '.join(f"{p['text']} ({p['units']:,})" for p in rest)]
        lines.append('')
    return lines


def main():
    products = json.loads((OUT / 'products.json').read_text(encoding='utf-8'))
    sys.modules['tensorflow'] = None  # see run_product.py
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, str(OUT.parent / 'src'))
    from bibi_reviews.paths import KURE_V1
    kure = SentenceTransformer(str(KURE_V1), device='cuda', local_files_only=True)
    encode = lambda texts: kure.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    rows = [summarize(p, merged_report(json.loads((RESULTS / p['id'] / 'report.json').read_text(encoding='utf-8')), encode))
            for p in products if (RESULTS / p['id'] / 'report.json').exists()]
    (RESULTS / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pct = lambda v: f'{100 * v:.0f}%'
    lines = [f'# 상품별 칭찬·불만 요약 ({len(rows)}개 상품)', '',
             '단위 비율은 긍정/부정/평가 아님/재확인(확신도 0.8 미만). 불만·칭찬 칸은 일반 묶음을 뺀 구체적 묶음 1위. '
             '괄호 숫자는 그 이전 → 최근 3개월, 리뷰 100건당 건수. 상품별 상세는 insights.md.', '',
             '| 분류 | 상품 | 리뷰 | 단위 | 긍정/부정/아님/재확인 | 불만 묶음 수(구체) | 가장 큰 구체적 불만 | 가장 큰 구체적 칭찬 |',
             '| --- | --- | ---: | ---: | --- | ---: | --- | --- |']
    for r in rows:
        s = r['share']
        neg = r['NEG']['top'][0] if r['NEG']['top'] else None
        pos = r['POS']['top'][0] if r['POS']['top'] else None
        lines.append(f"| {r['category']} | {r['name']} | {r['reviews']:,} | {r['units']:,} | "
                     f"{pct(s['POS'])}/{pct(s['NEG'])}/{pct(s['NONE'])}/{pct(s['REVIEW'])} | "
                     f"{r['NEG']['topics']}({r['NEG']['specific_topics']}) | "
                     f"{neg['label'] if neg else '-'} | {pos['label'] if pos else '-'} |")
    lines += ['', '## 최근 늘어난 불만 (구체적 묶음만)', '']
    for r in rows:
        if r['rising_NEG']:
            items = ', '.join(f"{x['label']} ({x['earlier']} → {x['recent']})" for x in r['rising_NEG'][:3])
            lines.append(f"- **{r['name']}**: {items}")
    (RESULTS / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (RESULTS / 'insights.md').write_text('\n'.join(insights(rows)) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
