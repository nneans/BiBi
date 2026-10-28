"""What does the analysis text (bibi_reviews.text_clean, v2) change, and does it keep the units right?

1. Every normal review: how many reviews each rule touches, with examples (raw -> v2).
2. Line breaks: how many are read as sentence ends vs wrapped lines, with random examples of both.
3. The 300 gold reviews: units on today's text (normalize_text) vs units on v2 text, scored the same way
   (5_감성분류/단위나누기/kiwi_units.score). Gold evidence lives on the normalize_text text, so its offsets are
   carried over to the v2 text by character alignment (difflib).
4. Copy-pasted reviews: the same v2 text (40+ characters) more than once in one product.
Output: output/checks/clean-report.json
"""
import difflib
import json
import random
import re
import sys
from collections import Counter, defaultdict

from bibi_reviews.cpu import pin_to_efficiency_cores
pin_to_efficiency_cores()  # this PC's P-cores are unstable (see bibi_reviews/cpu.py)
import pyarrow.parquet as pq  # noqa: E402
from kiwipiepy import Kiwi  # noqa: E402

from bibi_reviews import text_clean as C  # noqa: E402
from bibi_reviews.backbone import normalize_text  # noqa: E402
from bibi_reviews.paths import CANONICAL_REVIEWS, GOLD, PROCESSED, UNITS  # noqa: E402

sys.path.insert(0, str(UNITS))
from kiwi_units import score, is_line_end  # noqa: E402

OUT = PROCESSED / 'checks'
SEED = 20261028
COPY_MIN_CHARS = 40
RULES = {'invisible': C.INVISIBLE, 'old_dots': re.compile('[ᆢᆞ]'), 'link': C.LINK,
         'long_digits': C.LONG_DIGITS, 'spaced_dots': re.compile(r'\.(?:[ \t]+\.){2,}'), 'repeat': C.REPEAT,
         'line_break': re.compile(r'\n')}


def carry(old, new):
    """Offset map old text -> new text from a character alignment."""
    to_new = [0] * (len(old) + 1)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        for k in range(i1, i2):
            to_new[k] = j1 + (k - i1 if tag == 'equal' else 0)
        to_new[i2] = j2
    return to_new


def gold_check(kiwi, raw_of):
    reviews = json.loads(GOLD.read_text(encoding='utf-8'))['reviews']
    v2, lost = [], []
    for r in reviews:
        old, new = r['text'], C.clean_review(raw_of[r['review_id']])
        to_new = carry(old, new)
        ops = []
        for o in r['opinions']:
            s = old.index(o['evidence'])
            a, b = to_new[s], to_new[s + len(o['evidence'])]
            ev = new[a:b].strip()
            if not ev:
                lost.append(o['evidence'])
                continue
            ops.append({**o, 'evidence': ev})
        v2.append({'text': new, 'opinions': ops, 'changed': new != old, 'has_break': '\n' in new})
    today, after = score(kiwi, reviews, True), score(kiwi, v2, True)
    changed = [i for i, r in enumerate(v2) if r['changed']]
    sub_today = score(kiwi, [reviews[i] for i in changed], True) if changed else None
    sub_after = score(kiwi, [v2[i] for i in changed], True) if changed else None
    keys = ['gold_opinions', 'units', 'found_iou50', 'split_opinions', 'mixed_units', 'mixed_unit_share']
    return {'today_normalize_text': {k: today[k] for k in keys}, 'v2': {k: after[k] for k in keys},
            'reviews_whose_text_changed': len(changed), 'reviews_with_line_breaks': sum(r['has_break'] for r in v2),
            'changed_reviews_only': {'today': {k: sub_today[k] for k in keys}, 'v2': {k: sub_after[k] for k in keys},
                                     'v2_examples': sub_after['examples']} if changed else None,
            'evidence_lost_in_alignment': lost}


def main():
    kiwi = Kiwi()
    rows = pq.read_table(CANONICAL_REVIEWS, columns=['review_id', 'product_id', 'review_text_raw']).to_pylist()
    touched, examples, changed = Counter(), defaultdict(list), 0
    breaks, by_product = [], defaultdict(Counter)
    rng = random.Random(SEED)
    for r in rows:
        raw = r['review_text_raw'] or ''
        new = C.clean_review(raw)
        changed += new != normalize_text(raw)
        for name, rule in RULES.items():
            m = rule.search(raw.replace('\r', '') if name == 'line_break' else raw)
            if m:
                touched[name] += 1
                if len(examples[name]) < 5 and rng.random() < 0.05:
                    examples[name].append({'raw': raw[max(0, m.start() - 30):m.start() + 40], 'v2': new[:160]})
        if '\n' in new and rng.random() < 0.02:
            breaks.append(new)
        if len(new) >= COPY_MIN_CHARS:
            by_product[r['product_id']][new] += 1
    line_ends, wraps = [], []
    for text in breaks:
        start = 0
        for m in re.finditer('\n', text):
            line = text[start:m.start()]
            (line_ends if is_line_end(kiwi, line) else wraps).append(f'{line[-25:]} ⏎ {text[m.end():m.end() + 20]}')
            start = m.end()
    copies = sorted(((n, p, t[:80]) for p, c in by_product.items() for t, n in c.items() if n > 1), reverse=True)
    raw_of = {r['review_id']: r['review_text_raw'] for r in rows}
    report = {'clean_version': C.CLEAN_VERSION, 'reviews': len(rows),
              'reviews_whose_text_differs_from_normalize_text': changed,
              'reviews_touched_per_rule': dict(touched), 'examples': examples,
              'line_breaks_sampled': {'reviews': len(breaks), 'sentence_end': len(line_ends), 'wrapped_line': len(wraps),
                                      'sentence_end_examples': rng.sample(line_ends, min(25, len(line_ends))),
                                      'wrapped_line_examples': rng.sample(wraps, min(25, len(wraps)))},
              'copies_in_one_product': {'min_chars': COPY_MIN_CHARS, 'texts_repeated': len(copies),
                                        'extra_copies': sum(n - 1 for n, _, _ in copies),
                                        'top': [{'copies': n, 'product_id': p, 'text': t} for n, p, t in copies[:15]]},
              'gold_units': gold_check(kiwi, raw_of)}
    OUT.mkdir(exist_ok=True)
    (OUT / 'clean-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    g = report['gold_units']
    print(json.dumps({k: report[k] for k in ['reviews_whose_text_differs_from_normalize_text', 'reviews_touched_per_rule']}, ensure_ascii=False))
    print(json.dumps({k: report['line_breaks_sampled'][k] for k in ['reviews', 'sentence_end', 'wrapped_line']}))
    print(json.dumps({k: g[k] for k in ['today_normalize_text', 'v2', 'reviews_whose_text_changed', 'reviews_with_line_breaks']}, ensure_ascii=False))
    if g['changed_reviews_only']:
        print(json.dumps({k: g['changed_reviews_only'][k] for k in ['today', 'v2']}, ensure_ascii=False))
    print('evidence lost', g['evidence_lost_in_alignment'])
    print(json.dumps(report['copies_in_one_product']['extra_copies']))


if __name__ == '__main__':
    main()
