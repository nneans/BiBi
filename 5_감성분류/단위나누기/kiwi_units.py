"""How well do rule-based Kiwi units match the gold opinion spans? (no training)

A: Kiwi sentence split only.  B: A + split after contrast/listing connective endings (fixed before results).
Scored on the gold text (normalize_text output, where gold offsets live):
  found  - gold opinion has a unit with char IoU >= 0.5
  mixed  - unit overlapping >= 2 gold opinions that differ in (aspect, sentiment)
  split  - gold opinion whose best unit covers < 50% of it
"""
import json
from pathlib import Path

from kiwipiepy import Kiwi

from bibi_reviews.paths import GOLD  # noqa: E402
OUT = Path(__file__).resolve().parent
# Kiwi returns the final-consonant jamo U+11AB ('ᆫ데'), not the compatibility jamo 'ㄴ' (U+3134);
# v1 of this rule used only 'ㄴ데' and never matched. Both spellings are listed now.
CLAUSE_EC = {'는데', '은데', 'ㄴ데', 'ᆫ데', '던데', '지만', '고', '면서', '으면서'}
COLLOQUIAL_EC = {'구'}  # added after seeing results (하구 = 하고); reported as a separate variant
# Kiwi sometimes reads '맛있는데' as ETM '는' + NNB '데' instead of EC '는데'. Treat it as the connective when
# '데' is glued to the ending and not followed by a particle or copula ('먹는 데에', '하는 데이다' stay whole).
NOUN_DE_ETM = {'는', '은', 'ㄴ', 'ᆫ', '던'}


def is_clause_end(toks, i, endings):
    tok = toks[i]
    nxt = toks[i + 1] if i + 1 < len(toks) else None
    if tok.tag == 'EC' and tok.form in endings:
        # '먹고 있어요', '사고 싶어요' (auxiliary verb) and '말고는', '먹고도' (particle) are one clause, not two;
        # cutting them left fragments like '있습니다.' (seen in the 2026-10-24 extra-label sample).
        return not (nxt and (nxt.tag.startswith('VX') or nxt.tag.startswith('J')))
    if tok.tag != 'NNB' or tok.form != '데' or i == 0:
        return False
    prev = toks[i - 1]
    return (prev.tag == 'ETM' and prev.form in NOUN_DE_ETM and prev.start + prev.len == tok.start
            and not (nxt and (nxt.tag.startswith('J') or nxt.tag == 'VCP')))


def units(kiwi, text, clauses, endings=None):
    endings = endings or CLAUSE_EC
    spans = []
    for sent in kiwi.split_into_sents(text):
        cuts = [sent.start]
        if clauses:
            toks = kiwi.tokenize(sent.text)
            for i, tok in enumerate(toks):
                end = sent.start + tok.start + tok.len
                if is_clause_end(toks, i, endings) and end < sent.end:
                    cuts.append(end)
        cuts.append(sent.end)
        for s, e in zip(cuts, cuts[1:]):
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                spans.append((s, e))
    return join_dangling(kiwi, text, spans) if clauses else spans


PREDICATE_TAGS = ('VV', 'VA', 'VX', 'VCP', 'VCN', 'XR', 'EF', 'EC', 'ETM', 'ETN', 'MAG', 'IC')


def dangling(kiwi, piece):
    """A piece with no predicate that stops on a particle ('옆지느러미가.', '맛이.', '가성비도.', '여름엔') is the
    start of the next unit, cut off by a stray full stop. Short verdicts with no particle ('굿', '추천') are kept."""
    toks = [t for t in kiwi.tokenize(piece) if not t.tag.startswith('S')]
    return bool(toks) and toks[-1].tag.startswith('J') and not any(t.tag.startswith(PREDICATE_TAGS) for t in toks)


def join_dangling(kiwi, text, spans):
    out = []
    for s, e in spans:
        if out and out[-1][2]:  # previous piece was dangling: this one completes it
            s = out.pop()[0]
        out.append((s, e, dangling(kiwi, text[s:e])))
    return [(s, e) for s, e, _ in out]


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def score(kiwi, reviews, clauses, endings=None):
    found = split = mixed = n_units = n_ops = 0
    examples = {'mixed': [], 'missed': []}
    for r in reviews:
        text = r['text']
        us = units(kiwi, text, clauses, endings)
        n_units += len(us)
        ops = []
        for o in r['opinions']:
            s = text.index(o['evidence'])
            ops.append(((s, s + len(o['evidence'])), (o['aspect'], o['sentiment']), o['evidence']))
        for span, _, ev in ops:
            n_ops += 1
            best = max(us, key=lambda u: iou(u, span))
            if iou(best, span) >= 0.5:
                found += 1
            elif len(examples['missed']) < 8:
                examples['missed'].append({'opinion': ev, 'best_unit': text[best[0]:best[1]], 'review': text})
            cover = max(max(0, min(u[1], span[1]) - max(u[0], span[0])) for u in us) / (span[1] - span[0])
            split += cover < 0.5
        for u in us:
            inside = {key for span, key, _ in ops if max(0, min(u[1], span[1]) - max(u[0], span[0])) / (span[1] - span[0]) >= 0.5}
            if len(inside) >= 2:
                mixed += 1
                if len(examples['mixed']) < 8:
                    examples['mixed'].append({'unit': text[u[0]:u[1]], 'opinions': sorted(f'{a}/{s}' for a, s in inside)})
    return {'gold_opinions': n_ops, 'units': n_units, 'found_iou50': found / n_ops, 'split_opinions': split / n_ops,
            'mixed_units': mixed, 'mixed_unit_share': mixed / n_units, 'examples': examples}


def main():
    kiwi = Kiwi()
    reviews = json.loads(GOLD.read_text(encoding='utf-8'))['reviews']
    report = {'gold': str(GOLD), 'clause_endings': sorted(CLAUSE_EC),
              'A_sentences': score(kiwi, reviews, False), 'B_sentences_plus_clauses': score(kiwi, reviews, True),
              'C_plus_colloquial_gu': score(kiwi, reviews, True, CLAUSE_EC | COLLOQUIAL_EC),
              'reference': 'HEAD 1 CRF (trained, 240 reviews per fold): span+polarity IoU50 F1 0.722 in food-gold-cv'}
    (OUT / 'kiwi-units.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for k in ['A_sentences', 'B_sentences_plus_clauses', 'C_plus_colloquial_gu']:
        r = report[k]
        print(f"{k}: units {r['units']} | found {r['found_iou50']:.3f} | split {r['split_opinions']:.3f} | mixed units {r['mixed_units']} ({r['mixed_unit_share']:.3f})")


if __name__ == '__main__':
    main()
