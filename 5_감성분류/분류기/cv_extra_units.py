"""Does adding the 200 extra units (4_정답데이터/추가정답, drawn from the 30-product run) help?

Same 5-fold x 3 CV as cv_unit_sentiment.py, variant a (or b with UNIT_VARIANT=b). The extra units get their own folds (by review).
  base  - train on gold train folds only
  extra - train on gold train folds + extra train folds
Both are scored on the gold test fold (no regression?) and the extra test fold (the real 30-product
distribution, where NEG precision was ~55% on 2026-10-24). Each fold result is saved as it finishes, so a
crash-restart resumes. Settings are those of cv_unit_sentiment.py, fixed before results.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from cv_unit_sentiment import build_units, train_and_predict, ENCODER, LABELS, SEED, FOLDS, REPEATS  # noqa: E402
from bibi_reviews.paths import GOLD_DIR  # noqa: E402

EXTRA_DIR = GOLD_DIR / '추가정답'
KOREAN = {'긍정': 'POS', '부정': 'NEG', '평가 아님': 'NONE'}
# UNIT_VARIANT=b gives the classifier the whole review as a second segment (is '마트에선 비싼데' about this product?).
VARIANT = os.environ.get('UNIT_VARIANT', 'a')
SUFFIX = ('' if VARIANT == 'a' else f'-{VARIANT}') + os.environ.get('UNIT_TAG', '')
PARTIAL = OUT / f'cv-extra-units{SUFFIX}-partial.json'


def extra_units():
    # every batch in 추가정답 (candidates.json = 200, candidates-2.json = 600, ...)
    items = [it for f in sorted(EXTRA_DIR.glob('candidates*.json')) for it in json.loads(f.read_text(encoding='utf-8'))['items']
             if it['label'] != '제외']  # 제외: user decided the unit is too ambiguous to learn from
    return [{'unit': it['unit'], 'review_text': it['review'], 'label': LABELS.index(KOREAN[it['label']]),
             'review_id': it['meta']['review_id'], 'model': it['model'].rsplit(' (', 1)[0], 'checked': not it['check']}
            for it in items]


def scores(y, pred):
    y, pred = np.array(y), np.array(pred)
    neg = LABELS.index('NEG')
    return {'accuracy': float((y == pred).mean()), 'macro_f1': float(f1_score(y, pred, average='macro')),
            'per_class_f1': dict(zip(LABELS, map(float, f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)))),
            'NEG_precision': float(precision_score(y == neg, pred == neg, zero_division=0)),
            'NEG_recall': float(recall_score(y == neg, pred == neg, zero_division=0)),
            'confusion': [[int(((y == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)]}


def main():
    start = time.perf_counter()
    gold, _ = build_units()
    extra = extra_units()
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
    n_reviews = 1 + max(r['review'] for r in gold)
    extra_reviews = sorted({r['review_id'] for r in extra})
    partial = json.loads(PARTIAL.read_text(encoding='utf-8')) if PARTIAL.exists() else {}
    for repeat in range(REPEATS):
        order = list(range(n_reviews))
        random.Random(SEED + repeat).shuffle(order)
        fold_of = {i: k for k in range(FOLDS) for i in order[k::FOLDS]}
        gold_fold = np.array([fold_of[r['review']] for r in gold])
        ex_order = extra_reviews[:]
        random.Random(SEED + 1000 + repeat).shuffle(ex_order)
        ex_fold_of = {rid: k for k in range(FOLDS) for rid in ex_order[k::FOLDS]}
        extra_fold = np.array([ex_fold_of[r['review_id']] for r in extra])
        for config in ['base', 'extra']:
            for k in range(FOLDS):
                key = f'{repeat}-{config}-{k}'
                if key in partial:
                    continue
                train = [gold[i] for i in np.where(gold_fold != k)[0]]
                if config == 'extra':
                    train += [extra[i] for i in np.where(extra_fold != k)[0]]
                g_idx, e_idx = np.where(gold_fold == k)[0], np.where(extra_fold == k)[0]
                test = [gold[i] for i in g_idx] + [extra[i] for i in e_idx]
                pred = train_and_predict(tokenizer, train, test, VARIANT, SEED + 100 * repeat + k)
                partial[key] = {'gold_idx': g_idx.tolist(), 'gold_pred': pred[:len(g_idx)],
                                'extra_idx': e_idx.tolist(), 'extra_pred': pred[len(g_idx):]}
                PARTIAL.write_text(json.dumps(partial), encoding='utf-8')
                print(f'{key} done | {(time.perf_counter() - start) / 60:.1f} min', flush=True)
    report = {'encoder': str(ENCODER), 'variant': VARIANT, 'gold_units': len(gold), 'extra_units': len(extra),
              'extra_unchecked_drafts': sum(not r['checked'] for r in extra),
              'note': "extra labels are Claude drafts; items with check=true were not yet confirmed by the user",
              'runs': {}}
    y_gold, y_extra = [r['label'] for r in gold], [r['label'] for r in extra]
    for config in ['base', 'extra']:
        runs = []
        for repeat in range(REPEATS):
            gp, ep = np.empty(len(gold), int), np.empty(len(extra), int)
            for k in range(FOLDS):
                p = partial[f'{repeat}-{config}-{k}']
                gp[p['gold_idx']] = p['gold_pred']
                ep[p['extra_idx']] = p['extra_pred']
            runs.append({'gold': scores(y_gold, gp), 'extra': scores(y_extra, ep)})
        report['runs'][config] = runs
    report['summary'] = {c: {part: {m: round(float(np.mean([r[part][m] for r in report['runs'][c]])), 3)
                                    for m in ['accuracy', 'macro_f1', 'NEG_precision', 'NEG_recall']}
                             for part in ['gold', 'extra']} for c in ['base', 'extra']}
    deployed = [r['model'] for r in extra]  # what the 30-product run predicted for these units
    report['deployed_model_on_extra'] = scores(y_extra, [LABELS.index(KOREAN[m]) if m in KOREAN else LABELS.index('NONE')
                                                         for m in deployed])
    report['seconds'] = time.perf_counter() - start
    (OUT / f'cv-extra-units{SUFFIX}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
