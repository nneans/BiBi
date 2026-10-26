"""Unit-level sentiment (긍정 / 부정 / 평가 아님) on Kiwi units, fine-tuning the food-adapted KcELECTRA.

Units: Kiwi sentence + clause split of the gold text (../review-units-kiwi). Unit labels from gold opinions
covering >= 50% of an opinion inside the unit: only positive -> POS, only negative -> NEG, none or only
facts -> NONE. Excluded: units mixing positive and negative, and units holding 애매/혼합/중립 opinions.
Variants: (a) unit text only, (b) unit + whole review as the second segment.
5-fold CV by review x 3 repeats (same folds as the other CV runs). Settings fixed before results.
"""
import json
import os
from pathlib import Path
import random
import sys
import time

from bibi_reviews.cpu import pin_to_efficiency_cores  # noqa: E402
pin_to_efficiency_cores()  # before numba/torch load: this PC's P-cores are unstable (see bibi_reviews/cpu.py)
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from kiwipiepy import Kiwi

OUT = Path(__file__).resolve().parent
from bibi_reviews.paths import UNITS, BACKBONE_ENCODER  # noqa: E402
sys.path.insert(0, str(UNITS))
from kiwi_units import units, GOLD  # noqa: E402

# Backbone to fine-tune; override with UNIT_ENCODER=<model dir> and UNIT_TAG=<name> to compare backbones.
ENCODER = Path(os.environ.get('UNIT_ENCODER', BACKBONE_ENCODER))  # Stage 1 v3 epoch 1 (2026-10-23)
TAG = os.environ.get('UNIT_TAG', '')
VARIANTS = os.environ.get('UNIT_VARIANTS', 'a,b').split(',')
LABELS = ['POS', 'NEG', 'NONE']
SEED, FOLDS, REPEATS = 20261022, 5, 3
EPOCHS, LR, BATCH, WARMUP = 5, 3e-5, 16, 0.1
CONCESSIVE = ('는데', '은데', '인데', '지만', '한데', '던데')


def build_units():
    kiwi = Kiwi()
    rows, excluded = [], {'mixed_pos_neg': 0, 'ambiguous_or_neutral': 0}
    for index, r in enumerate(json.loads(GOLD.read_text(encoding='utf-8'))['reviews']):
        text = r['text']
        ops = [((text.index(o['evidence']), text.index(o['evidence']) + len(o['evidence'])), o['sentiment']) for o in r['opinions']]
        for s, e in units(kiwi, text, True):
            # v2 rule: a unit takes an opinion's sentiment if it holds >= 50% of the opinion OR >= 50% of the unit
            # lies inside the opinion. v1 used only the first test, so units cut out of one long gold opinion
            # (e.g. '팔다니 악덕판매자네요') were wrongly labelled NONE.
            inside = {sent for (a, b), sent in ops
                      if (ov := max(0, min(e, b) - max(s, a))) / (b - a) >= 0.5 or ov / (e - s) >= 0.5}
            if inside & {'애매', '혼합', '중립'}:
                excluded['ambiguous_or_neutral'] += 1
                continue
            if {'긍정', '부정'} <= inside:
                excluded['mixed_pos_neg'] += 1
                continue
            label = 'POS' if '긍정' in inside else 'NEG' if '부정' in inside else 'NONE'
            unit = text[s:e]
            rows.append({'review': index, 'unit': unit, 'review_text': text, 'label': LABELS.index(label),
                         'concessive': unit.rstrip(' ,.~!?ㅠㅜ').endswith(CONCESSIVE)})
    return rows, excluded


def encode(tokenizer, rows, variant):
    if variant == 'a':
        return tokenizer([r['unit'] for r in rows], truncation=True, max_length=128, padding=True, return_tensors='pt')
    return tokenizer([r['unit'] for r in rows], [r['review_text'] for r in rows], truncation='only_second',
                     max_length=256, padding=True, return_tensors='pt')


def train_and_predict(tokenizer, train_rows, test_rows, variant, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(str(ENCODER), num_labels=len(LABELS), local_files_only=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(train_rows) + BATCH - 1) // BATCH)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * steps), steps)
    rng = random.Random(seed)
    model.train()
    for _ in range(EPOCHS):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for k in range(0, len(order), BATCH):
            batch = [train_rows[i] for i in order[k:k + BATCH]]
            enc = {n: t.cuda() for n, t in encode(tokenizer, batch, variant).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**enc, labels=torch.tensor([r['label'] for r in batch]).cuda()).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()
    preds = []
    with torch.inference_mode():
        for k in range(0, len(test_rows), 64):
            enc = {n: t.cuda() for n, t in encode(tokenizer, test_rows[k:k + 64], variant).items()}
            preds += model(**enc).logits.argmax(-1).tolist()
    del model
    torch.cuda.empty_cache()
    return preds


def main():
    start = time.perf_counter()
    rows, excluded = build_units()
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
    n_reviews = 1 + max(r['review'] for r in rows)
    y = np.array([r['label'] for r in rows])
    conc = np.array([r['concessive'] for r in rows])
    report = {'encoder': str(ENCODER), 'units': len(rows), 'excluded': excluded, 'label_counts': {LABELS[i]: int((y == i).sum()) for i in range(3)},
              'concessive_units': int(conc.sum()), 'settings': {'epochs': EPOCHS, 'lr': LR, 'batch': BATCH, 'warmup': WARMUP},
              'runs': []}
    print(json.dumps({k: report[k] for k in ['units', 'excluded', 'label_counts', 'concessive_units']}, ensure_ascii=False), flush=True)
    total = REPEATS * len(VARIANTS) * FOLDS
    done = 0
    for repeat in range(REPEATS):
        order = list(range(n_reviews))
        random.Random(SEED + repeat).shuffle(order)
        fold_of = {i: k for k in range(FOLDS) for i in order[k::FOLDS]}
        folds = np.array([fold_of[r['review']] for r in rows])
        for variant in VARIANTS:
            pred = np.empty_like(y)
            for k in range(FOLDS):
                train = [rows[i] for i in np.where(folds != k)[0]]
                test_idx = np.where(folds == k)[0]
                pred[test_idx] = train_and_predict(tokenizer, train, [rows[i] for i in test_idx], variant, SEED + 100 * repeat + k)
                done += 1
                print(f'[{done}/{total}] repeat {repeat} variant {variant} fold {k} | {(time.perf_counter() - start) / 60:.1f} min', flush=True)
            per_class = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
            run = {'repeat': repeat, 'variant': variant, 'accuracy': float((pred == y).mean()),
                   'macro_f1': float(f1_score(y, pred, average='macro')),
                   'per_class_f1': {LABELS[i]: float(per_class[i]) for i in range(3)},
                   'concessive_accuracy': float((pred[conc] == y[conc]).mean()),
                   'confusion': [[int(((y == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)]}
            report['runs'].append(run)
            print(json.dumps(run), flush=True)
    report['summary'] = {v: {m: float(np.mean([r[m] for r in report['runs'] if r['variant'] == v]))
                             for m in ['accuracy', 'macro_f1', 'concessive_accuracy']} for v in VARIANTS}
    report['seconds'] = time.perf_counter() - start
    (OUT / ('cv-unit-sentiment-v2labels' + (f'-{TAG}' if TAG else '') + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
