"""Learning curve for the unit sentiment classifier (variant a): is it data-limited?

Same folds and hyperparameters as cv_unit_sentiment.py. Within each fold, train on 60/120/180/240 of the
training reviews (seeded subset) and score both the held-out fold and the training units themselves.
"""
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from cv_unit_sentiment import build_units, encode, ENCODER, LABELS, SEED, FOLDS, EPOCHS, LR, BATCH, WARMUP, OUT

SIZES = [60, 120, 180, 240]
REPEATS = 2


def fit(tokenizer, rows, seed):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(str(ENCODER), num_labels=len(LABELS), local_files_only=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(rows) + BATCH - 1) // BATCH)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * steps), steps)
    rng = random.Random(seed)
    model.train()
    for _ in range(EPOCHS):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for k in range(0, len(order), BATCH):
            batch = [rows[i] for i in order[k:k + BATCH]]
            enc = {n: t.cuda() for n, t in encode(tokenizer, batch, 'a').items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**enc, labels=torch.tensor([r['label'] for r in batch]).cuda()).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()
    return model


@torch.inference_mode()
def predict(model, tokenizer, rows):
    out = []
    for k in range(0, len(rows), 64):
        enc = {n: t.cuda() for n, t in encode(tokenizer, rows[k:k + 64], 'a').items()}
        out += model(**enc).logits.argmax(-1).tolist()
    return np.array(out)


def main():
    start = time.perf_counter()
    rows, _ = build_units()
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
    n_reviews = 1 + max(r['review'] for r in rows)
    y = np.array([r['label'] for r in rows])
    runs, done, total = [], 0, REPEATS * len(SIZES) * FOLDS
    for repeat in range(REPEATS):
        order = list(range(n_reviews))
        random.Random(SEED + repeat).shuffle(order)
        folds = [order[k::FOLDS] for k in range(FOLDS)]
        for size in SIZES:
            pred = np.empty_like(y)
            train_acc = []
            train_units = []
            for k, test_reviews in enumerate(folds):
                pool = [i for j, f in enumerate(folds) if j != k for i in f]
                chosen = set(random.Random(SEED + 1000 * repeat + 10 * k + size).sample(pool, min(size, len(pool))))
                train = [r for r in rows if r['review'] in chosen]
                test_idx = [i for i, r in enumerate(rows) if r['review'] in set(test_reviews)]
                model = fit(tokenizer, train, SEED + 100 * repeat + k)
                pred[test_idx] = predict(model, tokenizer, [rows[i] for i in test_idx])
                tp = predict(model, tokenizer, train)
                train_acc.append(float((tp == np.array([r['label'] for r in train])).mean()))
                train_units.append(len(train))
                del model
                torch.cuda.empty_cache()
                done += 1
                print(f'[{done}/{total}] repeat {repeat} size {size} fold {k} | {(time.perf_counter() - start) / 60:.1f} min', flush=True)
            per_class = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
            run = {'repeat': repeat, 'train_reviews': size, 'train_units_mean': float(np.mean(train_units)),
                   'heldout_accuracy': float((pred == y).mean()), 'heldout_macro_f1': float(f1_score(y, pred, average='macro')),
                   'per_class_f1': {LABELS[i]: float(per_class[i]) for i in range(3)}, 'train_accuracy': float(np.mean(train_acc))}
            runs.append(run)
            print(json.dumps(run), flush=True)
    summary = {}
    for size in SIZES:
        rs = [r for r in runs if r['train_reviews'] == size]
        summary[size] = {'train_units': round(np.mean([r['train_units_mean'] for r in rs])),
                         **{m: round(float(np.mean([r[m] for r in rs])), 4) for m in ['heldout_accuracy', 'heldout_macro_f1', 'train_accuracy']},
                         'per_class_f1': {c: round(float(np.mean([r['per_class_f1'][c] for r in rs])), 4) for c in LABELS}}
    report = {'protocol': f'{FOLDS}-fold CV by review x {REPEATS} repeats; training subsets of the 4 training folds',
              'sizes_train_reviews': SIZES, 'summary': summary, 'runs': runs, 'seconds': time.perf_counter() - start}
    (OUT / 'learning-curve.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for size, s in summary.items():
        print(f"{size:3d} reviews (~{s['train_units']} units): held-out acc {s['heldout_accuracy']:.3f} macroF1 {s['heldout_macro_f1']:.3f} "
              f"{s['per_class_f1']} | train acc {s['train_accuracy']:.3f}")


if __name__ == '__main__':
    main()
