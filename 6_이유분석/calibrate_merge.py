"""Where does KURE-v1 similarity mean "same opinion"? Calibrated on the 148 hand-judged pairs (4_정답데이터/같은뜻판정).

Prints AUC and, per threshold, precision/recall of "same" (O); picks the threshold with the best F1 among those with
precision >= 0.8, because a wrong merge hides a complaint inside another while a missed merge only splits it.
Usage: python calibrate_merge.py   -> merge-threshold.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / 'src'))
from bibi_reviews.paths import SAME_MEANING, KURE_V1  # noqa: E402

MIN_PRECISION = 0.8


def main():
    pairs = [p for p in json.loads(SAME_MEANING.read_text(encoding='utf-8'))['pairs'] if p['같은뜻'] in ('O', 'X')]
    model = SentenceTransformer(str(KURE_V1), device='cuda', local_files_only=True)
    a = model.encode([p['a'] for p in pairs], normalize_embeddings=True)
    b = model.encode([p['b'] for p in pairs], normalize_embeddings=True)
    sim = (a * b).sum(1)
    y = np.array([p['같은뜻'] == 'O' for p in pairs])
    rows = []
    for t in np.arange(0.50, 0.96, 0.01):
        pred = sim >= t
        tp = int((pred & y).sum())
        prec = tp / max(pred.sum(), 1)
        rec = tp / y.sum()
        rows.append({'threshold': round(float(t), 2), 'precision': round(prec, 3), 'recall': round(rec, 3),
                     'f1': round(2 * prec * rec / max(prec + rec, 1e-9), 3), 'merged_pairs': int(pred.sum())})
    ok = [r for r in rows if r['precision'] >= MIN_PRECISION and r['merged_pairs'] >= 5]
    best = max(ok, key=lambda r: (r['f1'], -r['threshold']))
    report = {'pairs': len(pairs), 'same': int(y.sum()), 'auc': round(float(roc_auc_score(y, sim)), 3),
              'rule': f'best F1 with precision >= {MIN_PRECISION}', 'chosen': best, 'table': rows}
    (OUT / 'merge-threshold.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['pairs', 'same', 'auc', 'chosen']}, ensure_ascii=False))
    for r in rows[::5]:
        print(r)


if __name__ == '__main__':
    main()
