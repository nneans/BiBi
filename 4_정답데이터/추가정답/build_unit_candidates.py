"""Pick ~200 units from the 30-product run for extra sentiment labels (unit level: 긍정 / 부정 / 평가 아님).

Pools target what the classifier got wrong on 2026-10-24: praise written with a negation ('마트 초밥 살 필요 없어요'),
listed good points ending in a connective ('불향도 나고'), low-confidence units, and plain NEG / NONE for a
precision check. Units are spread round-robin over the products; gold-300 reviews and repeated texts are skipped.
Usage: python build_unit_candidates.py [batch]   -> candidates.json (batch 1) or candidates-<batch>.json
Batch 2 (600 units) leans on predicted NEG, the main error (~45% of NEG units were not complaints in batch 1),
and adds POS for a first precision check; texts already in earlier batches are skipped.
"""
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bibi_reviews.paths import CANONICAL_REVIEWS, GOLD, REASONS  # noqa: E402
from bibi_reviews.backbone import normalize_text  # noqa: E402

SEED = 20261024
NEGATION = re.compile(r'없|안\s|않|못|아니|아녀|아님')
LISTING_END = re.compile(r'(고|며|면서|구)[\s.,!~^ㅎㅋ]*$')
POOLS_1 = [  # (name, size, predicate on a unit row)
    ('부정+부정표현', 45, lambda r: r['label'] == 'NEG' and NEGATION.search(r['unit'])),
    ('부정+나열어미', 35, lambda r: r['label'] == 'NEG' and LISTING_END.search(r['unit']) and not NEGATION.search(r['unit'])),
    ('재확인', 50, lambda r: r['label'] == 'REVIEW'),
    ('평가아님', 35, lambda r: r['label'] == 'NONE'),
    ('부정', 35, lambda r: r['label'] == 'NEG'),
]
POOLS_2 = [
    ('부정+부정표현', 100, lambda r: r['label'] == 'NEG' and NEGATION.search(r['unit'])),
    ('부정', 300, lambda r: r['label'] == 'NEG'),
    ('재확인', 100, lambda r: r['label'] == 'REVIEW'),
    ('긍정', 60, lambda r: r['label'] == 'POS'),
    ('평가아님', 40, lambda r: r['label'] == 'NONE'),
]
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
POOLS = POOLS_1 if BATCH == 1 else POOLS_2
TARGET = OUT / ('candidates.json' if BATCH == 1 else f'candidates-{BATCH}.json')
KOREAN = {'POS': '긍정', 'NEG': '부정', 'NONE': '평가 아님'}


def main():
    rng = random.Random(SEED + BATCH - 1)
    products = json.loads((REASONS / 'products.json').read_text(encoding='utf-8'))
    gold_ids = {r['review_id'] for r in json.loads(GOLD.read_text(encoding='utf-8'))['reviews']}
    units = {}
    for p in products:
        rows = [json.loads(line) for line in (REASONS / 'results' / p['id'] / 'units.jsonl').read_text(encoding='utf-8').splitlines()]
        rows = [r for r in rows if r['review_id'] not in gold_ids and len(re.sub(r'\s', '', r['unit'])) >= 4]
        rng.shuffle(rows)
        units[p['id']] = rows
    seen, picked = set(), []
    for earlier in sorted(OUT.glob('candidates*.json')):
        if earlier != TARGET:
            seen |= {re.sub(r'\s+', ' ', it['unit']).strip() for it in json.loads(earlier.read_text(encoding='utf-8'))['items']}
    for pool, size, keep in POOLS:
        queues = {pid: [r for r in rows if keep(r)] for pid, rows in units.items()}
        order = [p['id'] for p in products]
        rng.shuffle(order)
        got = 0
        while got < size and any(queues.values()):
            for pid in order:
                while queues[pid]:
                    r = queues[pid].pop()
                    key = re.sub(r'\s+', ' ', r['unit']).strip()
                    if key not in seen:
                        seen.add(key)
                        picked.append({**r, 'product_id': pid, 'pool': pool})
                        got += 1
                        break
                if got >= size:
                    break
    ids = sorted({r['review_id'] for r in picked})
    table = pq.read_table(CANONICAL_REVIEWS, columns=['review_id', 'review_text_raw'])
    table = table.filter(pc.is_in(table['review_id'], value_set=pa_array(ids)))
    text = {r['review_id']: normalize_text(r['review_text_raw']) for r in table.to_pylist()}
    names = {p['id']: p['name'] for p in products}
    rng.shuffle(picked)
    items = [{'id': i + 1, 'product': names[r['product_id']], 'unit': r['unit'], 'review': text[r['review_id']],
              'model': f"{KOREAN[r['pred']]} ({r['confidence']:.2f})", 'label': '',
              'meta': {'pool': r['pool'], 'product_id': r['product_id'], 'review_id': r['review_id'], 'month': r['month']}}
             for i, r in enumerate(picked)]
    out = {'guide': "label에 '긍정' / '부정' / '평가 아님' 중 하나. 단위(unit)만 보고 판단하고, review는 앞뒤 맥락 확인용.",
           'items': items}
    TARGET.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(len(items), {pool: sum(r['pool'] == pool for r in picked) for pool, _, _ in POOLS},
          'products', len({r['product_id'] for r in picked}))


def pa_array(values):
    import pyarrow as pa
    return pa.array(values, type=pa.string())


if __name__ == '__main__':
    main()
