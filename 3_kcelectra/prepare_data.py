"""Stage 1 v3 data: every eligible review (not a 50k sample), on the cleaned text.

Same rules as v2: one canonical copy per text_hash, families = original/follow-up links only, family-hash
80/10/10 partition, prior-100 experiment excluded by direct links. Changes from v2:
  - text = review_text (normalize_text output), not review_text_raw
  - train = all train-partition candidates; validation/test = 2,000 diverse reviews each (as in v2)
  - the 300 gold reviews (their families and identical texts) are excluded too, since they score the
    downstream sentiment classifier
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from rtd_core import review_families, canonical_text_rows, diverse_sample
from bibi_reviews.paths import BASE_KCELECTRA, CANONICAL_REVIEWS, GOLD as GOLD_PATH

OUT = Path(__file__).resolve().parent
BASE = BASE_KCELECTRA
SEED = 20261022
GOLD = GOLD_PATH
HELDOUT = {'validation': 2000, 'test': 2000}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out=OUT):
    """out: where the splits and sampling-report.json go (default: this folder; the prior-100 list is read from here)."""
    source = CANONICAL_REVIEWS
    fields = ['review_id', 'product_id', 'review_text', 'text_hash', 'review_group_id', 'related_review_id']
    rows = pq.read_table(source, columns=fields).to_pylist()
    assert len({r['review_id'] for r in rows}) == len(rows)
    families = review_families(rows)
    prior_path = OUT / 'prior-100-dataset.json'
    protected_ids = {r['review_id'] for r in json.loads(prior_path.read_text(encoding='utf-8'))['reviews']}
    gold_ids = {r['review_id'] for r in json.loads(GOLD.read_text(encoding='utf-8'))['reviews']}
    assert len(protected_ids) == 100 and len(gold_ids) == 300
    blocked_ids = protected_ids | gold_ids
    blocked_families = {families[i] for i in blocked_ids}
    blocked_texts = {r['text_hash'] for r in rows if r['review_id'] in blocked_ids}
    canonical = canonical_text_rows(rows, SEED)
    excluded = Counter()
    kept = []
    for index, row in enumerate(rows):
        family = families[row['review_id']]
        if family in blocked_families or row['text_hash'] in blocked_texts:
            excluded['prior_100_or_gold_300_direct_link'] += 1
        elif index not in canonical:
            excluded['duplicate_text_copy'] += 1
        else:
            kept.append(row | {'family': family, 'source_row': index})
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    candidates = {s: [] for s in ['train', 'validation', 'test']}
    for start in range(0, len(kept), 4096):
        chunk = kept[start:start + 4096]
        encoded = tokenizer([r['review_text'] for r in chunk], truncation=False, return_attention_mask=False,
                            return_token_type_ids=False, verbose=False)['input_ids']
        for row, ids in zip(chunk, encoded):
            if not 2 < len(ids) <= 512:
                excluded['outside_whole_review_length_limit'] += 1
                continue
            assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
            partition = int(hashlib.sha256(f'{SEED}:{row["family"]}'.encode()).hexdigest()[:12], 16) % 10000
            split = 'train' if partition < 8000 else 'validation' if partition < 9000 else 'test'
            candidates[split].append(row | {'input_ids': ids})
        if start // 4096 % 25 == 0:
            print(f'tokenized {start + len(chunk)}/{len(kept)}', flush=True)
    print(json.dumps({'population': len(rows), 'candidate_reviews': {k: len(v) for k, v in candidates.items()},
                      'excluded': dict(excluded)}), flush=True)
    selected = {}
    for i, (split, count) in enumerate(HELDOUT.items(), 1):
        selected[split] = diverse_sample(candidates[split], count, SEED + i)
    # Held-out partitions supply only 2,000 reviews each; their other reviews join training unless they share
    # a family with a held-out review.
    heldout_families = {r['family'] for split in HELDOUT for r in selected[split]}
    recycled = [r for split in HELDOUT for r in candidates[split] if r['family'] not in heldout_families]
    selected = {'train': candidates['train'] + recycled, **selected}
    excluded['heldout_family_partner_not_used'] = sum(len(candidates[s]) for s in HELDOUT) - len(recycled) - sum(HELDOUT.values())
    counts, split_families = {}, {}
    for split, part in selected.items():
        path = out / f'{split}.jsonl'
        content_tokens = unknown = 0
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for row in part:
                record = {k: row[k] for k in ['review_id', 'product_id', 'family', 'source_row', 'text_hash', 'review_group_id', 'related_review_id']}
                record.update(text=row['review_text'], input_ids=row['input_ids'], split=split,
                              raw_sha256=hashlib.sha256(row['review_text'].encode()).hexdigest())
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                content_tokens += len(row['input_ids']) - 2
                unknown += row['input_ids'].count(tokenizer.unk_token_id)
        products = Counter(r['product_id'] for r in part)
        counts[split] = {'reviews': len(part), 'products': len(products), 'content_tokens': content_tokens,
                         'unknown_tokens': unknown, 'min_reviews_per_product': min(products.values()),
                         'max_reviews_per_product': max(products.values()), 'file_sha256': sha(path)}
        split_families[split] = {r['family'] for r in part}
    assert not split_families['train'] & split_families['validation']
    assert not split_families['train'] & split_families['test']
    assert not split_families['validation'] & split_families['test']
    assert not set.union(*split_families.values()) & blocked_families
    texts = [r['text_hash'] for part in selected.values() for r in part]
    assert len(texts) == len(set(texts)) and not set(texts) & blocked_texts
    report = {'seed': SEED, 'source_path': str(source), 'source_sha256': sha(source), 'text_column': 'review_text',
              'population_reviews': len(rows), 'population_products': len({r['product_id'] for r in rows}),
              'excluded_reviews': dict(excluded), 'prior_dataset_sha256': sha(prior_path), 'gold_sha256': sha(GOLD),
              'sampling': 'validation/test = diverse_sample of 2,000 each from their family partitions; train = every other '
                          'eligible review (train partition + held-out partitions minus held-out families)',
              'family_rule': 'review_group_id + related_review_id only; identical text deduplicated, never chained',
              'tokenizer_path': str(BASE), 'vocab_sha256': sha(BASE / 'vocab.txt'), 'counts': counts}
    (out / 'sampling-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({s: {k: v for k, v in c.items() if k != 'file_sha256'} for s, c in counts.items()}), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
