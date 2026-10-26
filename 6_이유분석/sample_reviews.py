"""Readable per-review check: original text, then every unit with its sentiment and confidence.

Per product it picks mixed reviews (praise and complaint units), low-rated ones (1-2 stars), and a few plain random
ones, so both the hard and the ordinary cases are visible.
From run v3 on, each praise/complaint unit also shows the group it was put in (keywords + central sentence).
Usage: python sample_reviews.py [run] [per_product]   -> results-<run>/review-samples.md (default v3, 4)
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / 'src'))
from bibi_reviews.paths import CANONICAL_REVIEWS  # noqa: E402
from bibi_reviews.backbone import normalize_text  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'v3'
PER_PRODUCT = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RESULTS = OUT / f'results-{RUN}'
KOREAN = {'POS': '긍정', 'NEG': '**부정**', 'NONE': '평가 아님', 'REVIEW': '_재확인_'}
SEED = 20261025


def main():
    rng = random.Random(SEED)
    products = json.loads((OUT / 'products.json').read_text(encoding='utf-8'))
    picked = {}
    for p in products:
        by_review = defaultdict(list)
        for line in (RESULTS / p['id'] / 'units.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            by_review[r['review_id']].append(r)
        report = json.loads((RESULTS / p['id'] / 'report.json').read_text(encoding='utf-8'))
        names = {(label, g['topic']): f"[{'·'.join(g['keywords'][:3])}] {(g.get('representative') or [''])[0][:20]}"
                 for label in ['NEG', 'POS'] for g in report['groups'][label]['groups'] if g['topic'] != -1}
        picked[p['id']] = {'name': p['name'], 'reviews': by_review, 'groups': names}
    table = pq.read_table(CANONICAL_REVIEWS, columns=['product_id', 'review_id', 'rating', 'review_text_raw'])
    table = table.filter(pc.is_in(table['product_id'], value_set=pa.array(list(picked), type=pa.string())))
    meta = {r['review_id']: r for r in table.drop_columns(['product_id']).to_pylist()}
    lines = ['# 리뷰별 판정 확인', '',
             f'run {RUN}. 상품마다 칭찬·불만이 섞인 리뷰 2개, 별점 1~2점 리뷰 1개, 나머지는 무작위. '
             '판정은 단위(조각)마다 긍정 / **부정** / 평가 아님 / _재확인_(확신도 0.8 미만).', '']
    for pid, v in picked.items():
        by_review = {rid: us for rid, us in v['reviews'].items() if rid in meta}
        mixed = sorted(rid for rid, us in by_review.items() if {'POS', 'NEG'} <= {u['label'] for u in us} and len(us) >= 3)
        low = sorted(rid for rid, us in by_review.items() if meta[rid]['rating'] <= 2 and len(us) >= 2)
        plain = sorted(rid for rid, us in by_review.items() if len(us) >= 2)
        chosen = rng.sample(mixed, min(2, len(mixed))) + rng.sample(low, min(1, len(low)))
        chosen += [rid for rid in rng.sample(plain, min(10, len(plain))) if rid not in chosen]
        chosen = list(dict.fromkeys(chosen))[:PER_PRODUCT]
        lines += [f'## {v["name"]}', '']
        for rid in chosen:
            m = meta[rid]
            lines += [f'**별점 {m["rating"]}** · `{rid}`', '', '> ' + normalize_text(m['review_text_raw']).replace('\n', ' '), '',
                      '| 단위 | 판정 | 확신도 | 묶음 |', '| --- | --- | ---: | --- |']
            for u in v['reviews'][rid]:
                shown = KOREAN[u['label']] + (f" ({KOREAN[u['pred']].strip('*_')}?)" if u['label'] == 'REVIEW' else '')
                group = v['groups'].get((u['label'], u.get('group')), '묶이지 않음' if 'group' in u else '')
                lines.append(f"| {u['unit'].replace('|', '/')} | {shown} | {u['confidence']:.2f} | {group} |")
            lines.append('')
    target = RESULTS / 'review-samples.md'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(target, sum(1 for l in lines if l.startswith('**별점')), 'reviews')


if __name__ == '__main__':
    main()
