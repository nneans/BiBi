"""Preprocessing, final form: every review CSV the store has delivered -> every file the later steps read.

New reviews come as new CSVs (a new period, or a fresh full export that overlaps the old ones). Each is added once
with `add`, kept unchanged in raw/ and listed in raw/deliveries.json; every run rebuilds from all of them:

  0. merge          deliveries in the order they were added, one row per 리뷰글번호 (the key, not the date: one
                    day holds many reviews, and an old review can be edited or blinded later). A newer version wins
                    (최종수정일, else 리뷰등록일), an identical row is skipped, an older one is ignored, the same time
                    with other content -> the later delivery (the DB import, backbone.ReviewStore, uses the same
                    rules). A review missing from a new CSV is not deleted (a new-period CSV holds no old reviews);
                    only a review shown as 블라인드 leaves the analysis.
  1. read + check   each CSV with the rules of the DB import (ReviewStore.prepare): required columns, unique review
                    IDs, ratings 1..5, dates in Asia/Seoul, normalize_text, text_hash
  2. normal reviews blind reviews dropped; review_group_id = 'review:' + (관련리뷰글번호 or 리뷰글번호), so an
                    original review and its 한달사용 follow-up share it; rows ordered by product, time, review
                    -> output/reviews_normal.parquet
  3. product names  one display name per product_id (the most frequent; canonicalize_products.canonicalize); the
                    product_id stays the key and products are never merged
                    -> output/canonical_products/
  4. analysis text  text_clean.clean_review (v2: line breaks kept, repeats shortened, invisible characters and
                    filler removed; emoji and spelling kept) and purchase_id = one customer's purchase: the
                    review_group_id, or the first copy's purchase for a long text (COPY_MIN_CHARS+) pasted again
                    into the same product
                    -> output/analysis_text.parquet
  5. changes        reviews new / updated / removed (blinded) by the latest delivery, so later steps can redo only
                    the products that changed -> output/changes.parquet (kept as it is by a rebuild with no new CSV)
  6. manifest       deliveries and their hashes, merge counts, counts of every step, rules, output hashes
                    -> output/preprocess-manifest.json
Raw CSVs are never changed. Rebuilding with no new delivery must give the same files; if it does not (a rule in
the code changed), the run stops unless --replace is given.
Usage: python 1_preprocessing/preprocess.py                 rebuild from raw/deliveries.json and compare
       python 1_preprocessing/preprocess.py add <new.csv>   check, copy into raw/, list it, rebuild
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from bibi_reviews.backbone import ReviewStore
from bibi_reviews.canonicalize_products import canonicalize
from bibi_reviews.paths import PREPROCESS, PROCESSED, RAW
from bibi_reviews.text_clean import CLEAN_VERSION, clean_review

DELIVERIES = RAW / 'deliveries.json'
COPY_MIN_CHARS = 40  # a pasted text this long is one customer's, not a coincidence ('맛있어요' is not)
NORMAL_COLUMNS = ['review_id', 'product_id', 'product_name', 'review_type', 'rating', 'created_at', 'source_updated_at',
                  'review_text_raw', 'review_text', 'related_review_id', 'review_group_id', 'text_hash', 'display_status']
OUTPUTS = ['reviews_normal.parquet', 'canonical_products/reviews_normal_canonical.parquet',
           'canonical_products/product_name_mapping.csv', 'canonical_products/product_name_aliases.csv',
           'analysis_text.parquet']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_raw(csv_path):
    raw = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    return raw, pd.DataFrame(ReviewStore.prepare(raw))


def merge(csv_paths):
    """One record per review_id over every delivery, in order (see 0. above); counts per delivery."""
    current, per_file, raw_rows = {}, [], 0
    for path in csv_paths:
        raw, records = read_raw(path)
        raw_rows += len(raw)
        c = Counter()
        for r in records.to_dict('records'):
            old = current.get(r['review_id'])
            if old is None:
                c['new'] += 1
            elif old['row_hash'] == r['row_hash']:
                c['unchanged'] += 1
                continue
            elif r['effective_at'] < old['effective_at']:
                c['older_ignored'] += 1
                continue
            else:
                c['updated'] += 1
                c['same_time_other_content'] += r['effective_at'] == old['effective_at']
                c['now_blind'] += r['display_status'] == '블라인드' and old['display_status'] != '블라인드'
            current[r['review_id']] = r
        per_file.append({'rows': len(raw), **{k: int(v) for k, v in c.items()}})
    return pd.DataFrame(list(current.values())), per_file, raw_rows


def normal_reviews(records):
    d = records[records.display_status == '정상'].copy()
    d['product_name'] = d.product_name_original
    d['review_group_id'] = 'review:' + d.related_review_id.where(d.related_review_id != '', d.review_id)
    for col in ['created_at', 'source_updated_at']:
        d[col] = pd.to_datetime(d[col], utc=True).dt.tz_convert('Asia/Seoul').astype('datetime64[ns, Asia/Seoul]')
    d['rating'] = d.rating.astype('int8')
    d = d.sort_values(['product_id', 'created_at', 'review_id'], kind='stable').reset_index(drop=True)
    return d[NORMAL_COLUMNS]


def analysis_text(canonical):
    d = canonical[['review_id', 'product_id', 'created_at', 'review_text_raw', 'review_group_id']].copy()
    d['analysis_text'] = d.review_text_raw.map(clean_review)
    purchase, first_copy = {}, {}
    for r in d.sort_values(['product_id', 'created_at', 'review_id'], kind='stable').itertuples():
        own = r.review_group_id
        if len(r.analysis_text) >= COPY_MIN_CHARS:
            own = first_copy.setdefault((r.product_id, r.analysis_text), own)
        purchase[r.review_id] = own
    d['purchase_id'] = d.review_id.map(purchase)
    d['clean_version'] = CLEAN_VERSION
    return d[['review_id', 'product_id', 'analysis_text', 'purchase_id', 'clean_version']]


def changes(previous, normal):
    """review_id, product_id, change (new / updated / removed) against the previous reviews_normal.parquet."""
    cols = ['review_id', 'product_id', 'text_hash', 'rating']
    old = pd.read_parquet(previous, columns=cols) if previous.exists() else pd.DataFrame(columns=cols)
    m = old.merge(normal[cols], on='review_id', how='outer', suffixes=('_old', ''), indicator=True)
    m['product_id'] = m.product_id.fillna(m.product_id_old)
    m['change'] = 'unchanged'
    m.loc[m._merge == 'right_only', 'change'] = 'new'
    m.loc[m._merge == 'left_only', 'change'] = 'removed'
    both = m._merge == 'both'
    m.loc[both & ((m.text_hash != m.text_hash_old) | (m.rating != m.rating_old)), 'change'] = 'updated'
    return m.loc[m.change != 'unchanged', ['review_id', 'product_id', 'change']].sort_values(['product_id', 'review_id']).reset_index(drop=True)


def build(csv_paths, out):
    records, per_file, raw_rows = merge(csv_paths)
    normal = normal_reviews(records)
    canonical, mapping, aliases = canonicalize(normal)
    text = analysis_text(canonical)
    (out / 'canonical_products').mkdir(parents=True, exist_ok=True)
    for name, frame in {'reviews_normal.parquet': normal, 'canonical_products/reviews_normal_canonical.parquet': canonical,
                        'analysis_text.parquet': text}.items():
        frame.to_parquet(out / name, index=False)
    mapping.to_csv(out / 'canonical_products/product_name_mapping.csv', index=False, encoding='utf-8-sig')
    aliases.to_csv(out / 'canonical_products/product_name_aliases.csv', index=False, encoding='utf-8-sig')
    month = normal.review_type == '한달사용'
    counts = {
        'raw_rows': raw_rows, 'distinct_reviews': len(records),
        'blind_reviews': int((records.display_status == '블라인드').sum()),
        'normal_reviews': len(normal), 'products': int(normal.product_id.nunique()),
        'month_followups': int(month.sum()),
        'month_followups_whose_original_is_in_data': int(normal[month].related_review_id.isin(set(normal.review_id)).sum()),
        'review_groups': int(normal.review_group_id.nunique()), 'purchases': int(text.purchase_id.nunique()),
        'reviews_folded_as_copies': int((text.purchase_id != canonical.review_group_id).sum()),
        'analysis_text_differs_from_review_text': int((text.analysis_text != canonical.review_text).sum()),
        'empty_analysis_text': int((text.analysis_text == '').sum()),
        'product_ids_with_several_names': int(mapping.original_name_count.gt(1).sum()),
    }
    return counts, per_file, normal, mapping


def same(a, b):
    """Two output files hold the same table (parquet: frame equality; csv: bytes)."""
    if a.suffix == '.parquet':
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
            return True
        except AssertionError:
            return False
    return a.read_bytes() == b.read_bytes()


def deliveries():
    return json.loads(DELIVERIES.read_text(encoding='utf-8')) if DELIVERIES.exists() else []


def add(csv_path):
    """Check a new CSV, copy it into raw/ unchanged and list it. The same file twice is refused."""
    listed, digest = deliveries(), sha256(csv_path)
    if any(d['sha256'] == digest for d in listed):
        sys.exit(f'already added: {csv_path.name}')
    if (RAW / csv_path.name).exists():
        sys.exit(f'raw/ already has a file named {csv_path.name}; rename the new CSV')
    raw, records = read_raw(csv_path)  # stops here if the format is wrong
    shutil.copy2(csv_path, RAW / csv_path.name)
    assert sha256(RAW / csv_path.name) == digest
    created = pd.to_datetime(records.created_at, utc=True).dt.tz_convert('Asia/Seoul')
    listed.append({'file': csv_path.name, 'sha256': digest, 'rows': len(raw), 'added': date.today().isoformat(),
                   'reviews_from': created.min().isoformat(), 'reviews_to': created.max().isoformat()})
    DELIVERIES.write_text(json.dumps(listed, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'added {csv_path.name}: {len(raw):,} rows, {listed[-1]["reviews_from"][:10]} ~ {listed[-1]["reviews_to"][:10]}')


def rebuild(replace=False):
    listed = deliveries()
    if not listed:
        sys.exit('raw/deliveries.json lists no CSV; add one: preprocess.py add <csv>')
    for d in listed:
        if sha256(RAW / d['file']) != d['sha256']:
            sys.exit(f"raw/{d['file']} differs from the one that was added; raw CSVs are never changed")
    manifest_path = PROCESSED / 'preprocess-manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    new_delivery = [d['sha256'] for d in previous.get('deliveries', [])] != [d['sha256'] for d in listed]
    with tempfile.TemporaryDirectory(dir=PREPROCESS) as tmp:
        tmp = Path(tmp)
        counts, per_file, normal, mapping = build([RAW / d['file'] for d in listed], tmp)
        status = {name: 'new' if not (PROCESSED / name).exists() else 'unchanged' if same(tmp / name, PROCESSED / name)
                  else 'differs' for name in OUTPUTS}
        differs = [n for n, s in status.items() if s == 'differs']
        if differs and not new_delivery and not replace:
            sys.exit(f'no new CSV, yet the outputs differ (a rule in the code changed?); --replace to accept: {differs}')
        diff = changes(PROCESSED / 'reviews_normal.parquet', normal)
        for name, s in status.items():
            if s != 'unchanged':
                (PROCESSED / name).parent.mkdir(parents=True, exist_ok=True)
                (tmp / name).replace(PROCESSED / name)
    if new_delivery or replace or 'changes_from_latest_delivery' not in previous:
        diff.to_parquet(PROCESSED / 'changes.parquet', index=False)
        latest = {**{k: int(v) for k, v in diff.change.value_counts().items()}, 'products': int(diff.product_id.nunique())}
    else:
        latest = previous['changes_from_latest_delivery']
    manifest = {
        'deliveries': [{**d, 'merge': m} for d, m in zip(listed, per_file)],
        'counts': counts,
        'changes_from_latest_delivery': latest,
        'rules': {'merge': 'one row per review_id; newer 최종수정일/리뷰등록일 wins; same time, other content -> later '
                           'delivery; a review missing from a CSV is not deleted, a blinded one leaves',
                  'checks': 'backbone.ReviewStore.prepare (same as the DB import)',
                  'review_text': 'backbone.normalize_text (DB text and text_hash)',
                  'review_group_id': "'review:' + (related_review_id or review_id)",
                  'order': 'product_id, created_at, review_id',
                  'product_name': 'most frequent name per product_id, ties by name (canonicalize_products)',
                  'analysis_text': f'text_clean.clean_review v{CLEAN_VERSION}',
                  'purchase_id': f'review_group_id; a text of {COPY_MIN_CHARS}+ characters pasted again into the same '
                                 'product takes the first copy\'s purchase'},
        'outputs': {name: {'status': s, 'sha256': sha256(PROCESSED / name)} for name, s in status.items()},
        'tie_product_ids': mapping.loc[mapping.top_frequency_tie_count.gt(1), 'product_id'].tolist(),
        'product_ids_sharing_a_name': mapping[mapping.canonical_product_name.duplicated(keep=False)][
            ['product_id', 'canonical_product_name']].to_dict('records'),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'counts': counts, 'changes_from_latest_delivery': latest, 'outputs': status},
                     ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=['add'], help='add <csv>: register a new delivery, then rebuild')
    parser.add_argument('csv', nargs='?', type=Path)
    parser.add_argument('--replace', action='store_true', help='accept outputs that differ with no new CSV (rule change)')
    args = parser.parse_args()
    if args.command == 'add':
        if not args.csv:
            parser.error('add needs the path of the new CSV')
        add(args.csv)
    rebuild(args.replace)


if __name__ == '__main__':
    main()
