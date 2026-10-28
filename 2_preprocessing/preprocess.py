"""Step 2: the review DB (the ledger, 1_ingest) -> every file the later steps read.

  python 2_preprocessing/preprocess.py              read the configured DB (PostgreSQL; BIBI_DATABASE_URL overrides)
  python 2_preprocessing/preprocess.py --from-raw   no DB: import every CSV of 1_ingest/raw/deliveries.json, in order,
                                                    into a temporary SQLite DB with the same import code, then read it

Which version of a review counts, duplicates, edits and blinding are decided once, by the DB import
(backbone.ReviewStore; see 1_ingest/ingest.py). This step only reads the result:
  1. normal reviews blind reviews dropped; review_group_id = 'review:' + (관련리뷰글번호 or 리뷰글번호), so an
                    original review and its 한달사용 follow-up share it; rows ordered by product, time, review
                    -> output/reviews_normal.parquet
  2. product names  one display name per product_id (the most frequent; canonicalize_products.canonicalize; the DB's
                    products table follows the same rule and is checked against it); the product_id stays the key
                    and products are never merged -> output/canonical_products/
  3. analysis text  text_clean.clean_review (v2: line breaks kept, repeats shortened, invisible characters and
                    filler removed; emoji and spelling kept) and purchase_id = one customer's purchase: the
                    review_group_id, or the first copy's purchase for a long text (COPY_MIN_CHARS+) pasted again
                    into the same product -> output/analysis_text.parquet
  4. changes        reviews new / updated / removed (blinded) since the last time the DB changed, so later steps
                    can redo only those products -> output/changes.parquet
  5. manifest       source and its fingerprint, deliveries, counts of every step, rules, output hashes
                    -> output/preprocess-manifest.json
If the DB has not changed since the last build, the rebuilt files must be the same; if they are not (a rule in the
code changed), the run stops unless --replace is given.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from bibi_reviews.backbone import ReviewStore
from bibi_reviews.canonicalize_products import canonicalize
from bibi_reviews.paths import DELIVERIES, PREPROCESS, PROCESSED, RAW
from bibi_reviews.text_clean import CLEAN_VERSION, clean_review

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


def deliveries():
    return json.loads(DELIVERIES.read_text(encoding='utf-8')) if DELIVERIES.exists() else []


def read_store(store):
    """Every review of the DB (blind ones too) and its products table."""
    with store.engine.connect() as c:
        reviews = pd.read_sql_query(select(store.review), c)
        products = pd.read_sql_query(select(store.product), c)
    reviews['related_review_id'] = reviews.related_review_id.fillna('')
    return reviews, products


def fingerprint(reviews):
    """Changes whenever any review of the DB changes (its row_hash covers every stored field)."""
    digest = hashlib.sha256()
    for line in sorted(reviews.review_id + ':' + reviews.row_hash):
        digest.update(line.encode() + b'\n')
    return digest.hexdigest()


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


def build(reviews, products, out):
    normal = normal_reviews(reviews)
    canonical, mapping, aliases = canonicalize(normal)
    text = analysis_text(canonical)
    (out / 'canonical_products').mkdir(parents=True, exist_ok=True)
    for name, frame in {'reviews_normal.parquet': normal, 'canonical_products/reviews_normal_canonical.parquet': canonical,
                        'analysis_text.parquet': text}.items():
        frame.to_parquet(out / name, index=False)
    mapping.to_csv(out / 'canonical_products/product_name_mapping.csv', index=False, encoding='utf-8-sig')
    aliases.to_csv(out / 'canonical_products/product_name_aliases.csv', index=False, encoding='utf-8-sig')
    db_name = products.set_index('product_id').product_name
    name_mismatch = mapping[mapping.canonical_product_name != mapping.product_id.map(db_name)].product_id.tolist()
    month = normal.review_type == '한달사용'
    counts = {
        'db_reviews': len(reviews), 'blind_reviews': int((reviews.display_status == '블라인드').sum()),
        'normal_reviews': len(normal), 'products': int(normal.product_id.nunique()),
        'month_followups': int(month.sum()),
        'month_followups_whose_original_is_in_data': int(normal[month].related_review_id.isin(set(normal.review_id)).sum()),
        'review_groups': int(normal.review_group_id.nunique()), 'purchases': int(text.purchase_id.nunique()),
        'reviews_folded_as_copies': int((text.purchase_id != canonical.review_group_id).sum()),
        'analysis_text_differs_from_review_text': int((text.analysis_text != canonical.review_text).sum()),
        'empty_analysis_text': int((text.analysis_text == '').sum()),
        'product_ids_with_several_names': int(mapping.original_name_count.gt(1).sum()),
        'product_names_differing_from_db': len(name_mismatch),
    }
    return counts, normal, mapping


def same(a, b):
    """Two output files hold the same table (parquet: frame equality; csv: bytes)."""
    if a.suffix == '.parquet':
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
            return True
        except AssertionError:
            return False
    return a.read_bytes() == b.read_bytes()


def from_raw(folder):
    """A temporary SQLite review DB holding every delivery, imported in order with the same code as the real DB."""
    store = ReviewStore(f"sqlite:///{(Path(folder) / 'reviews.sqlite3').as_posix()}")
    for d in deliveries():
        if sha256(RAW / d['file']) != d['sha256']:
            sys.exit(f"raw/{d['file']} differs from the one that was added; raw CSVs are never changed")
        store.import_csv(RAW / d['file'])
    return store


def rebuild(replace=False, store=None, use_raw=False):
    manifest_path = PROCESSED / 'preprocess-manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    with tempfile.TemporaryDirectory(dir=PREPROCESS) as tmp:
        tmp = Path(tmp)
        if use_raw:
            store, source = from_raw(tmp), 'raw CSVs of 1_ingest/raw/deliveries.json, imported into a temporary SQLite DB'
        else:
            store, source = store or ReviewStore(), 'review DB'
        reviews, products = read_store(store)
        store.engine.dispose()
        key = fingerprint(reviews)
        db_changed = key != previous.get('source_fingerprint')
        counts, normal, mapping = build(reviews, products, tmp)
        status = {name: 'new' if not (PROCESSED / name).exists() else 'unchanged' if same(tmp / name, PROCESSED / name)
                  else 'differs' for name in OUTPUTS}
        differs = [n for n, s in status.items() if s == 'differs']
        if differs and not db_changed and not replace:
            sys.exit(f'the DB has not changed, yet the outputs differ (a rule in the code changed?); --replace to accept: {differs}')
        diff = changes(PROCESSED / 'reviews_normal.parquet', normal)
        for name, s in status.items():
            if s != 'unchanged':
                (PROCESSED / name).parent.mkdir(parents=True, exist_ok=True)
                (tmp / name).replace(PROCESSED / name)
    if db_changed or replace or 'changes_from_latest_db_change' not in previous:
        diff.to_parquet(PROCESSED / 'changes.parquet', index=False)
        latest = {**{k: int(v) for k, v in diff.change.value_counts().items()}, 'products': int(diff.product_id.nunique())}
    else:
        latest = previous['changes_from_latest_db_change']
    manifest = {
        'source': source, 'source_fingerprint': key, 'deliveries': deliveries(), 'counts': counts,
        'changes_from_latest_db_change': latest,
        'rules': {'versions': 'decided by the DB import (backbone.ReviewStore): one row per review_id, newer '
                              '최종수정일/리뷰등록일 wins, a review missing from a CSV is not deleted, a blinded one leaves',
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
    print(json.dumps({'source': source, 'counts': counts, 'changes_from_latest_db_change': latest, 'outputs': status},
                     ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from-raw', action='store_true', help='no DB: rebuild from the raw CSVs via a temporary SQLite DB')
    parser.add_argument('--replace', action='store_true', help='accept outputs that differ with an unchanged DB (rule change)')
    args = parser.parse_args()
    rebuild(args.replace, use_raw=args.from_raw)


if __name__ == '__main__':
    main()
