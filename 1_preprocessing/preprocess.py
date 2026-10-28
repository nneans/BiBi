"""Preprocessing, final form: the raw Naver smartstore CSV -> every file the later steps read, in one run.

  1. read + check   the raw CSV with the rules the DB import uses (backbone.ReviewStore.prepare): required columns,
                    unique review IDs, ratings 1..5, dates in Asia/Seoul, normalize_text, text_hash
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
  5. manifest       input hash, counts of every step, versions, output hashes -> output/preprocess-manifest.json
The raw CSV is never changed. Run again on the same CSV and every output is rebuilt and compared with the files
already there: equal files are left alone, a difference stops the run unless --replace is given (a new CSV).
Usage: python 1_preprocessing/preprocess.py [--csv <raw.csv>] [--replace]
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from bibi_reviews.backbone import ReviewStore
from bibi_reviews.canonicalize_products import canonicalize
from bibi_reviews.paths import PREPROCESS, PROCESSED, RAW_CSV, ROOT
from bibi_reviews.text_clean import CLEAN_VERSION, clean_review

COPY_MIN_CHARS = 40  # a pasted text this long is one customer's, not a coincidence ('맛있어요' is not)
NORMAL_COLUMNS = ['review_id', 'product_id', 'product_name', 'review_type', 'rating', 'created_at', 'source_updated_at',
                  'review_text_raw', 'review_text', 'related_review_id', 'review_group_id', 'text_hash', 'display_status']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_raw(csv_path):
    raw = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    return raw, pd.DataFrame(ReviewStore.prepare(raw))


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


def build(csv_path, out):
    raw, records = read_raw(csv_path)
    normal = normal_reviews(records)
    canonical, mapping, aliases = canonicalize(normal)
    text = analysis_text(canonical)
    (out / 'canonical_products').mkdir(parents=True, exist_ok=True)
    files = {'reviews_normal.parquet': normal, 'canonical_products/reviews_normal_canonical.parquet': canonical,
             'analysis_text.parquet': text}
    for name, frame in files.items():
        frame.to_parquet(out / name, index=False)
    mapping.to_csv(out / 'canonical_products/product_name_mapping.csv', index=False, encoding='utf-8-sig')
    aliases.to_csv(out / 'canonical_products/product_name_aliases.csv', index=False, encoding='utf-8-sig')
    month = normal.review_type == '한달사용'
    counts = {
        'raw_rows': len(raw), 'blind_reviews': int((records.display_status == '블라인드').sum()),
        'normal_reviews': len(normal), 'products': int(normal.product_id.nunique()),
        'month_followups': int(month.sum()),
        'month_followups_whose_original_is_in_data': int(normal[month].related_review_id.isin(set(normal.review_id)).sum()),
        'review_groups': int(normal.review_group_id.nunique()), 'purchases': int(text.purchase_id.nunique()),
        'reviews_folded_as_copies': int((text.purchase_id != canonical.review_group_id).sum()),
        'analysis_text_differs_from_review_text': int((text.analysis_text != canonical.review_text).sum()),
        'empty_analysis_text': int((text.analysis_text == '').sum()),
        'product_ids_with_several_names': int(mapping.original_name_count.gt(1).sum()),
    }
    return counts, mapping


def same(a, b):
    """Two output files hold the same table (parquet: frame equality; csv: bytes)."""
    if a.suffix == '.parquet':
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
            return True
        except AssertionError:
            return False
    return a.read_bytes() == b.read_bytes()


OUTPUTS = ['reviews_normal.parquet', 'canonical_products/reviews_normal_canonical.parquet',
           'canonical_products/product_name_mapping.csv', 'canonical_products/product_name_aliases.csv',
           'analysis_text.parquet']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=RAW_CSV)
    parser.add_argument('--replace', action='store_true', help='overwrite outputs that differ (a new CSV)')
    args = parser.parse_args()
    input_hash = sha256(args.csv)
    with tempfile.TemporaryDirectory(dir=PREPROCESS) as tmp:
        tmp = Path(tmp)
        counts, mapping = build(args.csv, tmp)
        assert sha256(args.csv) == input_hash, 'the raw CSV changed while reading it'
        status = {}
        for name in OUTPUTS:
            new, old = tmp / name, PROCESSED / name
            status[name] = 'new' if not old.exists() else 'unchanged' if same(new, old) else 'differs'
        differs = [n for n, s in status.items() if s == 'differs']
        if differs and not args.replace:
            sys.exit(f'outputs differ from the files already there (use --replace for a new CSV): {differs}')
        for name, s in status.items():
            if s != 'unchanged':
                (PROCESSED / name).parent.mkdir(parents=True, exist_ok=True)
                (tmp / name).replace(PROCESSED / name)
    manifest = {
        'input': args.csv.relative_to(ROOT).as_posix() if args.csv.is_relative_to(ROOT) else args.csv.as_posix(),
        'input_sha256': input_hash, 'counts': counts,
        'rules': {'checks': 'backbone.ReviewStore.prepare (same as the DB import)',
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
    (PROCESSED / 'preprocess-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'counts': counts, 'outputs': status}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
