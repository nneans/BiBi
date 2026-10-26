"""Select the most frequent name per product ID without changing review records.

Run: python -m bibi_reviews.canonicalize_products
Outputs are separate from the input and existing experiment datasets.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import paths


def canonicalize(reviews):
    required = {'review_id', 'product_id', 'product_name'}
    if not required.issubset(reviews.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(reviews.columns))}')
    if 'product_name_original' in reviews.columns:
        raise ValueError('Input already contains product_name_original')
    if reviews[['product_id', 'product_name']].isna().any().any():
        raise ValueError('Missing product ID or name')
    if not reviews.review_id.is_unique:
        raise ValueError('Review IDs must be unique')

    counts = (reviews.groupby(['product_id', 'product_name'], sort=False)
              .size().rename('name_review_count').reset_index())
    counts = counts.sort_values(
        ['product_id', 'name_review_count', 'product_name'],
        ascending=[True, False, True], kind='stable')
    grouped = counts.groupby('product_id', sort=False)
    counts['name_rank'] = grouped.cumcount() + 1
    counts['is_top_frequency'] = counts.name_review_count.eq(
        grouped.name_review_count.transform('max'))
    counts['top_frequency_tie_count'] = counts.groupby('product_id').is_top_frequency.transform('sum')
    counts['product_review_count'] = grouped.name_review_count.transform('sum')
    counts['original_name_count'] = grouped.product_name.transform('size')

    mapping = counts[counts.name_rank.eq(1)].copy().rename(
        columns={'product_name': 'canonical_product_name',
                 'name_review_count': 'canonical_name_review_count'})
    mapping = mapping[['product_id', 'canonical_product_name', 'original_name_count',
                       'product_review_count', 'canonical_name_review_count',
                       'top_frequency_tie_count']].reset_index(drop=True)
    mapping['canonical_name_share_pct'] = (
        mapping.canonical_name_review_count / mapping.product_review_count * 100)
    lookup = mapping.set_index('product_id').canonical_product_name

    aliases = counts.rename(columns={'product_name': 'product_name_original'}).copy()
    aliases['canonical_product_name'] = aliases.product_id.map(lookup)
    result = reviews.copy()
    result.insert(result.columns.get_loc('product_name') + 1,
                  'product_name_original', reviews.product_name)
    result['product_name'] = result.product_id.map(lookup)
    return result, mapping, aliases


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=paths.REVIEWS_NORMAL)
    parser.add_argument('--output-dir', type=Path, default=paths.CANONICAL_DIR)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output_dir}')
    source_hash = file_hash(args.input)
    reviews = pd.read_parquet(args.input)
    result, mapping, aliases = canonicalize(reviews)
    # Verify every non-name value, record order and original name before writing.
    untouched_columns = [c for c in reviews.columns if c != 'product_name']
    pd.testing.assert_frame_equal(result[untouched_columns], reviews[untouched_columns])
    assert result.product_name_original.equals(reviews.product_name)
    assert result.groupby('product_id').product_name.nunique().eq(1).all()
    assert result.product_name.notna().all()
    assert mapping.product_id.is_unique
    assert aliases.name_review_count.sum() == len(reviews)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_dir / 'reviews_normal_canonical.parquet', index=False)
    mapping.to_csv(args.output_dir / 'product_name_mapping.csv', index=False, encoding='utf-8-sig')
    aliases.to_csv(args.output_dir / 'product_name_aliases.csv', index=False, encoding='utf-8-sig')

    # Validate actual exported files, not only the in-memory transformation.
    loaded = pd.read_parquet(args.output_dir / 'reviews_normal_canonical.parquet')
    pd.testing.assert_frame_equal(loaded, result)
    loaded_mapping = pd.read_csv(args.output_dir / 'product_name_mapping.csv',
                                 dtype={'product_id': str}, keep_default_na=False)
    assert loaded_mapping.product_id.nunique() == mapping.product_id.nunique()
    assert loaded_mapping.canonical_product_name.tolist() == mapping.canonical_product_name.tolist()
    loaded_aliases = pd.read_csv(args.output_dir / 'product_name_aliases.csv',
                                 dtype={'product_id': str}, keep_default_na=False)
    assert len(loaded_aliases) == len(aliases)
    assert loaded_aliases.name_review_count.sum() == len(reviews)
    assert file_hash(args.input) == source_hash

    shared = mapping[mapping.canonical_product_name.duplicated(keep=False)]
    summary = {
        'input': str(args.input), 'input_sha256': source_hash,
        'scope': 'All normal reviews; blind reviews already excluded from input',
        'selection_rule': 'Highest review count per ID; ties resolved by ascending exact name string',
        'mapping_direction': 'Each product_id has exactly one product_name; names may be shared across IDs',
        'review_rows': len(result), 'product_ids': len(mapping),
        'original_id_name_pairs': len(aliases),
        'formerly_multiple_name_ids': int(mapping.original_name_count.gt(1).sum()),
        'changed_review_name_rows': int(result.product_name.ne(result.product_name_original).sum()),
        'top_frequency_tie_ids': mapping.loc[mapping.top_frequency_tie_count.gt(1), 'product_id'].tolist(),
        'canonical_distinct_names': int(mapping.canonical_product_name.nunique()),
        'shared_canonical_names': shared[['product_id', 'canonical_product_name']].to_dict('records'),
        'all_non_name_columns_and_row_order_unchanged': True,
        'original_names_preserved': True,
        'all_ids_have_one_name': True,
        'export_reload_verified': True,
        'input_hash_unchanged': True,
    }
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
