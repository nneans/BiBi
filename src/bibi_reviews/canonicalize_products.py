"""One display name per product ID (the most frequent one; ties by name), without changing review records.

Used by 1_preprocessing/preprocess.py, which writes and checks the outputs. The product_id stays the key: names may be
shared by several IDs and products are never merged.
"""


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

