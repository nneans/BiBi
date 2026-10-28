"""Does the review DB hold exactly what the delivered CSVs say? (read-only)

Every CSV of raw/deliveries.json is imported, in order, into a temporary SQLite DB with the same code as the real
import (backbone.ReviewStore); then every review of the real DB is compared with it, column by column (the storage
bookkeeping stored_at / last_upload_id aside). A difference means the DB was changed outside ingest.py, or a
delivery is missing from the list. Also lists upload jobs that never finished.
Output: verification/db-vs-raw.json
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from bibi_reviews.backbone import ReviewStore
from bibi_reviews.paths import DELIVERIES, INGEST, RAW

OUT = INGEST / 'verification' / 'db-vs-raw.json'
COMPARED = ['product_id', 'product_name_original', 'review_type', 'rating', 'created_at', 'source_updated_at',
            'effective_at', 'related_review_id', 'display_status', 'review_text_raw', 'review_text', 'text_hash', 'row_hash']


def reviews(store):
    with store.engine.connect() as c:
        frame = pd.read_sql_query(select(store.review), c)
    return frame[['review_id'] + COMPARED].fillna('').astype(str)


def main():
    listed = json.loads(DELIVERIES.read_text(encoding='utf-8'))
    real = ReviewStore()
    with tempfile.TemporaryDirectory(dir=INGEST) as tmp:
        expected_store = ReviewStore(f"sqlite:///{(Path(tmp) / 'expected.sqlite3').as_posix()}")
        for d in listed:
            expected_store.import_csv(RAW / d['file'])
        expected = reviews(expected_store)
        expected_store.engine.dispose()
    actual = reviews(real)
    with real.engine.connect() as c:
        jobs = pd.read_sql_query(select(real.upload), c)
    m = expected.merge(actual, on='review_id', how='outer', suffixes=('_raw', '_db'), indicator=True)
    both = m[m._merge == 'both']
    report = {'checked_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'deliveries': [d['file'] for d in listed], 'raw_reviews': len(expected), 'db_reviews': len(actual),
              'only_in_raw': int((m._merge == 'left_only').sum()), 'only_in_db': int((m._merge == 'right_only').sum()),
              'differing': {k: int((both[f'{k}_raw'] != both[f'{k}_db']).sum()) for k in COMPARED},
              'unfinished_upload_jobs': jobs.loc[~jobs.status.isin(['completed', 'failed']),
                                                 ['job_id', 'filename', 'status', 'created_at']].to_dict('records')}
    report['same'] = report['only_in_raw'] == report['only_in_db'] == 0 and not any(report['differing'].values())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if report['same'] else 1)


if __name__ == '__main__':
    main()
