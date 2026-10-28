"""Step 1: a review CSV the store delivered -> the review DB (the ledger) and raw/.

  python 1_ingest/ingest.py add <new.csv>    check, import into the DB, keep a copy, list it, rebuild 2_preprocessing

A new period, or a fresh full export that overlaps the old ones: both are added the same way. The DB import
(backbone.ReviewStore.import_csv, also behind the web upload of the dashboard) keeps one row per 리뷰글번호 (the key,
not the date: one day holds many reviews, and an old review can be edited or blinded later): a newer version wins
(최종수정일, else 리뷰등록일), an identical row is skipped, an older one is ignored, a changed review keeps its
previous version in review_history, a blinded one loses its text. A review missing from a new CSV is not deleted
(a new-period CSV holds no old reviews).
The same file twice is refused (its hash); the copy in raw/ is never changed; raw/deliveries.json lists every
delivery with its hash, rows, review period and what the DB import did with it.
The DB is the one configured in configs/review_pipeline/pipeline.json (BIBI_DATABASE_URL overrides).
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from bibi_reviews.backbone import ReviewStore
from bibi_reviews.paths import DELIVERIES, PREPROCESS, RAW

IMPORT_COUNTS = ['rows', 'new', 'updated', 'unchanged', 'stale', 'same_timestamp_conflicts', 'blind_rows']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def deliveries():
    return json.loads(DELIVERIES.read_text(encoding='utf-8')) if DELIVERIES.exists() else []


def add(csv_path, store=None):
    listed, digest = deliveries(), sha256(csv_path)
    if any(d['sha256'] == digest for d in listed):
        sys.exit(f'already added: {csv_path.name}')
    if (RAW / csv_path.name).exists():
        sys.exit(f'raw/ already has a file named {csv_path.name}; rename the new CSV')
    raw = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    records = pd.DataFrame(ReviewStore.prepare(raw))  # stops here if the format is wrong; nothing is changed yet
    counts = (store or ReviewStore()).import_csv(csv_path)
    shutil.copy2(csv_path, RAW / csv_path.name)
    assert sha256(RAW / csv_path.name) == digest
    created = pd.to_datetime(records.created_at, utc=True).dt.tz_convert('Asia/Seoul')
    listed.append({'file': csv_path.name, 'sha256': digest, 'rows': len(raw), 'added': date.today().isoformat(),
                   'reviews_from': created.min().isoformat(), 'reviews_to': created.max().isoformat(),
                   'db_import': {k: counts[k] for k in IMPORT_COUNTS}})
    DELIVERIES.write_text(json.dumps(listed, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f"added {csv_path.name}: {len(raw):,} rows ({listed[-1]['reviews_from'][:10]} ~ {listed[-1]['reviews_to'][:10]}), "
          f"DB: {json.dumps(listed[-1]['db_import'], ensure_ascii=False)}")


def preprocessing():
    spec = importlib.util.spec_from_file_location('preprocess', PREPROCESS / 'preprocess.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['add'])
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    add(args.csv)
    preprocessing().rebuild()


if __name__ == '__main__':
    main()
