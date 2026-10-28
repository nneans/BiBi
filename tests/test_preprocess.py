import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preprocess', ROOT / '1_preprocessing' / 'preprocess.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

PASTED = '배송도 아주 빠르게 잘왔구 가격도 상품도 다좋네요 진짜 ~~~~~잘먹었습니다 또 살게요'
COLUMNS = ['리뷰글번호', '상품번호', '상품명', '리뷰구분', '관련리뷰글번호', '전시상태', '리뷰등록일', '최종수정일', '리뷰상세내용']
FIRST = [
    ('1', 'p1', '새우 A', '일반', '', '정상', '2025.01.01. 10:00:00', '', '맛있어요&hellip;'),
    ('2', 'p1', '새우 A', '한달사용', '1', '정상', '2025.02.01. 10:00:00', '', '한 달 먹어보니 좋아요~~~~~'),
    ('3', 'p1', '새우 B', '일반', '', '정상', '2025.03.01. 10:00:00', '', PASTED),
    ('4', 'p1', '새우 A', '일반', '', '정상', '2025.04.01. 10:00:00', '2025.04.05. 09:00:00', PASTED.replace('~~~~~', '~~~~~~~')),
    ('5', 'p1', '새우 A', '일반', '', '블라인드', '2025.04.02. 10:00:00', '', '가려진 리뷰'),
    ('6', 'p2', '고등어', '일반', '', '정상', '2025.01.05. 10:00:00', '', PASTED),
]
SECOND = [  # a later export that overlaps the first one
    FIRST[0],                                                                                   # same row: skipped
    ('2', 'p1', '새우 A', '한달사용', '1', '정상', '2025.02.01. 10:00:00', '2025.05.01. 10:00:00', '수정했어요 별로예요'),  # edited
    ('3', 'p1', '새우 B', '일반', '', '블라인드', '2025.03.01. 10:00:00', '2025.05.02. 10:00:00', ''),  # blinded later
    ('4', 'p1', '새우 A', '일반', '', '정상', '2025.04.01. 10:00:00', '', '예전 버전'),              # older version
    ('7', 'p2', '고등어', '일반', '', '정상', '2025.05.03. 10:00:00', '', '새 리뷰예요'),             # new
]


def write_csv(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame['구매자평점'] = '5'
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return path


@pytest.fixture
def folders(tmp_path, monkeypatch):
    raw, out = tmp_path / 'raw', tmp_path / 'output'
    raw.mkdir()
    monkeypatch.setattr(P, 'RAW', raw)
    monkeypatch.setattr(P, 'DELIVERIES', raw / 'deliveries.json')
    monkeypatch.setattr(P, 'PROCESSED', out)
    monkeypatch.setattr(P, 'PREPROCESS', tmp_path)
    return tmp_path, out


def test_preprocess_rules(folders):
    tmp, out = folders
    P.add(write_csv(tmp / 'first.csv', FIRST))
    P.rebuild()
    normal = pd.read_parquet(out / 'reviews_normal.parquet')
    text = pd.read_parquet(out / 'analysis_text.parquet').set_index('review_id')
    assert normal.columns.tolist() == P.NORMAL_COLUMNS
    assert normal.review_id.tolist() == ['1', '2', '3', '4', '6']  # blind dropped; product, time, review
    group = normal.set_index('review_id').review_group_id
    assert group['2'] == group['1'] == 'review:1'  # 한달사용 follow-up joins its original
    assert text.loc['1', 'analysis_text'] == '맛있어요…'
    assert text.loc['2', 'analysis_text'] == '한 달 먹어보니 좋아요~~'
    # the same long text pasted again into the same product is the first copy's purchase; another product is not
    assert text.loc['4', 'purchase_id'] == text.loc['3', 'purchase_id'] == 'review:3'
    assert text.loc['6', 'purchase_id'] == 'review:6'
    names = pd.read_csv(out / 'canonical_products/product_name_mapping.csv', dtype=str)
    assert names.set_index('product_id').canonical_product_name['p1'] == '새우 A'  # most frequent name


def test_new_delivery_merges_by_review_id(folders):
    tmp, out = folders
    P.add(write_csv(tmp / 'first.csv', FIRST))
    P.rebuild()
    P.add(write_csv(tmp / 'second.csv', SECOND))
    P.rebuild()
    normal = pd.read_parquet(out / 'reviews_normal.parquet').set_index('review_id')
    assert sorted(normal.index) == ['1', '2', '4', '6', '7']           # 3 blinded later leaves, 7 is new
    assert normal.loc['2', 'review_text'] == '수정했어요 별로예요'          # newer edit wins
    assert normal.loc['4', 'review_text'].startswith('배송도')           # older version ignored
    change = pd.read_parquet(out / 'changes.parquet').set_index('review_id').change.to_dict()
    assert change == {'2': 'updated', '3': 'removed', '7': 'new'}
    with pytest.raises(SystemExit):                                     # the same CSV twice
        P.add(tmp / 'second.csv')
    P.rebuild()                                                          # no new CSV: same files, changes kept
    assert pd.read_parquet(out / 'changes.parquet').set_index('review_id').change.to_dict() == change
