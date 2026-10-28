import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preprocess', ROOT / '1_preprocessing' / 'preprocess.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

PASTED = '배송도 아주 빠르게 잘왔구 가격도 상품도 다좋네요 진짜 ~~~~~잘먹었습니다 또 살게요'


def raw_csv(tmp_path):
    rows = [  # 리뷰글번호, 상품번호, 상품명, 리뷰구분, 관련리뷰글번호, 전시상태, 리뷰등록일, 본문
        ('1', 'p1', '새우 A', '일반', '', '정상', '2025.01.01. 10:00:00', '맛있어요&hellip;'),
        ('2', 'p1', '새우 A', '한달사용', '1', '정상', '2025.02.01. 10:00:00', '한 달 먹어보니 좋아요~~~~~'),
        ('3', 'p1', '새우 B', '일반', '', '정상', '2025.03.01. 10:00:00', PASTED),
        ('4', 'p1', '새우 A', '일반', '', '정상', '2025.04.01. 10:00:00', PASTED.replace('~~~~~', '~~~~~~~')),
        ('5', 'p1', '새우 A', '일반', '', '블라인드', '2025.04.02. 10:00:00', '가려진 리뷰'),
        ('6', 'p2', '고등어', '일반', '', '정상', '2025.01.05. 10:00:00', PASTED),
    ]
    frame = pd.DataFrame(rows, columns=['리뷰글번호', '상품번호', '상품명', '리뷰구분', '관련리뷰글번호', '전시상태', '리뷰등록일', '리뷰상세내용'])
    frame['구매자평점'] = '5'
    frame['최종수정일'] = ''
    path = tmp_path / 'raw.csv'
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def test_preprocess_rules(tmp_path):
    out = tmp_path / 'out'
    counts, mapping = P.build(raw_csv(tmp_path), out)
    normal = pd.read_parquet(out / 'reviews_normal.parquet')
    text = pd.read_parquet(out / 'analysis_text.parquet').set_index('review_id')
    assert counts['blind_reviews'] == 1 and counts['normal_reviews'] == 5
    assert normal.columns.tolist() == P.NORMAL_COLUMNS
    assert normal.review_id.tolist() == ['1', '2', '3', '4', '6']  # product, time, review
    group = normal.set_index('review_id').review_group_id
    assert group['2'] == group['1'] == 'review:1'  # 한달사용 follow-up joins its original
    assert text.loc['1', 'analysis_text'] == '맛있어요…'
    assert text.loc['2', 'analysis_text'] == '한 달 먹어보니 좋아요~~'
    # the same long text pasted again into the same product is the first copy's purchase; another product is not
    assert text.loc['4', 'purchase_id'] == text.loc['3', 'purchase_id'] == 'review:3'
    assert text.loc['6', 'purchase_id'] == 'review:6'
    assert counts['purchases'] == 3
    assert mapping.set_index('product_id').canonical_product_name['p1'] == '새우 A'  # most frequent name
