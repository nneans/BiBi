import sys
from pathlib import Path

import pytest
from bibi_reviews.text_clean import clean_review

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '5_감성분류' / '단위나누기'))


@pytest.mark.parametrize('raw,expected', [
    ('맛있어요&hellip;', '맛있어요…'),
    ('아주 좋아요~~~~~~~~~~~', '아주 좋아요~~'),
    ('맛있어요ㅋㅋㅋㅋㅋ', '맛있어요ㅋㅋ'),
    ('............', '..'),
    ('굿  .  .  .', '굿 ..'),
    ('1234567890', ''),
    ('1000원에 샀고 포장일 20250211', '1000원에 샀고 포장일 20250211'),
    ('불편하긴한데ᆢ좀녹여서', '불편하긴한데..좀녹여서'),
    ('​명란젓은﻿', '명란젓은'),
    ('괜츈해요  \r\n\r\n  많이 파세요 ', '괜츈해요\n많이 파세요'),
    ('https://m.blog.naver.com/x/1 사진 참고', '사진 참고'),
    ('하트❤️ 😭 ㅠㅠ ㅡㅡ ^^', '하트❤️ 😭 ㅠㅠ ㅡㅡ ^^'),
])
def test_clean_review(raw, expected):
    assert clean_review(raw) == expected


def test_line_breaks_split_only_at_sentence_ends():
    from kiwipiepy import Kiwi
    from kiwi_units import units
    kiwi = Kiwi()
    pieces = lambda t: [t[s:e] for s, e in units(kiwi, t, True)]  # noqa: E731
    assert pieces('간식으로 좋아요\n많이 파세요') == ['간식으로 좋아요', '많이 파세요']
    assert pieces('달라붙어서 떼어내느라\n고생했어요') == ['달라붙어서 떼어내느라\n고생했어요']
    assert pieces('몇번째 구매인줄 모를정도로\n자주 사먹어요') == ['몇번째 구매인줄 모를정도로\n자주 사먹어요']
    text = '간식으로 좋아요 많이 파세요'
    assert pieces(text) == pieces(text.replace(' 많', ' 많'))  # no line break: same as before
