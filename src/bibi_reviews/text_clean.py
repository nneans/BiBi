"""Analysis text (v2): what Kiwi units, the models and the reports read.

backbone.normalize_text stays as it is: it is the DB text, its hash finds new/edited reviews, the 300 gold
reviews' offsets and the post-training data are on it. clean_review works from the raw review and gives the same
text plus, measured on the 642,172 normal reviews (2_preprocessing/check_clean.py):
  - line breaks kept (15% of reviews have them): a break after a sentence end is a unit boundary, a break inside
    a sentence ('떼어내느라\\n고생했어요') is a wrapped line; kiwi_units decides which (is_line_end)
  - invisible characters removed (zero-width space/joiner, BOM, soft hyphen ...); emoji variation selectors stay
  - old-hangul dots used as punctuation by some keyboards: 'ᆢ' -> '..', 'ᆞ' -> '.'
  - links removed (blog addresses), runs of 10+ digits removed ('1234567890' typed to reach a length limit,
    phone numbers; 8-digit dates like 20250211 stay)
  - a character repeated 3+ times kept twice ('좋아요~~~~~~~~' -> '좋아요~~', 'ㅋㅋㅋㅋ' -> 'ㅋㅋ'), as in
    KcELECTRA's own data ('ㅋㅋㅋㅋㅋ' -> 'ㅋㅋ'); digits are never shortened ('1000원'); spaced dot runs
    ('굿 . . .') become '..'
Kept on purpose: emoji, ♡, ㅠㅠ, ㅡㅡ, ^^ (KcELECTRA learned them: emoji embeddings are as trained as hangul
ones and sit next to their meaning, 😭 next to 😢 and ㅠㅠ), spelling and spacing (evidence must stay the
customer's own words).
"""
import html
import re
import unicodedata

CLEAN_VERSION = 2

CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
INVISIBLE = re.compile('[­͏؜ᅟᅠ᠎​-‏‪-‮⁠-⁤ㅤ﻿ﾠ]')
OLD_DOTS = {'ᆢ': '..', 'ᆞ': '.'}
LINK = re.compile(r'(?:https?://|www\.)\S+')
LONG_DIGITS = re.compile(r'(?<!\d)\d{10,}(?!\d)')
SPACED_DOTS = re.compile(r'\.(?:[ \t]*\.){2,}')
REPEAT = re.compile(r'([^\d\s])\1{2,}')
INLINE_SPACE = re.compile(r'[^\S\n]+')
LINE_BREAKS = re.compile(r' *\n[\s]*')


def clean_review(raw):
    text = unicodedata.normalize('NFC', html.unescape(str(raw)))
    text = CONTROL.sub('', text.replace('\r\n', '\n').replace('\r', '\n'))
    text = INVISIBLE.sub('', text)
    for old, new in OLD_DOTS.items():
        text = text.replace(old, new)
    text = LONG_DIGITS.sub(' ', LINK.sub(' ', text))
    text = REPEAT.sub(r'\1\1', SPACED_DOTS.sub('..', text))
    text = LINE_BREAKS.sub('\n', INLINE_SPACE.sub(' ', text))
    return text.strip()
