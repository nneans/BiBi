"""Project folder layout (2026-10-23). Every script takes its locations from here, so a rename touches one file.

1_전처리     원본 CSV -> 정리된 리뷰        2_DB          PostgreSQL 백업 · 증분 반영
3_사후학습   식품 리뷰 적응 KcELECTRA      4_정답데이터   정답 300건 · 같은 뜻 판정
5_감성분류   Kiwi 단위 + 감성 분류기        6_이유분석     KURE + BERTopic, 상품별 결과
7_보고       집계 · 그림 · 대시보드 · 현황
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PREPROCESS = ROOT / '1_전처리'
RAW = PREPROCESS / '원본'
RAW_CSV = RAW / '리뷰데이터(2025.01.01~2026.10.13).csv'
PREPARED = PREPROCESS / '결과'
REVIEWS_NORMAL = PREPARED / 'processed' / 'reviews_normal.parquet'
CANONICAL_DIR = PREPARED / 'processed' / 'canonical_products'
CANONICAL_REVIEWS = CANONICAL_DIR / 'reviews_normal_canonical.parquet'

DB = ROOT / '2_DB'
DB_FILES = DB / 'db'

BACKBONE = ROOT / '3_사후학습'
BACKBONE_ENCODER = BACKBONE / 'frozen-encoder'
BASE_KCELECTRA = (Path.home() / '.cache/huggingface/hub/models--beomi--KcELECTRA-base-v2022/snapshots/'
                  '568fe985a10c013c8f605e3c80a199413bbe9d00')

GOLD_DIR = ROOT / '4_정답데이터'
GOLD = GOLD_DIR / '정답300' / 'gold-300-final.json'
SAME_MEANING = GOLD_DIR / '같은뜻판정' / 'judgments.json'

SENTIMENT = ROOT / '5_감성분류'
UNITS = SENTIMENT / '단위나누기'
CLASSIFIER = SENTIMENT / '분류기'

REASONS = ROOT / '6_이유분석'
KURE_V1 = (Path.home() / '.cache/huggingface/hub/models--nlpai-lab--KURE-v1/snapshots/'
           '8b418a58414668e75532ed045c22d9ca018ae2b2')

REPORT = ROOT / '7_보고'
STATISTICS = REPORT / '집계'
FIGURES = REPORT / '그림'
