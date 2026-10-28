"""Project folder layout (2026-10-23). Every script takes its locations from here, so a rename touches one file.

1_ingest         CSV -> 리뷰 DB(원본 장부)  2_preprocessing DB -> 분석용 리뷰 파일
3_kcelectra      KcELECTRA 사후학습(식품 리뷰 적응)  4_정답데이터   정답 300건 · 같은 뜻 판정
5_감성분류   Kiwi 단위 + 감성 분류기        5_속성분류     속성 사전 + 속성 분류기 (감성과 나란히)
6_이유분석   KURE + BERTopic, 상품별 결과   7_보고         집계 · 그림 · 대시보드 · 현황
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# step 1: every delivered CSV (raw/, listed in raw/deliveries.json) goes into the review DB, the ledger
INGEST = ROOT / '1_ingest'
RAW = INGEST / 'raw'
DELIVERIES = RAW / 'deliveries.json'
RAW_CSV = RAW / '리뷰데이터(2025.01.01~2026.10.13).csv'  # the first delivery
RAW_AUDIT = RAW / 'audit'
DB = INGEST
DB_FILES = DB / 'db'  # PostgreSQL backups, web uploads

# step 2: the DB's reviews -> analysis files (2_preprocessing/preprocess.py)
PREPROCESS = ROOT / '2_preprocessing'
PROCESSED = PREPROCESS / 'output'
REVIEWS_NORMAL = PROCESSED / 'reviews_normal.parquet'
CANONICAL_DIR = PROCESSED / 'canonical_products'
CANONICAL_REVIEWS = CANONICAL_DIR / 'reviews_normal_canonical.parquet'
PRODUCT_NAMES = CANONICAL_DIR / 'product_name_mapping.csv'  # every product_id, its display name and review count
ANALYSIS_TEXT = PROCESSED / 'analysis_text.parquet'  # review_id -> analysis_text (text_clean v2), purchase_id

# not official: drafts and old records (archive/README.md)
ARCHIVE = ROOT / 'archive'
PRODUCT_CATEGORIES = ARCHIVE / 'preprocessing' / 'category-draft' / 'product-categories.json'  # draft 상품군

# step 3: KcELECTRA post-trained on our reviews; frozen-encoder/ is what every later step loads
BACKBONE = ROOT / '3_kcelectra'
BACKBONE_ENCODER = BACKBONE / 'frozen-encoder'
BASE_KCELECTRA = (Path.home() / '.cache/huggingface/hub/models--beomi--KcELECTRA-base-v2022/snapshots/'
                  '568fe985a10c013c8f605e3c80a199413bbe9d00')

GOLD_DIR = ROOT / '4_정답데이터'
GOLD = GOLD_DIR / '정답300' / 'gold-300-final.json'
SAME_MEANING = GOLD_DIR / '같은뜻판정' / 'judgments.json'

SENTIMENT = ROOT / '5_감성분류'
UNITS = SENTIMENT / '단위나누기'
CLASSIFIER = SENTIMENT / '분류기'

ASPECT = ROOT / '5_속성분류'
ASPECT_DICTIONARY = ASPECT / 'aspects.json'

REASONS = ROOT / '6_이유분석'
# LoRA + head pairs on the shared frozen encoder (bibi_reviews.lora_pairs)
PAIR_SENTIMENT = REASONS / 'models' / 'pair-sentiment'
PAIR_ASPECT = REASONS / 'models' / 'pair-aspect'
KURE_V1 = (Path.home() / '.cache/huggingface/hub/models--nlpai-lab--KURE-v1/snapshots/'
           '8b418a58414668e75532ed045c22d9ca018ae2b2')

REPORT = ROOT / '7_보고'
STATISTICS = REPORT / '집계'
FIGURES = REPORT / '그림'
