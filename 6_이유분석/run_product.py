"""Tool-based pipeline, one product end to end.

normalize_text -> Kiwi units (sentence + clause) -> unit sentiment classifier (food-adapted KcELECTRA,
fine-tuned on all 739 gold units, variant a) -> KURE-v1 embeddings of unique unit texts -> BERTopic per
sentiment (UMAP -> HDBSCAN, Kiwi c-TF-IDF, no auto merge) -> counts per topic, recent vs earlier.
Usage: python run_product.py [product_id]   (default 296889036). Results go to results/<product_id>/.
       python run_product.py --all          every unfinished product in products.json, in one process
"""
import sys

PRODUCT = sys.argv[1] if len(sys.argv) > 1 else '296889036'
# umap imports TensorFlow only for ParametricUMAP, which is unused; blocking it keeps a large DLL set out of the
# process (this PC crashes at random, and every extra native library is another place to crash).
sys.modules['tensorflow'] = None
from bibi_reviews.cpu import pin_to_efficiency_cores  # noqa: E402
pin_to_efficiency_cores()  # before numba/torch load: this PC's P-cores are unstable (see bibi_reviews/cpu.py)

import csv  # noqa: E402
import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from bertopic import BERTopic  # noqa: E402
from hdbscan import HDBSCAN  # noqa: E402
from kiwipiepy import Kiwi  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402
from umap import UMAP  # noqa: E402

OUT = Path(__file__).resolve().parent
from bibi_reviews.paths import CLASSIFIER, UNITS, CANONICAL_DIR, CANONICAL_REVIEWS, KURE_V1  # noqa: E402
for p in [CLASSIFIER, UNITS]:
    sys.path.insert(0, str(p))
from cv_unit_sentiment import build_units, encode, ENCODER, LABELS, SEED, EPOCHS, LR, BATCH, WARMUP  # noqa: E402
from cv_extra_units import extra_units  # noqa: E402
from merge_groups import merge  # noqa: E402
from summarize_all import is_specific  # noqa: E402
from kiwi_units import units as kiwi_units  # noqa: E402
from bibi_reviews.backbone import normalize_text  # noqa: E402

KURE = KURE_V1
# v3 (2026-10-26): classifier on gold units + the 799 user-confirmed extra units (CV: gold acc 0.862, NEG precision on
# real product units 0.76), dangling-particle join in kiwi_units, same-meaning groups merged here (merge_groups.py),
# and every NEG/POS unit stores its final group. Runs on the E-cores (bibi_reviews/cpu.py).
RUN = 'v3'
MODEL_DIR = OUT / f'models/unit-sentiment-{RUN}'
RESULTS_ROOT = OUT / f'results-{RUN}'
# Report version 3: groups merged (merge_groups.py) and recorded per unit in units.jsonl ('group').
# Report version 5: membership by distance to the group centre (MEMBER_MIN): far members leave, unclustered texts
# close to a centre join. Version 4 compared with 3 central texts only and left 33-72% of complaints unclustered
# ('저번보다 크기는 살짝 작은데' sat at 0.89 from the size centre). Garbled texts are not representatives.
REPORT_VERSION = 5


def finished(product):
    path = RESULTS_ROOT / product / 'report.json'
    return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('report_version') == REPORT_VERSION
KEEP_TAGS = {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'}
RECENT_MONTHS = 3
# Units below this classifier confidence go to a re-check queue instead of topics and counts (any label).
# 0.8 was read off the cheese-stick NEG confidence profile (2026-10-23); re-check it against gold later.
REVIEW_THRESHOLD = 0.8
HAS_LETTER = re.compile(r'[가-힣A-Za-z]')
# Below this many unique texts UMAP/HDBSCAN are not meaningful; the sentiment is reported as one ungrouped list.
MIN_DOCS_FOR_TOPICS = 50
# A text belongs to a group when its KURE cosine to the group centre (count-weighted mean of the members) reaches
# this. Mackerel complaints, 2026-10-26: 90% of HDBSCAN's own members sit at >= 0.78; below 0.75 the texts were
# mostly off-topic ('4토막 이네요'), above it mostly right ('작은느낌임' -> size).
MEMBER_MIN = 0.75
# a lone vowel jamo inside the text = a typo like '규ㅐㄴ찮읔듯요' (ㅠ/ㅜ are crying faces, not typos)
GARBLED = re.compile(r'[ㅏ-ㅛㅡ-ㅣ]|[ㄱ-ㅎ](?=[가-힣])')
kiwi = Kiwi()

NAME_CHARS = re.compile(r'[^가-힣A-Za-z0-9]')
NAME_WORDS = re.compile(r'[가-힣A-Za-z0-9]+')


def product_names(product_id):
    """Every name the product was sold under (canonical + original), for dropping units that only repeat it."""
    with (CANONICAL_DIR / 'product_name_aliases.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['product_id'] == product_id]
    return {n for r in rows for n in (r['product_name_original'], r['canonical_product_name']) if n}


def is_name_only(unit, names):
    """'손질 생 새우살 900g 깐새우' copied from the title is not an opinion; '... 새우살 좋습니다' keeps its evaluation."""
    flat = NAME_CHARS.sub('', unit)
    if len(flat) < 6:
        return False
    words = set(NAME_WORDS.findall(unit))
    return (any(flat in NAME_CHARS.sub('', n) for n in names)
            or words <= set().union(*(NAME_WORDS.findall(n) for n in names)))


def read_reviews(columns, product_id=None):
    """Canonical normal reviews; tz-aware timestamps are cast to text (to_pylist on them crashes pyarrow)."""
    source = CANONICAL_REVIEWS
    filtering = product_id is not None and 'product_id' not in columns
    table = pq.read_table(source, columns=columns + ['product_id'] if filtering else columns)
    if product_id is not None:
        table = table.filter(pc.equal(table['product_id'], product_id))
        if filtering:
            table = table.drop_columns(['product_id'])
    if 'created_at' in columns:
        table = table.set_column(table.schema.get_field_index('created_at'), 'created_at',
                                 pc.cast(table['created_at'], pa.string()))
    return table.to_pylist()


def kiwi_tokens(text):
    return [t.form + ('다' if t.tag in ('VV', 'VA') else '') for t in kiwi.tokenize(text) if t.tag in KEEP_TAGS]


def train_classifier(tokenizer):
    if (MODEL_DIR / 'config.json').exists():
        return AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True).cuda().eval()
    rows = build_units()[0] + extra_units()
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(str(ENCODER), num_labels=len(LABELS), local_files_only=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(rows) + BATCH - 1) // BATCH)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * steps), steps)
    rng = random.Random(SEED)
    model.train()
    for _ in range(EPOCHS):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for k in range(0, len(order), BATCH):
            batch = [rows[i] for i in order[k:k + BATCH]]
            enc = {n: t.cuda() for n, t in encode(tokenizer, batch, 'a').items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**enc, labels=torch.tensor([r['label'] for r in batch]).cuda()).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    return model


def topics_for(texts_with_counts, embedder):
    docs = list(texts_with_counts)
    if len(docs) < MIN_DOCS_FOR_TOPICS:
        return docs, np.full(len(docs), -1), None, 0, None
    emb = embedder.encode(docs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    size = max(15, round(0.01 * len(docs)))
    model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=size, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
        vectorizer_model=CountVectorizer(tokenizer=kiwi_tokens, lowercase=False, min_df=2, token_pattern=None),
        top_n_words=8, nr_topics=None, verbose=False)
    topics, _ = model.fit_transform(docs, emb)
    return docs, np.array(topics), model, size, emb


def write_jsonl(path, rows):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)


def classify(reviews, tokenizer, classifier, names):
    rows = []
    for r in reviews:
        text = normalize_text(r['review_text_raw'])
        for s, e in kiwi_units(kiwi, text, True):
            # units like '..' or 'ㅋㅋ' carry no opinion, nor a copy of the product title
            if HAS_LETTER.search(text[s:e]) and not is_name_only(text[s:e], names):
                rows.append({'review_id': r['review_id'], 'month': r['created_at'][:7], 'unit': text[s:e]})
    with torch.inference_mode():
        for k in range(0, len(rows), 128):
            batch = rows[k:k + 128]
            enc = {n: t.cuda() for n, t in encode(tokenizer, batch, 'a').items()}
            prob = torch.softmax(classifier(**enc).logits.float(), -1).cpu()
            for row, p in zip(batch, prob):
                row['pred'] = LABELS[int(p.argmax())]
                row['confidence'] = round(float(p.max()), 3)
                row['label'] = row['pred'] if row['confidence'] >= REVIEW_THRESHOLD else 'REVIEW'
    return rows


def central_texts(texts, row_of, emb, k=3):
    """The group's own wording: texts nearest the count-weighted embedding centroid, preferring ones long enough
    to read alone. The most frequent text was a poor label ('냉동고 자리가;;' for a [맛·없다·바삭] group)."""
    if emb is None:
        return []
    names = list(texts)
    vecs = emb[[row_of[x] for x in names]]
    centre = (vecs * np.array([texts[x] for x in names])[:, None]).sum(0)
    order = np.argsort(-(vecs @ (centre / np.linalg.norm(centre))))
    readable = [names[i] for i in order if len(names[i].replace(' ', '')) >= 8 and not GARBLED.search(names[i])]
    return (readable or [names[i] for i in order])[:k]


def keep_close_members(docs, topics, counts, emb):
    """Re-assign every text by distance to the group centres: a text joins the nearest centre at >= MEMBER_MIN,
    otherwise it is unclustered (-1). Members far from their own centre leave ('편해요' in a freshness group);
    HDBSCAN outliers that sit close to a centre join it."""
    names = sorted(set(topics.tolist()) - {-1})
    if emb is None or not names:
        return topics
    weight = np.array([counts[d] for d in docs], dtype=float)
    centres = np.stack([(emb[topics == t] * weight[topics == t, None]).sum(0) for t in names])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    sim = emb @ centres.T
    best = sim.argmax(1)
    return np.where(sim.max(1) >= MEMBER_MIN, np.array(names)[best], -1)


def group_label(rows, label, recent, recent_reviews, earlier_reviews, embedder):
    sel = [r for r in rows if r['label'] == label]
    key = lambda r: re.sub(r'\s+', ' ', r['unit']).strip()
    counts = Counter(key(r) for r in sel)
    docs, topics, model, size, emb = topics_for(counts, embedder)
    topics = keep_close_members(docs, topics, counts, emb)
    row_of = {d: i for i, d in enumerate(docs)}
    topic_of = dict(zip(docs, topics.tolist()))
    groups = defaultdict(list)
    for r in sel:
        groups[topic_of[key(r)]].append(r)
    out = []
    for t, members in groups.items():
        texts = Counter(key(m) for m in members)
        recent_n = sum(m['month'] in recent for m in members)
        out.append({'topic': int(t), 'units': len(members), 'share': round(len(members) / len(sel), 3),
                    'unique_texts': len(texts),
                    'keywords': [w for w, _ in model.get_topic(t)][:8] if t != -1 else [],
                    'top_texts': [f'{x} ({n})' for x, n in texts.most_common(5)],
                    'representative': central_texts(texts, row_of, emb) if t != -1 else [],
                    'per_100_reviews_recent': round(100 * recent_n / max(recent_reviews, 1), 2),
                    'per_100_reviews_earlier': round(100 * (len(members) - recent_n) / max(earlier_reviews, 1), 2)})
    out.sort(key=lambda g: (g['topic'] == -1, -g['units']))
    return {'units': len(sel), 'unique_texts': len(docs), 'min_cluster_size': size,
            'topics': len([g for g in out if g['topic'] != -1]),
            'outlier_share': round(sum(g['units'] for g in out if g['topic'] == -1) / max(len(sel), 1), 3),
            'groups': out, 'text_topic': topic_of}


def merge_same_meaning(result, embedder, product_name):
    """Merge same-meaning groups (threshold calibrated on the 148 judged pairs; specific and generic kept apart)
    and point every text at its merged group."""
    groups, log = merge(result['groups'], lambda xs: embedder.encode(xs, normalize_embeddings=True, show_progress_bar=False),
                        kind=lambda g: is_specific(g, product_name))
    into = {t: g['topic'] for g in groups for t in g.get('merged_from', [g['topic']])}
    return {**result, 'groups': groups, 'topics': len([g for g in groups if g['topic'] != -1]), 'merges': log,
            'text_topic': {x: into.get(t, t) for x, t in result['text_topic'].items()}}


def main(product):
    results = RESULTS_ROOT / product
    start = time.perf_counter()
    results.mkdir(parents=True, exist_ok=True)
    reviews = read_reviews(['review_id', 'product_name', 'rating', 'created_at', 'review_text_raw'], product)
    # Every finished stage is saved, so a crash-retry resumes where it stopped instead of starting over.
    units_path = results / 'units.jsonl'
    if units_path.exists():
        rows = [json.loads(line) for line in units_path.read_text(encoding='utf-8').splitlines()]
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
        classifier = train_classifier(tokenizer)
        print(f'classifier ready | {(time.perf_counter() - start) / 60:.1f} min', flush=True)
        rows = classify(reviews, tokenizer, classifier, product_names(product))
        write_jsonl(units_path, rows)
        del classifier
        torch.cuda.empty_cache()
    queue = sorted((r for r in rows if r['label'] == 'REVIEW'), key=lambda r: r['confidence'])
    write_jsonl(results / 'review-queue.jsonl', queue)
    print(f'{len(reviews)} reviews -> {len(rows)} units {dict(Counter(r["label"] for r in rows))} | '
          f'{(time.perf_counter() - start) / 60:.1f} min', flush=True)

    months = sorted({r['month'] for r in rows})
    recent = set(months[-RECENT_MONTHS:])
    review_month = {r['review_id']: r['created_at'][:7] for r in reviews}
    recent_reviews = sum(m in recent for m in review_month.values())
    earlier_reviews = len(review_month) - recent_reviews
    report = {'product_id': product, 'product_name': reviews[0]['product_name'], 'reviews': len(reviews),
              'units': len(rows), 'unit_labels': dict(Counter(r['label'] for r in rows)),
              'period': [months[0], months[-1]], 'recent_months': sorted(recent), 'recent_reviews': recent_reviews,
              'pipeline': 'normalize_text -> Kiwi units (letterless and title-copy units dropped) -> KcELECTRA unit classifier -> confidence < '
                          f'{REVIEW_THRESHOLD} to re-check queue -> KURE-v1 -> BERTopic (unique texts, no auto merge)',
              'review_threshold': REVIEW_THRESHOLD,
              'review_queue': {'units': len(queue), 'by_predicted_label': dict(Counter(r['pred'] for r in queue))},
              'report_version': REPORT_VERSION, 'groups': {}, 'checks': {}}
    embedder = None
    for label in ['NEG', 'POS']:
        done = results / f'groups-{label}-r{REPORT_VERSION}.json'
        if not done.exists():
            embedder = embedder or SentenceTransformer(str(KURE), device='cuda', local_files_only=True)
            result = group_label(rows, label, recent, recent_reviews, earlier_reviews, embedder)
            result = merge_same_meaning(result, embedder, report['product_name'])
            done.write_text(json.dumps(result, ensure_ascii=False) + '\n', encoding='utf-8')
        g = json.loads(done.read_text(encoding='utf-8'))
        text_topic = g.pop('text_topic')
        report['groups'][label] = g
        for r in rows:
            if r['label'] == label:
                r['group'] = text_topic[re.sub(r'\s+', ' ', r['unit']).strip()]
        print(f"{label}: {g['units']} units ({g['unique_texts']} unique) -> {g['topics']} topics, "
              f"기타 {g['outlier_share']:.1%} | {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    receipt = [r for r in rows if re.fullmatch(r'\s*(잘\s*받았(어요|습니다|어용|음)|잘\s*받았습니다\s*감사합니다)[\s.!~^]*', r['unit'])]
    report['checks'] = {
        'plain_receipt_by_label': dict(Counter(r['label'] for r in receipt)),
        'html_entities_left_after_normalize': sum(bool(re.search(r'&[a-z]+;', r['unit'])) for r in rows)}
    print(json.dumps(report['checks'], ensure_ascii=False)[:600], flush=True)
    report['merged'] = True
    write_jsonl(units_path, rows)  # now with each NEG/POS unit's final group
    report['seconds'] = time.perf_counter() - start
    (results / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_all():
    """One process for many products: numba compiles UMAP once instead of once per product (most crashes on this
    PC happen while compiling). Python-level errors skip to the next product; a hard crash ends the process and
    run_all.sh starts it again."""
    products = [p['id'] for p in json.loads((OUT / 'products.json').read_text(encoding='utf-8'))]
    # Tries per product survive hard crashes; the fewest-tried product goes first, so one product that keeps
    # crashing (the 83k-review one, at its POS clustering) drifts to the end instead of blocking all others.
    tries_path = RESULTS_ROOT / 'tries.json'
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    tries = json.loads(tries_path.read_text(encoding='utf-8')) if tries_path.exists() else {}
    for attempt in range(3):
        pending = [p for p in products if not finished(p)]
        if not pending:
            break
        for product in sorted(pending, key=lambda p: tries.get(p, 0)):
            tries[product] = tries.get(product, 0) + 1
            tries_path.write_text(json.dumps(tries), encoding='utf-8')
            try:
                main(product)
                print(f'ok {product}', flush=True)
            except Exception:
                traceback.print_exc()
                print(f'error {product} (attempt {attempt + 1})', flush=True)


if __name__ == '__main__':
    run_all() if PRODUCT == '--all' else main(PRODUCT)
