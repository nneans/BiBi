"""Builds and runs 리뷰_한_건_따라가기.ipynb: one new review through the whole pipeline, with every step's output.

Usage: python make_notebook.py   (writes the notebook next to this file and executes it in place)
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NB = HERE / '리뷰_한_건_따라가기.ipynb'

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 리뷰 한 건 따라가기

새 리뷰 한 건이 들어오면 어떤 순서로 처리되는지, 단계마다 실제 출력을 보면서 따라갑니다.

```
원문 리뷰
 → ① 전처리 (normalize_text: HTML 기호, 공백 정리)
 → ② Kiwi: 형태소 분석 → 문장 나누기 → 연결어미에서 절 나누기 → 토막 합치기  = "단위"
 → ③ KcELECTRA(식품 리뷰로 사후학습 + 단위 감성 미세조정): 단위마다 긍정 / 부정 / 평가 아님
 → ④ KURE-v1: 단위 문장을 1,024차원 벡터로 (뜻이 비슷하면 가까움)
 → ⑤ BERTopic: 상품의 같은 감성 단위들을 UMAP → HDBSCAN으로 묶고, c-TF-IDF로 키워드
 → ⑥ 새 단위를 가장 가까운 묶음에 배정 (유사도 0.75 이상일 때만)
 → ⑦ 집계: 묶음별 건수, 최근 3개월 vs 그 이전 (리뷰 100건당)
```

예시 상품은 **노르웨이고등어**(상품 ID `300868056`, 리뷰 2만 건)입니다. ⑤의 묶음은 이 상품의 실제 리뷰로 이미 만들어 둔 결과(`6_이유분석/results-v3`)를 씁니다.
""")

code(r"""
import os, sys, json, re
os.environ.update(USE_TF='0', TRANSFORMERS_NO_TF='1', TOKENIZERS_PARALLELISM='false')
sys.modules['tensorflow'] = None          # umap이 쓰지도 않는 TensorFlow를 불러오지 않게
from pathlib import Path
# 프로젝트 맨 위 폴더 찾기: VS Code는 노트북이 있는 폴더(7_보고/노트북)에서 실행하므로 위로 올라가며 찾음
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'bibi_reviews').is_dir())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '5_감성분류/단위나누기'), str(ROOT / '5_감성분류/분류기'), str(ROOT / '6_이유분석')]

import bibi_reviews                       # 불러오는 순간 이 PC의 효율 코어에만 고정됨 (bibi_reviews/cpu.py)
from bibi_reviews.paths import KURE_V1
from bibi_reviews.backbone import normalize_text
import numpy as np, pandas as pd, torch
pd.set_option('display.max_colwidth', 80)
PRODUCT = '300868056'
RESULTS = ROOT / '6_이유분석/results-v3' / PRODUCT
print('준비 완료 | GPU:', torch.cuda.get_device_name(0))
""")

md("""
## 0. 새 리뷰

실제 리뷰처럼 칭찬과 불만이 섞인 문장입니다. 일부러 `&hellip;`(말줄임표 HTML 기호)와 겹친 공백을 넣었습니다. 원본 CSV에는 이런 기호가 섞여 들어옵니다.
""")
code(r"""
raw = "가시가 없어서 아이들 먹이기 편해요&hellip; 그런데 이번엔 크기가 좀 작고   살이 퍽퍽했어요. 비린내는 거의 안 나요! 배송은 빨랐는데 아이스팩이 다 녹아서 왔네요ㅠ 다음엔 특대로 시켜볼게요"
print(raw)
""")

md("""
## ① 전처리

`normalize_text`는 HTML 기호를 원래 문자로 되돌리고 공백을 정리합니다. **원문의 뜻은 바꾸지 않습니다**(오타를 고치지 않음). 결과의 근거 문장이 실제 리뷰와 같아야 하기 때문입니다.
""")
code(r"""
text = normalize_text(raw)
print('전:', raw)
print('후:', text)
""")

md("""
## ② Kiwi로 단위 나누기

### ②-1 형태소 분석
Kiwi는 문장을 형태소와 품사로 나눕니다. 단위를 자르는 기준이 되는 품사는 `EC`(연결어미: ~는데, ~지만, ~고, ~면서)입니다.
""")
code(r"""
from kiwipiepy import Kiwi
kiwi = Kiwi()
toks = kiwi.tokenize(text)
pd.DataFrame([(t.form, t.tag, t.start) for t in toks], columns=['형태소', '품사', '위치']).T
""")

md("""
### ②-2 문장 나누기 → 절 나누기 → 토막 합치기
1. 문장 단위로 나눕니다.
2. 문장 안에서 `-는데`, `-지만`, `-고`, `-면서` 같은 연결어미 뒤를 자릅니다. 단, 뒤에 보조동사가 오면('먹고 **있어요**') 자르지 않습니다.
3. 서술어 없이 조사로 끝나는 토막('옆지느러미가.')은 다음 단위에 붙입니다.
""")
code(r"""
from kiwi_units import units, is_clause_end, CLAUSE_EC
for s in kiwi.split_into_sents(text):
    sent_toks = kiwi.tokenize(s.text)
    cuts = [t.form for i, t in enumerate(sent_toks) if is_clause_end(sent_toks, i, CLAUSE_EC)]
    print(f'문장: {s.text}\n   자르는 곳(연결어미): {cuts}')
spans = units(kiwi, text, True)
unit_texts = [text[s:e] for s, e in spans]
pd.DataFrame({'단위': unit_texts, '원문 위치': [f'{s}~{e}' for s, e in spans]})
""")

md("""
## ③ KcELECTRA: 단위마다 감성 판정

- 바탕 모델은 `beomi/KcELECTRA-base-v2022`(한국어 댓글로 사전학습)입니다. 이 모델을 우리 리뷰 49.5만 건으로 **사후학습**해 식품 리뷰 표현에 익숙하게 만들었어요(`3_사후학습/frozen-encoder`).
- 그 위에 **감성 분류 미세조정**을 했어요. 정답 300건에서 나온 단위와, 30개 상품에서 뽑아 사람이 확인한 단위 799개를 썼습니다(`6_이유분석/models/unit-sentiment-v3`).
- 확신도가 0.8보다 낮으면 "재확인"으로 따로 모읍니다.

### ③-1 토큰화
KcELECTRA는 Kiwi와 별개로 자기 사전(WordPiece)으로 글자를 쪼개서 읽습니다.
""")
code(r"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = ROOT / '6_이유분석/models/unit-sentiment-v3'
tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
clf = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True).cuda().eval()
for u in unit_texts[:3]:
    print(f'{u}\n   → {tokenizer.tokenize(u)}')
""")
code(r"""
LABELS = {0: '긍정', 1: '부정', 2: '평가 아님'}
enc = tokenizer(unit_texts, truncation=True, max_length=128, padding=True, return_tensors='pt').to('cuda')
with torch.inference_mode():
    prob = torch.softmax(clf(**enc).logits.float(), -1).cpu().numpy()
judged = pd.DataFrame(prob.round(3), columns=['긍정', '부정', '평가 아님'])
judged.insert(0, '단위', unit_texts)
judged['판정'] = [LABELS[i] if p.max() >= 0.8 else f'재확인({LABELS[i]}?)' for i, p in zip(prob.argmax(1), prob)]
judged['확신도'] = prob.max(1).round(3)
judged
""")

md("""
## ④ KURE-v1: 문장을 벡터로

KURE-v1은 문장의 **뜻**을 1,024개 숫자로 바꿉니다. 뜻이 비슷한 문장은 벡터가 가깝습니다(코사인 유사도 1에 가까움). 아래 표는 단위끼리의 유사도입니다.
""")
code(r"""
from sentence_transformers import SentenceTransformer
kure = SentenceTransformer(str(KURE_V1), device='cuda', local_files_only=True)
vec = kure.encode(unit_texts, normalize_embeddings=True)
print('벡터 크기:', vec.shape)
pd.DataFrame((vec @ vec.T).round(2), index=unit_texts, columns=[u[:8] for u in unit_texts]).style.background_gradient(cmap='Blues', axis=None)
""")

md("""
## ⑤ BERTopic: 상품의 단위들은 어떻게 묶였나

상품 하나의 **같은 감성** 단위를 모아 묶습니다(불만은 불만끼리, 칭찬은 칭찬끼리).

1. **KURE 벡터** (④와 같음)
2. **UMAP**: 1,024차원을 5차원으로 줄여 가까운 것끼리 더 잘 보이게 합니다.
3. **HDBSCAN**: 빽빽하게 모인 곳을 묶음으로 잡고, 어디에도 안 맞는 문장은 "묶이지 않음"(-1)으로 둡니다.
4. **c-TF-IDF**(Kiwi 형태소 기준): 묶음마다 특징적인 단어를 키워드로 뽑습니다.
5. **같은 뜻 묶음 합치기**: 묶음끼리 대표 문장 유사도가 0.78 이상이면 합칩니다(사람이 판정한 148쌍으로 정한 경계). 구체적인 묶음과 일반 묶음은 합치지 않습니다.
6. **소속 확인**: 문장마다 가장 가까운 묶음 중심(묶음 문장 벡터의 평균)과 유사도가 0.75 이상이면 그 묶음, 아니면 "묶이지 않음"입니다.

아래는 고등어 불만 단위로 1~4를 그 자리에서 다시 돌린 결과입니다(1~2분).
""")
code(r"""
from collections import Counter
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

rows = [json.loads(l) for l in (RESULTS / 'units.jsonl').read_text(encoding='utf-8').splitlines()]
neg = Counter(re.sub(r'\s+', ' ', r['unit']).strip() for r in rows if r['label'] == 'NEG')
docs = list(neg)
emb = kure.encode(docs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
keep = {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'}
words = lambda s: [t.form + ('다' if t.tag in ('VV', 'VA') else '') for t in kiwi.tokenize(s) if t.tag in keep]
# embedding_model을 꼭 넘길 것: 안 넘기면 BERTopic 기본 언어가 'english'라서 키워드를 뽑기 전에 한글을 전부 지워 버림
topic_model = BERTopic(embedding_model=kure, umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=20261022),
                       hdbscan_model=HDBSCAN(min_cluster_size=max(15, round(0.01 * len(docs))), prediction_data=True),
                       vectorizer_model=CountVectorizer(tokenizer=words, lowercase=False, min_df=2, token_pattern=None),
                       top_n_words=8)
topics, _ = topic_model.fit_transform(docs, emb)
print(f'불만 문장 {len(docs):,}개(서로 다른 문장) → 묶음 {len(set(topics)) - 1}개, 묶이지 않음 {np.mean(np.array(topics) == -1):.0%}')
info = topic_model.get_topic_info()
info['키워드'] = info['Topic'].map(lambda t: '·'.join(w for w, _ in topic_model.get_topic(t)[:5]) if t != -1 else '(묶이지 않음)')
info['예시'] = info['Representative_Docs'].map(lambda d: ' / '.join(d[:2]))
info[['Topic', 'Count', '키워드', '예시']].head(12)
""")

md("""
실제 분석(`run_product.py`)은 여기에 합치기(5)와 소속 확인(6)까지 한 결과를 저장해 둡니다. 저장된 고등어 불만 묶음:
""")
code(r"""
report = json.loads((RESULTS / 'report.json').read_text(encoding='utf-8'))
saved = pd.DataFrame([{'묶음': g['topic'], '건수': g['units'], '키워드': '·'.join(g['keywords'][:4]),
                       '대표 문장': (g.get('representative') or [''])[0], '합친 묶음 수': len(g.get('merged_from', [g['topic']]))}
                      for g in report['groups']['NEG']['groups'] if g['topic'] != -1])
saved.head(10)
""")

md("""
## ⑥ 새 단위를 기존 묶음에 배정

새 리뷰가 들어올 때마다 묶음 전체를 다시 만들지 않습니다. 새 단위를 **같은 감성**의 기존 묶음과 비교해서, 가장 가까운 묶음의 대표 문장과 유사도가 0.75 이상이면 그 묶음에 넣습니다. 0.75보다 낮으면 "새 의견 후보"로 남깁니다. 이런 후보가 쌓이면 다음 정기 실행 때 새 묶음이 생길 수 있어요.
""")
code(r"""
MEMBER_MIN = 0.75
KEY = {'긍정': 'POS', '부정': 'NEG'}
group_vecs = {}
for label in ['NEG', 'POS']:
    for g in report['groups'][label]['groups']:
        if g['topic'] != -1 and g.get('representative'):
            group_vecs[(label, g['topic'])] = (g, kure.encode(g['representative'], normalize_embeddings=True))
out = []
for u, v, verdict in zip(unit_texts, vec, judged['판정']):
    label = KEY.get(verdict)
    if label is None:
        out.append((u, verdict, '-', None, '감성 평가가 아니라 묶지 않음'))
        continue
    best = max(((g, float((gv @ v).max())) for (l, _), (g, gv) in group_vecs.items() if l == label), key=lambda x: x[1])
    g, sim = best
    name = f"[{'·'.join(g['keywords'][:3])}] {g['representative'][0]}"
    out.append((u, verdict, name, round(sim, 3), '배정' if sim >= MEMBER_MIN else '새 의견 후보'))
assigned = pd.DataFrame(out, columns=['단위', '판정', '가장 가까운 묶음', '유사도', '결과'])
assigned
""")

md("""
## ⑦ 집계

배정된 단위는 그 묶음의 건수에 더해집니다. 보고서는 묶음마다 **최근 3개월**과 **그 이전**의 리뷰 100건당 건수를 비교해서, 최근에 늘어난 불만을 찾습니다. 일반적인 칭찬·불만('맛있어요', '그냥 그래요')은 건수에는 남기고 보고서 앞에는 두지 않습니다.
""")
code(r"""
months, recent = report['period'], report['recent_months']
print(f"기간 {months[0]} ~ {months[1]} | 최근 3개월 {recent} | 최근 리뷰 {report['recent_reviews']:,}건 / 전체 {report['reviews']:,}건")
hit = [a for a in out if a[4] == '배정' and a[1] == '부정']
rows_ = []
for g in report['groups']['NEG']['groups']:
    if g['topic'] == -1:
        continue
    name = f"[{'·'.join(g['keywords'][:3])}] {g['representative'][0]}"
    if any(h[2] == name for h in hit):
        rows_.append({'불만 묶음': name, '지금까지 건수': g['units'], '이번 리뷰로 +': sum(h[2] == name for h in hit),
                      '이전 (100건당)': g['per_100_reviews_earlier'], '최근 (100건당)': g['per_100_reviews_recent']})
pd.DataFrame(rows_)
""")

md("""
## 정리

| 단계 | 도구 | 이 리뷰에서 한 일 |
| --- | --- | --- |
| ① 전처리 | `normalize_text` | `&hellip;` → `…`, 겹친 공백 정리 |
| ② 단위 나누기 | Kiwi | 한 리뷰를 의견 단위 여러 개로 자름 |
| ③ 감성 | KcELECTRA (사후학습 + 미세조정) | 단위마다 긍정 / 부정 / 평가 아님과 확신도 |
| ④ 의미 벡터 | KURE-v1 | 뜻이 비슷한 단위는 유사도가 높음 |
| ⑤ 묶기 | BERTopic (UMAP → HDBSCAN → c-TF-IDF) + 합치기 + 소속 확인 | 상품 전체 단위로 묶음과 키워드를 만듦 |
| ⑥ 배정 | KURE 유사도 ≥ 0.75 | 새 단위를 기존 묶음에 넣거나 새 의견 후보로 남김 |
| ⑦ 집계 | 리뷰 100건당 건수 | 묶음별 건수, 최근 3개월 vs 이전 |

이 노트북은 `7_보고/노트북/make_notebook.py`로 다시 만들 수 있습니다. 0번 셀의 `raw`를 바꾸면 다른 리뷰로도 따라가 볼 수 있습니다.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                                                 'language_info': {'name': 'python'}})
# run from the notebook's own folder, as VS Code does, so the path lookup in the first cell is exercised
ExecutePreprocessor(timeout=1800, kernel_name='python3').preprocess(nb, {'metadata': {'path': str(HERE)}})
nbf.write(nb, NB)
print('written', NB)
