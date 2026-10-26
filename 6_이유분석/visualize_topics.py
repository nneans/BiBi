"""Map of one product's opinion groups: does similar meaning land in the same group?

Re-runs the same grouping as run_product.py (KURE-v1 -> UMAP 5D -> HDBSCAN) on the saved units, then projects the
same embeddings to 2D (UMAP, cosine) only for drawing. One dot = one distinct unit text (size ~ how often it was
written). The 8 largest groups get the categorical palette in fixed order; the rest are gray, unclustered lighter.
Usage: python visualize_topics.py <product_id> [run]
  -> results-<run>/<product_id>/topics-map.html (hover = text, dropdown = highlight one group) and topics-map-NEG/POS.png
"""
import sys

PRODUCT = sys.argv[1]
RUN = sys.argv[2] if len(sys.argv) > 2 else 'v2'
sys.modules['tensorflow'] = None  # see run_product.py
from bibi_reviews.cpu import pin_to_efficiency_cores  # noqa: E402
pin_to_efficiency_cores()  # before numba/torch load: this PC's P-cores are unstable (see bibi_reviews/cpu.py)

import json  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from umap import UMAP  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import run_product as R  # noqa: E402
from merge_groups import merge  # noqa: E402
from summarize_all import is_specific  # noqa: E402

RESULTS = OUT / f'results-{RUN}' / PRODUCT
# reference categorical palette (dataviz skill, validated), fixed order; gray for folded groups
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER, UNCLUSTERED, SURFACE, INK, MUTED = '#c3c2b7', '#e1e0d9', '#fcfcfb', '#0b0b0b', '#898781'
TITLE = {'NEG': '불만', 'POS': '칭찬'}
plt.rcParams['font.family'] = 'Malgun Gothic'


def grouped(label, units, embedder):
    counts = Counter(re.sub(r'\s+', ' ', u['unit']).strip() for u in units if u['label'] == label)
    docs, topics, model, size, emb = R.topics_for(counts, embedder)
    xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=R.SEED).fit_transform(emb)
    row_of = {d: i for i, d in enumerate(docs)}
    raw = []
    for t in sorted(set(topics.tolist()) - {-1}):
        idx = np.where(topics == t)[0]
        texts = Counter({docs[i]: counts[docs[i]] for i in idx})
        raw.append({'topic': t, 'idx': idx, 'units': sum(texts.values()), 'share': 0, 'unique_texts': len(idx),
                    'keywords': [w for w, _ in model.get_topic(t)][:8], 'per_100_reviews_recent': 0,
                    'per_100_reviews_earlier': 0, 'representative': R.central_texts(texts, row_of, emb),
                    'top_texts': [f'{x} ({n})' for x, n in texts.most_common(5)]})
    # same merge step as the reports (merge_groups.py), so the map shows what the summary counts
    name = R.read_reviews(['product_name'], PRODUCT)[0]['product_name']
    merged, _ = merge(raw, lambda xs: embedder.encode(xs, normalize_embeddings=True, show_progress_bar=False),
                      kind=lambda g: is_specific(g, name))
    by_topic = {g['topic']: g for g in raw}
    groups = []
    for m in merged:
        if m['topic'] == -1:
            continue
        members = m.get('merged_from', [m['topic']])
        groups.append({'topic': m['topic'], 'units': m['units'], 'keywords': m['keywords'][:3],
                       'rep': m['representative'][0], 'merged': len(members),
                       'idx': np.concatenate([by_topic[t]['idx'] for t in members])})
    groups.sort(key=lambda g: -g['units'])
    for rank, g in enumerate(groups):
        g['color'] = SERIES[rank] if rank < len(SERIES) else OTHER
        g['name'] = (f"{rank + 1}. [{'·'.join(g['keywords'])}] {g['rep'][:24]} ({g['units']:,})"
                     + (f" · {g['merged']}개 합침" if g['merged'] > 1 else ''))
    outliers = np.where(topics == -1)[0]
    return docs, counts, xy, groups, outliers


def size_of(n):
    return 4 + 3 * np.log1p(n)


def html(panels):
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[f'{TITLE[l]} 묶음 지도' for l, *_ in panels],
                        horizontal_spacing=0.04)
    buttons_traces, traces = [], 0
    for col, (label, docs, counts, xy, groups, outliers) in enumerate(panels, start=1):
        for g in [{'name': '묶이지 않음', 'idx': outliers, 'color': UNCLUSTERED}] + groups:
            i = g['idx']
            fig.add_trace(go.Scattergl(
                x=xy[i, 0], y=xy[i, 1], mode='markers', name=f"{TITLE[label]} {g['name']}",
                legendgroup=label, showlegend=g['color'] != OTHER,
                marker=dict(color=g['color'], size=size_of(np.array([counts[docs[k]] for k in i])), opacity=0.8,
                            line=dict(width=0.5, color=SURFACE)),
                text=[f"{docs[k]}<br><span style='color:{MUTED}'>{counts[docs[k]]}회 · {g['name']}</span>" for k in i],
                hovertemplate='%{text}<extra></extra>'), row=1, col=col)
            buttons_traces.append((label, g['name']))
            traces += 1
        for g in groups[:len(SERIES)]:  # selective direct labels: the colored groups only
            c = np.median(xy[g['idx']], axis=0)
            fig.add_annotation(x=c[0], y=c[1], text=f"{groups.index(g) + 1}. {'·'.join(g['keywords'][:2])}",
                               showarrow=False, font=dict(size=11, color=INK), bgcolor='rgba(252,252,251,0.75)',
                               row=1, col=col)
    buttons = [dict(label='전체 보기', method='restyle', args=[{'marker.opacity': [0.8] * traces}])]
    for n, (label, name) in enumerate(buttons_traces):
        if name == '묶이지 않음':
            continue
        buttons.append(dict(label=f'{TITLE[label]} {name}'[:60], method='restyle',
                            args=[{'marker.opacity': [0.9 if k == n else 0.06 for k in range(traces)]}]))
    fig.update_layout(template='plotly_white', paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=820,
                      font=dict(family='Malgun Gothic, sans-serif', color=INK), hoverlabel=dict(bgcolor='white'),
                      title=dict(text=f'{R.read_reviews(["product_name"], PRODUCT)[0]["product_name"][:40]} · 점 = 문장, '
                                      '크기 = 나온 횟수, 색 = 묶음 (큰 8개만 색, 나머지 회색)', font=dict(size=15)),
                      legend=dict(font=dict(size=11), itemsizing='constant'),
                      updatemenus=[dict(buttons=buttons, direction='down', x=0, y=1.08, xanchor='left',
                                        showactive=True, bgcolor='white')])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    path = RESULTS / 'topics-map.html'
    fig.write_html(path, include_plotlyjs='cdn')
    return path


def png(label, docs, counts, xy, groups, outliers):
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(xy[outliers, 0], xy[outliers, 1], s=6, c=UNCLUSTERED, linewidths=0)
    for g in reversed(groups):
        sizes = size_of(np.array([counts[docs[k]] for k in g['idx']])) ** 1.6
        ax.scatter(xy[g['idx'], 0], xy[g['idx'], 1], s=sizes, c=g['color'], alpha=0.75, linewidths=0.3,
                   edgecolors=SURFACE, label=g['name'] if g['color'] != OTHER else None)
    for rank, g in enumerate(groups[:len(SERIES)]):
        c = np.median(xy[g['idx']], axis=0)
        ax.text(c[0], c[1], f"{rank + 1}. {'·'.join(g['keywords'][:2])}", fontsize=10, color=INK, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc=SURFACE, ec='none', alpha=0.8))
    handles, names = ax.get_legend_handles_labels()  # drawn largest-last for z-order; list largest first
    handles, names = handles[::-1], names[::-1]
    handles += [ax.scatter([], [], s=20, c=OTHER), ax.scatter([], [], s=20, c=UNCLUSTERED)]
    names += ['그 밖의 묶음', '묶이지 않음']
    ax.legend(handles, names, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=False)
    ax.set_title(f'{TITLE[label]} 묶음 지도 — 점 = 문장, 크기 = 나온 횟수', fontsize=14, color=INK, loc='left')
    ax.axis('off')
    path = RESULTS / f'topics-map-{label}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    units = [json.loads(l) for l in (RESULTS / 'units.jsonl').read_text(encoding='utf-8').splitlines()]
    embedder = SentenceTransformer(str(R.KURE), device='cuda', local_files_only=True)
    panels = []
    for label in ['NEG', 'POS']:
        docs, counts, xy, groups, outliers = grouped(label, units, embedder)
        panels.append((label, docs, counts, xy, groups, outliers))
        print(label, len(docs), 'texts,', len(groups), 'groups ->', png(label, docs, counts, xy, groups, outliers), flush=True)
    print(html(panels))


if __name__ == '__main__':
    main()
