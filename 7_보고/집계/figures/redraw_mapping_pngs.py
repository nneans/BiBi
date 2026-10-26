"""Rebuild the four presentation PNGs without source/method footnotes."""
from pathlib import Path
import json
import hashlib
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / '02_상품정리/product_name_mapping/2026-10-16'
OUT = ROOT / '06_집계/figures/product-name-mapping-flow'
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12,
                     'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
BLUE, ORANGE = '#008cff', '#f68b32'
summary = json.loads((REPORT / 'summary.json').read_text(encoding='utf-8'))
multi = pd.read_csv(REPORT / 'multiple_name_product_ids.csv', dtype={'product_id': str})
details = pd.read_csv(REPORT / 'all_product_name_counts.csv', dtype={'product_id': str})
saved = []

def style(ax, title):
    ax.set_title(title, loc='left', fontsize=19, weight='bold', pad=24)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.tick_params(axis='both', length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e7e7e7')

def save(fig, filename):
    path = OUT / filename
    fig.savefig(path, dpi=160, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    with Image.open(path) as im:
        im.verify()
    saved.append({'file': str(path.relative_to(ROOT)),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

fig, ax = plt.subplots(figsize=(12, 4.3))
counts = [summary['ids_with_one_name'], summary['ids_with_multiple_names']]
labels = ['1대1 · ID당 상품명 1개', '1대다 · ID당 상품명 2개 이상']
ax.barh(labels, counts, height=.47, color=[BLUE, ORANGE])
ax.invert_yaxis()
for i, n in enumerate(counts):
    ax.text(n+5, i, f'{n:,}개 ({n/summary["product_ids"]*100:.1f}%)', va='center')
ax.set_xlim(0, 370)
ax.set_xlabel('상품 ID 수 (개)', labelpad=15)
style(ax, '1. 상품 ID와 상품명 연결 비율')
fig.tight_layout()
save(fig, '01-mapping-type-share.png')

CAP = 10000
assert len(multi) == 246
values = multi.review_count.to_numpy()
names = multi.name_count.to_numpy()
clipped = values > CAP
xs = np.minimum(values, CAP)
# Small deterministic vertical spread keeps products at identical counts visible.
rng = np.random.default_rng(20261016)
ys = names + rng.uniform(-.11, .11, len(multi))
fig, ax = plt.subplots(figsize=(13, 6.4))
style(ax, '2. 여러 상품명을 가진 상품 246개')
ax.scatter(xs[~clipped], ys[~clipped], s=30, color=BLUE, alpha=.55, linewidths=0)
ax.scatter(xs[clipped], ys[clipped], s=58, marker='>', color=ORANGE, alpha=.85, linewidths=0)
ax.set_xlim(-200, 10500)
ax.set_ylim(1.65, 9.65)
ax.set_yticks(range(2, 10))
ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000],
              ['0', '2,000', '4,000', '6,000', '8,000', '10,000 이상'])
ax.set_xlabel('리뷰 수 (건)', labelpad=18)
ax.set_ylabel('ID당 상품명 수 (개)', labelpad=16)
ax.grid(axis='y', color='#eeeeee')
for pid, label, offset in [
    ('140442363', '비바크부어스트 소시지\n140442363 · 1,049건', (30, -20)),
    ('4320235869', '칵테일새우\n4320235869 · 83,185건', (-225, 38)),
]:
    idx = multi.index[multi.product_id == pid][0]
    ax.scatter([xs[idx]], [ys[idx]], s=85, color=ORANGE,
               marker='>' if clipped[idx] else 'o', zorder=4)
    ax.annotate(label, (xs[idx], ys[idx]), xytext=offset, textcoords='offset points',
                fontsize=11, arrowprops={'arrowstyle': '-', 'color': '#888888'},
                va='center')
fig.tight_layout()
save(fig, '02-multiple-name-products-scatter.png')

dist = multi.name_count.value_counts().reindex(range(2, 10), fill_value=0)
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.barh([f'상품명 {i}개' for i in dist.index], dist.values, color=BLUE, height=.58)
ax.invert_yaxis()
for i, n in enumerate(dist.values):
    ax.text(n+2, i, f'{n}개 ({n/len(multi)*100:.1f}%)', va='center')
ax.set_xlim(0, 165)
ax.set_xlabel('상품 ID 수 (개)', labelpad=16)
style(ax, '3. 1대다 상품 246개의 상품명 개수 분포')
fig.tight_layout()
save(fig, '03-name-count-distribution.png')

selected = details[details.product_id == '4320235869'].sort_values('review_count', ascending=False)
assert selected.review_count.sum() == 83185
assert len(selected) == 7
fig, ax = plt.subplots(figsize=(14, 8.4))
labels = [textwrap.fill(n, width=36) for n in selected.product_name]
ax.barh(labels, selected.review_count, height=.48, color=BLUE)
ax.invert_yaxis()
for i, n in enumerate(selected.review_count):
    pct = n / selected.review_count.sum() * 100
    share = f'{pct:.1f}' if pct >= 1 else f'{pct:.2f}'
    ax.text(n+650, i, f'{n:,} ({share}%)', va='center', fontsize=11)
ax.set_xlim(0, 53000)
ax.set_xticks([0, 15000, 30000, 45000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax.set_xlabel('리뷰 수 (건)', labelpad=18)
style(ax, '4. 칵테일새우 상품명별 리뷰 분포 · ID 4320235869')
fig.tight_layout()
save(fig, '04-selected-product-name-share.png')

verification = {
    'png_revision': 'Remove source/method footnotes; linear clipped scatter x axis',
    'scatter_x_scale': 'linear', 'scatter_review_count_cap': CAP,
    'scatter_product_count': len(multi), 'scatter_above_cap_count': int(clipped.sum()),
    'scatter_all_products_retained': len(xs) == len(multi),
    'scatter_values_equal_minimum_of_actual_and_cap': bool(np.array_equal(xs, np.minimum(values, CAP))),
    'source_method_footnotes_drawn': False,
    'figures': saved,
}
(REPORT / 'png_revision_verification.json').write_text(
    json.dumps(verification, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(verification, indent=2, ensure_ascii=False))
