"""Merge opinion groups that say the same thing (HDBSCAN split '크기가 작아요' into four groups for mackerel).

Two groups are compared by the mean KURE-v1 cosine between their texts (central texts + most frequent readable
ones), the same pair-level similarity that was calibrated on the 148 hand-judged pairs (calibrate_merge.py:
>= 0.78 -> 'same' with precision 0.81). Groups are merged by average linkage, so a chain of near neighbours
cannot pull unrelated groups together. Counts, recent/earlier rates and keywords are combined; the largest
member gives the representative sentence. Specific and generic groups are never merged with each other
('가시 없어서 먹기 편해요' must not vanish into '편해서 먹기 좋아요'); pass `kind` to split them.
"""
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

OUT = Path(__file__).resolve().parent
THRESHOLD = json.loads((OUT / 'merge-threshold.json').read_text(encoding='utf-8'))['chosen']['threshold']
TEXTS_PER_GROUP = 6


def group_texts(g):
    texts = list(g.get('representative', []))
    for t in g['top_texts']:
        t = t.rsplit(' (', 1)[0]
        if t not in texts and len(t.replace(' ', '')) >= 4:
            texts.append(t)
    return texts[:TEXTS_PER_GROUP] or [t.rsplit(' (', 1)[0] for t in g['top_texts'][:1]]


def merge(groups, encode, threshold=THRESHOLD, kind=None):
    """groups: report groups of one sentiment (topic -1 is left alone). encode: list[str] -> normalized vectors.
    kind: group -> key; only groups with the same key may merge (e.g. specific vs generic)."""
    rest = [g for g in groups if g['topic'] == -1]
    named = [g for g in groups if g['topic'] != -1]
    if kind is not None:
        merged, log = [], []
        for key in dict.fromkeys(kind(g) for g in named):
            part, part_log = merge([g for g in named if kind(g) == key], encode, threshold)
            merged += part
            log += part_log
        merged.sort(key=lambda g: -g['units'])
        return merged + rest, log
    if len(named) < 2:
        return groups, []
    vecs = [encode(group_texts(g)) for g in named]
    n = len(named)
    sim = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = float((vecs[i] @ vecs[j].T).mean())
    labels = fcluster(linkage(squareform(1 - sim, checks=False), method='average'), t=1 - threshold, criterion='distance')
    merged, log = [], []
    for c in sorted(set(labels)):
        members = sorted((named[i] for i in np.where(labels == c)[0]), key=lambda g: -g['units'])
        if len(members) == 1:
            merged.append(members[0])
            continue
        units = sum(m['units'] for m in members)
        keywords = list(dict.fromkeys(k for m in members for k in m['keywords']))[:8]
        top = members[0]
        merged.append({**top, 'units': units, 'share': round(sum(m['share'] for m in members), 3),
                       'unique_texts': sum(m['unique_texts'] for m in members), 'keywords': keywords,
                       'per_100_reviews_recent': round(sum(m['per_100_reviews_recent'] for m in members), 2),
                       'per_100_reviews_earlier': round(sum(m['per_100_reviews_earlier'] for m in members), 2),
                       'merged_from': [m['topic'] for m in members]})
        log.append({'into': top['topic'], 'members': [(m['topic'], m['units'], (m.get('representative') or [''])[0])
                                                      for m in members]})
    merged.sort(key=lambda g: -g['units'])
    return merged + rest, log
