"""Targets, leakage-safe splits and masked metrics for AI draft experiments."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SENTIMENTS = ("positive", "neutral", "negative")


def build_targets(records, aspects):
    names = [f"{a}:{s}" for a in aspects for s in SENTIMENTS]
    targets = np.zeros((len(records), len(names)), dtype=np.float32)
    masks = np.ones_like(targets)
    for i, row in enumerate(records):
        for j, aspect in enumerate(aspects):
            opinions = [o for o in row["opinion_units"] if o["aspect"] == aspect]
            current = [o for o in opinions if o.get("temporal_scope") != "past"
                       and o["assessment_status"] == "evaluated"]
            if current:
                for opinion in current:
                    targets[i, j * 3 + SENTIMENTS.index(opinion["sentiment"])] = 1
            elif opinions:
                # Neither deferred-only nor past-only is a reviewed negative target.
                masks[i, j * 3:j * 3 + 3] = 0
    return targets, masks, names


def split_records(records, seed):
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        for field in ("review_id", "review_group_id", "text_hash"):
            key = (field, row[field])
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = np.array([root(i) for i in range(len(records))])
    indices = np.arange(len(records))
    train, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.3,
                                           random_state=seed).split(indices, groups=groups))
    val_pos, test_pos = next(GroupShuffleSplit(n_splits=1, test_size=.5,
                                              random_state=seed + 1).split(
                                                  holdout, groups=groups[holdout]))
    result = {"train": sorted(train.tolist()), "validation": sorted(holdout[val_pos].tolist()),
              "test": sorted(holdout[test_pos].tolist())}
    for field in ("review_id", "review_group_id", "text_hash"):
        sets = [{records[i][field] for i in ids} for ids in result.values()]
        assert all(not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3))
    return result


def score_predictions(targets, masks, probabilities, supported, threshold, label_names):
    truth = np.asarray(targets).astype(bool)
    valid = np.asarray(masks).astype(bool)
    supported = np.asarray(supported).astype(bool)
    pred = (np.asarray(probabilities) >= threshold) & supported[None, :]
    tp = (pred & truth & valid).sum(axis=0)
    fp = (pred & ~truth & valid).sum(axis=0)
    fn = (~pred & truth & valid).sum(axis=0)

    def values(t, f, n):
        precision = float(t / (t + f)) if t + f else 0.
        recall = float(t / (t + n)) if t + n else 0.
        f1 = float(2 * t / (2 * t + f + n)) if 2 * t + f + n else 0.
        return precision, recall, f1

    precision, recall, f1 = values(tp.sum(), fp.sum(), fn.sum())
    per_label = []
    for j, name in enumerate(label_names):
        p, r, f = values(tp[j], fp[j], fn[j])
        support = int((truth[:, j] & valid[:, j]).sum())
        per_label.append({"label": name, "precision": p if support else None,
                          "recall": r if support else None, "f1": f if support else None,
                          "support": support, "predictions": int((pred[:, j] & valid[:, j]).sum()),
                          "false_positives": int(fp[j]), "train_supported": bool(supported[j])})
    present = [x["f1"] for x in per_label if x["support"]]
    negative_indices = [j for j, name in enumerate(label_names) if name.endswith(":negative")]
    nt, nf, nn = (arr[negative_indices].sum() for arr in (tp, fp, fn))
    np_, nr, nf1 = values(nt, nf, nn)
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1,
            "macro_f1_present_labels": float(np.mean(present)) if present else None,
            "negative_precision": np_, "negative_recall": nr, "negative_f1": nf1,
            "unsupported_positive_targets": int((truth & valid & ~supported[None, :]).sum()),
            "masked_cells": int((~valid).sum()), "per_label": per_label}
