import importlib
import numpy as np


def api():
    try:
        return importlib.import_module("bibi_reviews.experiment_data")
    except ModuleNotFoundError:
        assert False, "experiment_data is not implemented"


def record(rid, group=None, text_hash=None, opinions=None):
    return {"review_id": rid, "review_group_id": group or rid,
            "text_hash": text_hash or rid, "opinion_units": opinions or []}


def opinion(aspect, sentiment, scope="current_or_habitual"):
    return {"aspect": aspect, "sentiment": sentiment, "temporal_scope": scope,
            "assessment_status": "deferred" if sentiment is None else "evaluated"}


def test_deferred_and_past_are_not_negative_supervision():
    module = api()
    rows = [record("a", opinions=[opinion("taste", None),
                                  opinion("delivery", "positive", "past")])]
    targets, masks, names = module.build_targets(rows, ["taste", "delivery", "quality"])
    assert names[0] == "taste:positive"
    assert not masks[0, :6].any()
    assert masks[0, 6:].all()
    assert targets.sum() == 0


def test_current_evaluation_preserved_alongside_deferred_and_past():
    module = api()
    rows = [record("a", opinions=[opinion("taste", None),
                                  opinion("taste", "negative"),
                                  opinion("taste", "positive", "past")])]
    targets, masks, _ = module.build_targets(rows, ["taste"])
    assert targets.tolist() == [[0, 0, 1]]
    assert masks.all()


def test_transitive_group_and_hash_connections_stay_together():
    module = api()
    rows = [record("a", "g1", "h1"), record("b", "g1", "h2"),
            record("c", "g2", "h2")]
    rows += [record(str(i)) for i in range(25)]
    split = module.split_records(rows, 20261015)
    assigned = {i: name for name, indices in split.items() for i in indices}
    assert len(assigned) == len(rows)
    assert assigned[0] == assigned[1] == assigned[2]
    assert split == module.split_records(rows, 20261015)
    assert all(split.values())


def test_unsupported_test_positive_counts_as_false_negative():
    module = api()
    y = np.array([[1, 1]], dtype=np.float32)
    mask = np.ones_like(y)
    metrics = module.score_predictions(y, mask, np.array([[.9, .99]]),
                                       np.array([True, False]), .5, ["a", "b"])
    assert metrics["micro_recall"] == .5
    assert metrics["unsupported_positive_targets"] == 1
    assert metrics["per_label"][1]["recall"] == 0


def test_masked_deferred_cells_do_not_affect_metric():
    module = api()
    metrics = module.score_predictions(np.array([[1, 0]]), np.array([[1, 0]]),
                                       np.array([[.9, .99]]), np.array([True, True]),
                                       .5, ["a", "b"])
    assert metrics["micro_f1"] == 1
    assert metrics["per_label"][1]["f1"] is None
