import csv
import json
import numpy as np


def test_masked_predictions_do_not_count_as_draft_mismatch(tmp_path):
    from bibi_reviews.train import save_predictions
    path = tmp_path / "predictions.jsonl"
    records = [{"review_id": "a", "draft_index": 1, "rating": 5,
                "review_type": "general", "review_text": "not tasted yet, delivery was fast"}]
    save_predictions(path, records, [0], np.array([[.9, .9]]), np.array([[1, 0]]),
                     np.array([[1, 0]]), np.array([True, True]), .5,
                     ["delivery:positive", "taste:positive"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["matches_draft"] is True
    assert saved["predicted"] == ["delivery:positive", "taste:positive"]
    assert saved["evaluated_predicted"] == ["delivery:positive"]
    with path.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as f:
        exported = next(csv.DictReader(f))
    assert exported["matches_draft"] == "True"
