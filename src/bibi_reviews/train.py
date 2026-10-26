"""Run the two-encoder, review-level aspect/sentiment comparison locally.

No evidence extraction is claimed by this first classification experiment.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
import transformers

from .experiment_data import build_targets, split_records, score_predictions


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class AspectClassifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(encoder.config.hidden_size, num_labels))

    def forward(self, features):
        return self.head(self.encoder(**features).last_hidden_state[:, 0])

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory, safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(directory / "aspect_head.safetensors"))

    @classmethod
    def load(cls, directory, num_labels, device):
        model = cls(AutoModel.from_pretrained(directory, local_files_only=True,
                                             use_safetensors=True), num_labels)
        model.head.load_state_dict(load_file(str(directory / "aspect_head.safetensors")))
        return model.to(device)


def make_loader(tokenizer, encodings, indices, batch_size, shuffle, seed):
    data = [{k: v[i] for k, v in encodings.items()} | {"row_index": i} for i in indices]
    pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def collate(rows):
        ids = torch.tensor([r["row_index"] for r in rows], dtype=torch.long)
        features = pad([{k: v for k, v in r.items() if k != "row_index"} for r in rows])
        return features, ids

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def predict(model, loader, device):
    model.eval()
    probabilities, indices = [], []
    with torch.inference_mode():
        for features, ids in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model({k: v.to(device) for k, v in features.items()})
            probabilities.append(logits.float().sigmoid().cpu().numpy())
            indices.extend(ids.tolist())
    return np.concatenate(probabilities), indices


def choose_threshold(targets, masks, probabilities, supported, names, grid):
    scored = [(score_predictions(targets, masks, probabilities, supported, t, names)["micro_f1"], t)
              for t in grid]
    # A tie favors the higher threshold, reducing otherwise tied extra predictions.
    return max(scored)[1]


def validate_records(records, aspects):
    seen = set()
    for record in records:
        if record["review_id"] in seen:
            raise ValueError("Duplicate review_id")
        seen.add(record["review_id"])
        if record["product_id"] != "7826600725" or record["display_status"] != "정상":
            raise ValueError("Unexpected product or display scope")
        if record["annotation_origin"] != "codex_ai_draft" or record["human_approved"]:
            raise ValueError("This experiment expects explicitly unapproved AI drafts")
        if hashlib.sha256(record["review_text"].encode()).hexdigest() != record["text_hash"]:
            raise ValueError("Text hash mismatch")
        for opinion in record["opinion_units"]:
            if opinion["aspect"] not in aspects:
                raise ValueError("Unknown aspect")
            if record["review_text"][opinion["start"]:opinion["end"]] != opinion["evidence"]:
                raise ValueError("Evidence offset mismatch")


def save_predictions(path, records, ids, probabilities, targets, masks, supported, threshold, names):
    rows = []
    for row_index, probs in zip(ids, probabilities):
        record = records[row_index]
        expected = [names[j] for j in range(len(names)) if targets[row_index, j] and masks[row_index, j]]
        predicted = [names[j] for j in range(len(names)) if probs[j] >= threshold and supported[j]]
        evaluated_predicted = [names[j] for j in range(len(names))
                               if probs[j] >= threshold and supported[j] and masks[row_index, j]]
        rows.append({"review_id": record["review_id"], "draft_index": record["draft_index"],
                     "rating": record["rating"], "review_type": record["review_type"],
                     "review_text": record["review_text"], "expected_ai_draft": expected,
                     "predicted": predicted, "evaluated_predicted": evaluated_predicted,
                     "matches_draft": expected == evaluated_predicted,
                     "masked_labels": [names[j] for j in range(len(names)) if not masks[row_index, j]],
                     "probabilities": [float(p) for p in probs], "human_approved": False})
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[k for k in rows[0] if k != "probabilities"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: " | ".join(v) if isinstance(v, list) else v
                             for k, v in row.items() if k != "probabilities"})


def train_model(spec, config, records, targets, masks, names, split, report_dir, artifact_dir, device):
    seed_all(config["seed"])
    print(f"LOADING {spec['model_id']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["tokenizer_revision"])
    encodings = tokenizer([r["review_text"] for r in records], truncation=False, padding=False)
    lengths = [len(x) for x in encodings["input_ids"]]
    if max(lengths) > config["max_length"]:
        raise ValueError(f"Long review needs windowing: {max(lengths)} > {config['max_length']}")
    encoder = AutoModel.from_pretrained(spec["model_id"], revision=spec["revision"], use_safetensors=True)
    model = AspectClassifier(encoder, len(names)).to(device)
    loaders = {name: make_loader(tokenizer, encodings, ids,
                                config["micro_batch_size"] if name == "train" else config["evaluation_batch_size"],
                                name == "train", config["seed"])
               for name, ids in split.items()}
    train_ids = split["train"]
    positives = (targets[train_ids] * masks[train_ids]).sum(axis=0)
    supported = positives > 0
    negatives = ((1 - targets[train_ids]) * masks[train_ids]).sum(axis=0)
    weights = np.clip(negatives / np.maximum(positives, 1), 1, config["positive_weight_cap"])
    target_tensor = torch.tensor(targets, device=device)
    mask_tensor = torch.tensor(masks * supported[None, :], device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    accumulation = config["gradient_accumulation"]
    updates_per_epoch = math.ceil(len(loaders["train"]) / accumulation)
    total_steps = config["epochs"] * updates_per_epoch
    warmup = max(1, int(total_steps * .1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0, (total_steps - step) / max(1, total_steps - warmup))))
    report_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(artifact_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    history, best_score, stale, best_epoch, threshold = [], -1., 0, None, .5
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_valid = 0., 0.
        for batch, (features, ids) in enumerate(loaders["train"]):
            ids = ids.to(device)
            valid = mask_tensor[ids]
            # Normalize each accumulation group by its true number of supervised cells.
            # Its row ids are unknown until loaded, so accumulate loss sums and normalize gradients below.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model({k: v.to(device) for k, v in features.items()})
                loss_sum = (criterion(logits.float(), target_tensor[ids]) * valid).sum()
            loss_sum.backward()
            total_loss += loss_sum.item()
            total_valid += valid.sum().item()
            if batch % accumulation == 0:
                group_valid = 0.
            group_valid += valid.sum().item()
            if (batch + 1) % accumulation == 0 or batch + 1 == len(loaders["train"]):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(max(group_valid, 1))
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        probs, ids = predict(model, loaders["validation"], device)
        chosen = choose_threshold(targets[ids], masks[ids], probs, supported, names, config["threshold_grid"])
        metrics = score_predictions(targets[ids], masks[ids], probs, supported, chosen, names)
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_valid, 1),
                        "threshold": chosen, "validation": metrics})
        print(f"EPOCH {spec['name']} {epoch}: loss={history[-1]['train_loss']:.4f} "
              f"val_micro_f1={metrics['micro_f1']:.4f} threshold={chosen}", flush=True)
        if metrics["micro_f1"] > best_score:
            best_score, threshold, best_epoch, stale = metrics["micro_f1"], chosen, epoch, 0
            model.save(artifact_dir)
        else:
            stale += 1
        write_json(report_dir / "history.json", history)
        if stale >= config["patience"]:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None
    del optimizer, scheduler, criterion, model, encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AspectClassifier.load(artifact_dir, len(names), device)
    outputs = {}
    for name, loader in loaders.items():
        # Use ordered train loader too, by saving each returned row index explicitly.
        probs, ids = predict(model, loader, device)
        outputs[name] = score_predictions(targets[ids], masks[ids], probs, supported, threshold, names)
        if name in ("validation", "test"):
            save_predictions(report_dir / f"{name}_predictions.jsonl", records, ids, probs,
                             targets, masks, supported, threshold, names)
    result = {"model": spec, "best_epoch": best_epoch, "threshold": threshold,
              "threshold_selected_on": "validation_only", "supported": supported.tolist(),
              "train_positive_counts": positives.astype(int).tolist(), "label_names": names,
              "training_seconds": elapsed, "peak_gpu_mib": peak, "max_input_tokens": max(lengths),
              "truncated_reviews": 0, "metrics": outputs, "artifact_directory": str(artifact_dir),
              "label_source": "unapproved_codex_ai_draft", "human_gold_accuracy_measured": False,
              "evidence_extraction_implemented": False}
    write_json(report_dir / "result.json", result)
    write_json(artifact_dir / "experiment_metadata.json", result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"DONE {spec['name']}: test_micro_f1={outputs['test']['micro_f1']:.4f}", flush=True)
    return result


def verify_model(model_spec, config, records, targets, masks, names, report_dir, artifact_dir, device):
    saved = json.loads((report_dir / "result.json").read_text(encoding="utf-8"))
    entries = [json.loads(line) for line in (report_dir / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["review_id"]: i for i, r in enumerate(records)}
    ids = [by_id[r["review_id"]] for r in entries]
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)
    encodings = tokenizer([r["review_text"] for r in records], truncation=False, padding=False)
    loader = make_loader(tokenizer, encodings, ids, config["evaluation_batch_size"], False, config["seed"])
    model = AspectClassifier.load(artifact_dir, len(names), device)
    probs, actual_ids = predict(model, loader, device)
    expected = np.array([r["probabilities"] for r in entries])
    difference = float(np.max(np.abs(probs - expected)))
    metrics = score_predictions(targets[actual_ids], masks[actual_ids], probs,
                                np.array(saved["supported"]), saved["threshold"], names)
    if difference > 1e-5 or metrics != saved["metrics"]["test"]:
        raise AssertionError(f"Reload verification failed: {difference}")
    result = {"model": model_spec["name"], "fresh_model_loaded": True, "test_rows": len(ids),
              "maximum_probability_difference": difference, "test_metrics_identical": True}
    write_json(report_dir / "reload_verification.json", result)
    print(f"VERIFIED {model_spec['name']}: difference={difference}", flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/focus_bert_comparison.json")
    parser.add_argument("--output", default="보관함/2026-10-15-focus-bert-comparison/reports")
    parser.add_argument("--artifacts", default="보관함/2026-10-15-focus-bert-comparison/checkpoints")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = Path(config["dataset"])
    vocab = json.loads(Path(config["vocabulary"]).read_text(encoding="utf-8"))
    aspects = list(vocab["aspect_names"])
    all_records = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()]
    validate_records(all_records, aspects)
    records = [r for r in all_records if r["training_eligibility"] != "exclude_pending_review"]
    targets, masks, names = build_targets(records, aspects)
    split = split_records(records, config["seed"])
    output, artifacts = Path(args.output), Path(args.artifacts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.verify_only:
        if output.exists() or artifacts.exists():
            raise FileExistsError("Choose a new run output and artifacts directory; existing runs are never overwritten")
        manifest = {"config": config, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "vocabulary_sha256": hashlib.sha256(Path(config['vocabulary']).read_bytes()).hexdigest(),
                    "source_reviews": len(all_records), "eligible_reviews": len(records),
                    "excluded_review_ids": [r["review_id"] for r in all_records if r not in records],
                    "split": {key: [records[i]["review_id"] for i in ids] for key, ids in split.items()},
                    "split_counts": {key: len(ids) for key, ids in split.items()},
                    "overlap_review_group_and_text_hash": 0, "label_names": names,
                    "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                "transformers": transformers.__version__, "numpy": np.__version__,
                                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}}
        write_json(output / "manifest.json", manifest)
        print(f"SPLIT {manifest['split_counts']}; masked cells={int((masks == 0).sum())}", flush=True)
        results = []
        for spec in config["models"]:
            result = train_model(spec, config, records, targets, masks, names, split,
                                 output / spec["name"], artifacts / spec["name"], device)
            results.append(result)
            write_json(output / "comparison.json", {"results": results,
                       "selected_by_validation": max(results, key=lambda r: r["metrics"]["validation"]["micro_f1"])["model"]["name"],
                       "scope": "One-seed agreement with unapproved AI drafts; no human-gold accuracy measured"})
    else:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config or manifest["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Configuration or dataset changed since training")
        if manifest["vocabulary_sha256"] != hashlib.sha256(Path(config['vocabulary']).read_bytes()).hexdigest():
            raise ValueError("Vocabulary changed since training")
        for spec in config["models"]:
            verify_model(spec, config, records, targets, masks, names,
                         output / spec["name"], artifacts / spec["name"], device)


if __name__ == "__main__":
    main()
