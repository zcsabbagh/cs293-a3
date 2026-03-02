#!/usr/bin/env python3
"""
Fine-tune RoBERTa for multi-label CCSS prediction.

Usage:
  python3 train_roberta_ft.py \
      --dataset publisher_full \
      --output-dir results/roberta_ft_publisher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.utils import (
    Example,
    build_label_space,
    evaluate_all_levels,
    is_standard_in_grade_scope,
    load_assigned_problem_examples,
    load_publisher_examples,
    parse_relations_arg,
    parse_grade_key,
    split_examples,
    write_json,
    write_predictions_jsonl,
)


def load_examples(dataset: str, relations_csv: str) -> List[Example]:
    relations = parse_relations_arg(relations_csv)
    if dataset == "publisher_full":
        return load_publisher_examples("mathfish_train.jsonl", relations=relations)
    if dataset == "assigned":
        return load_assigned_problem_examples("annotations/problems.json", relations=relations)
    raise ValueError(f"Unknown dataset: {dataset}")


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vectorize_labels(labels: Sequence[str], label_to_idx: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(label_to_idx), dtype=np.float32)
    for sid in labels:
        idx = label_to_idx.get(sid)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def to_hf_dataset(examples: Sequence[Example], label_to_idx: Dict[str, int]) -> Dataset:
    rows = {
        "problem_id": [],
        "text": [],
        "labels": [],
    }
    for ex in examples:
        vec = vectorize_labels(ex.labels, label_to_idx)
        if vec.sum() == 0:
            continue
        rows["problem_id"].append(ex.problem_id)
        rows["text"].append(ex.text)
        rows["labels"].append(vec.tolist())
    return Dataset.from_dict(rows)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predictions_from_probs(
    probs: np.ndarray,
    example_ids: Sequence[str],
    label_list: Sequence[str],
    threshold: float,
    top_k_fallback: int = 1,
    allowed_labels_by_id: Dict[str, set] = None,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    label_to_idx = {sid: i for i, sid in enumerate(label_list)}
    for i, pid in enumerate(example_ids):
        p = probs[i]
        allowed_idx = None
        if allowed_labels_by_id and pid in allowed_labels_by_id:
            allowed = allowed_labels_by_id[pid]
            allowed_idx = [label_to_idx[sid] for sid in allowed if sid in label_to_idx]
            if not allowed_idx:
                allowed_idx = None

        if allowed_idx is None:
            idx = np.where(p >= threshold)[0].tolist()
        else:
            idx = [j for j in allowed_idx if p[j] >= threshold]
        if not idx and top_k_fallback > 0:
            if allowed_idx is None:
                idx = np.argsort(p)[::-1][:top_k_fallback].tolist()
            else:
                allowed_sorted = sorted(allowed_idx, key=lambda j: p[j], reverse=True)
                idx = allowed_sorted[:top_k_fallback]
        labels = [label_list[j] for j in idx]
        out[pid] = labels
    return out


def predictions_from_topk(
    probs: np.ndarray,
    example_ids: Sequence[str],
    label_list: Sequence[str],
    k: int,
    allowed_labels_by_id: Dict[str, set] = None,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    label_to_idx = {sid: i for i, sid in enumerate(label_list)}
    for i, pid in enumerate(example_ids):
        p = probs[i]
        allowed_idx = None
        if allowed_labels_by_id and pid in allowed_labels_by_id:
            allowed = allowed_labels_by_id[pid]
            allowed_idx = [label_to_idx[sid] for sid in allowed if sid in label_to_idx]
            if not allowed_idx:
                allowed_idx = None
        if allowed_idx is None:
            idx = np.argsort(p)[::-1][:k].tolist()
        else:
            idx = sorted(allowed_idx, key=lambda j: p[j], reverse=True)[:k]
        out[pid] = [label_list[j] for j in idx]
    return out


def search_best_threshold(
    probs: np.ndarray,
    val_ids: Sequence[str],
    gold: Dict[str, set],
    label_list: Sequence[str],
    allowed_labels_by_id: Dict[str, set] = None,
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.05,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    best_thr = 0.5
    best_metrics: Dict[str, Dict[str, float]] = {}
    best_f1 = -1.0
    thr = low
    while thr <= high + 1e-8:
        preds = predictions_from_probs(
            probs=probs,
            example_ids=val_ids,
            label_list=label_list,
            threshold=round(thr, 6),
            top_k_fallback=1,
            allowed_labels_by_id=allowed_labels_by_id,
        )
        pred_sets = {pid: set(labels) for pid, labels in preds.items()}
        metrics = evaluate_all_levels(pred_sets, gold)
        f1 = metrics["standard"]["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = round(thr, 6)
            best_metrics = metrics
        thr += step
    return best_thr, best_metrics


def search_best_decode(
    probs: np.ndarray,
    val_ids: Sequence[str],
    gold: Dict[str, set],
    label_list: Sequence[str],
    allowed_labels_by_id: Dict[str, set] = None,
    threshold_low: float = 0.05,
    threshold_high: float = 0.95,
    threshold_step: float = 0.05,
    max_top_k: int = 5,
) -> Dict:
    best = {
        "mode": "threshold",
        "param": 0.5,
        "metrics": {},
        "f1": -1.0,
    }

    thr, metrics = search_best_threshold(
        probs=probs,
        val_ids=val_ids,
        gold=gold,
        label_list=label_list,
        allowed_labels_by_id=allowed_labels_by_id,
        low=threshold_low,
        high=threshold_high,
        step=threshold_step,
    )
    f1 = metrics["standard"]["f1"]
    if f1 > best["f1"]:
        best = {"mode": "threshold", "param": thr, "metrics": metrics, "f1": f1}

    for k in range(1, max_top_k + 1):
        preds = predictions_from_topk(
            probs=probs,
            example_ids=val_ids,
            label_list=label_list,
            k=k,
            allowed_labels_by_id=allowed_labels_by_id,
        )
        pred_sets = {pid: set(labels) for pid, labels in preds.items()}
        m = evaluate_all_levels(pred_sets, gold)
        f1_k = m["standard"]["f1"]
        if f1_k > best["f1"]:
            best = {"mode": "topk", "param": k, "metrics": m, "f1": f1_k}

    return best


def compute_micro_f1_at_threshold(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    probs = sigmoid_np(logits)
    preds = (probs >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def domain_of(code: str) -> str:
    parts = code.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else code


def classify_error(predicted: set, gold: set) -> str:
    fp = predicted - gold
    fn = gold - predicted
    if fp and not fn:
        return "false_positive_only"
    if fn and not fp:
        return "false_negative_only"
    if fp and fn:
        fp_domains = {domain_of(x) for x in fp}
        fn_domains = {domain_of(x) for x in fn}
        if fp_domains.isdisjoint(fn_domains):
            return "cross_domain_substitution"
        return "within_domain_substitution"
    return "correct"


def build_gold_filtered(examples: Sequence[Example], label_space: set) -> Dict[str, set]:
    gold: Dict[str, set] = {}
    for ex in examples:
        kept = sorted([sid for sid in ex.labels if sid in label_space])
        if kept:
            gold[ex.problem_id] = set(kept)
    return gold


def build_allowed_labels_by_id(examples: Sequence[Example], label_list: Sequence[str]) -> Dict[str, set]:
    all_labels = set(label_list)
    out: Dict[str, set] = {}
    for ex in examples:
        grade_key = parse_grade_key(ex.metadata)
        allowed = {sid for sid in all_labels if is_standard_in_grade_scope(sid, grade_key)}
        out[ex.problem_id] = allowed if allowed else all_labels
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on CCSS labels")
    parser.add_argument("--dataset", choices=["publisher_full", "assigned"], default="publisher_full")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--output-dir", default="results/roberta_ft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-label-frequency", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--relations",
        default="Addressing,Alignment",
        help="Comma-separated relation labels to use as gold targets.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save final fine-tuned model weights/tokenizer under <output-dir>/model.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep per-epoch Trainer checkpoints (can consume several GB).",
    )
    parser.add_argument("--threshold-low", type=float, default=0.05)
    parser.add_argument("--threshold-high", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--max-top-k",
        type=int,
        default=5,
        help="Also evaluate top-k decoding (1..k) on validation and choose best strategy.",
    )
    parser.add_argument(
        "--disable-grade-filter",
        action="store_true",
        help="Disable grade-scoped decoding during threshold tuning/evaluation.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.dataset, args.relations)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if len(examples) < 20:
        raise RuntimeError(f"Not enough examples to train/eval: {len(examples)}")

    split = split_examples(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by_source=True,
    )
    label_list = build_label_space(split["train"], min_freq=args.min_label_frequency)
    if not label_list:
        raise RuntimeError("No labels left after min-label-frequency filtering.")
    label_to_idx = {sid: i for i, sid in enumerate(label_list)}

    train_ds = to_hf_dataset(split["train"], label_to_idx)
    val_ds = to_hf_dataset(split["val"], label_to_idx)
    test_ds = to_hf_dataset(split["test"], label_to_idx)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Empty split after label filtering: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)
    test_tok = test_ds.map(tokenize_fn, batched=True)

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        problem_type="multi_label_classification",
    )

    device_choice = pick_device(args.device)
    use_fp16 = device_choice == "cuda"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        precision, recall, f1 = compute_micro_f1_at_threshold(logits, labels, threshold=0.5)
        return {"micro_precision_0_5": precision, "micro_recall_0_5": recall, "micro_f1_0_5": f1}

    save_strategy = "epoch" if args.keep_checkpoints else "no"
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=str(out_dir / "logs"),
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        load_best_model_at_end=args.keep_checkpoints,
        metric_for_best_model="micro_f1_0_5" if args.keep_checkpoints else None,
        greater_is_better=True if args.keep_checkpoints else None,
        seed=args.seed,
        report_to=[],
        fp16=use_fp16,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_cpu=(device_choice == "cpu"),
        dataloader_pin_memory=(device_choice == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    if args.save_model:
        trainer.save_model(str(out_dir / "model"))
        tokenizer.save_pretrained(str(out_dir / "model"))

    # Threshold tuning on validation set.
    val_pred = trainer.predict(val_tok)
    val_probs = sigmoid_np(val_pred.predictions)
    val_ids = val_ds["problem_id"]
    val_gold = build_gold_filtered(
        [ex for ex in split["val"] if ex.problem_id in set(val_ids)],
        label_space=set(label_list),
    )
    val_allowed = (
        None
        if args.disable_grade_filter
        else build_allowed_labels_by_id(
            [ex for ex in split["val"] if ex.problem_id in set(val_ids)],
            label_list=label_list,
        )
    )
    best_decode = search_best_decode(
        probs=val_probs,
        val_ids=val_ids,
        gold=val_gold,
        label_list=label_list,
        allowed_labels_by_id=val_allowed,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        threshold_step=args.threshold_step,
        max_top_k=args.max_top_k,
    )
    best_mode = best_decode["mode"]
    best_param = best_decode["param"]
    best_val_metrics = best_decode["metrics"]

    # Evaluate on test with tuned threshold.
    test_pred = trainer.predict(test_tok)
    test_probs = sigmoid_np(test_pred.predictions)
    test_ids = test_ds["problem_id"]
    test_allowed = (
        None
        if args.disable_grade_filter
        else build_allowed_labels_by_id(
            [ex for ex in split["test"] if ex.problem_id in set(test_ids)],
            label_list=label_list,
        )
    )
    if best_mode == "threshold":
        test_preds = predictions_from_probs(
            probs=test_probs,
            example_ids=test_ids,
            label_list=label_list,
            threshold=float(best_param),
            top_k_fallback=1,
            allowed_labels_by_id=test_allowed,
        )
    else:
        test_preds = predictions_from_topk(
            probs=test_probs,
            example_ids=test_ids,
            label_list=label_list,
            k=int(best_param),
            allowed_labels_by_id=test_allowed,
        )
    test_gold = build_gold_filtered(
        [ex for ex in split["test"] if ex.problem_id in set(test_ids)],
        label_space=set(label_list),
    )
    test_metrics = evaluate_all_levels({k: set(v) for k, v in test_preds.items()}, test_gold)

    # Save val predictions with tuned threshold for reporting.
    if best_mode == "threshold":
        val_preds = predictions_from_probs(
            probs=val_probs,
            example_ids=val_ids,
            label_list=label_list,
            threshold=float(best_param),
            top_k_fallback=1,
            allowed_labels_by_id=val_allowed,
        )
    else:
        val_preds = predictions_from_topk(
            probs=val_probs,
            example_ids=val_ids,
            label_list=label_list,
            k=int(best_param),
            allowed_labels_by_id=val_allowed,
        )

    # Error examples for report writing.
    problem_by_id = {ex.problem_id: ex for ex in split["test"]}
    error_rows = []
    for pid in test_ids:
        gold = test_gold.get(pid, set())
        pred = set(test_preds.get(pid, []))
        if pred == gold:
            continue
        row = {
            "problem_id": pid,
            "text": problem_by_id.get(pid).text if pid in problem_by_id else "",
            "predicted": sorted(pred),
            "gold": sorted(gold),
            "false_positives": sorted(pred - gold),
            "false_negatives": sorted(gold - pred),
            "error_type": classify_error(pred, gold),
        }
        error_rows.append(row)

    # Save artifacts.
    write_predictions_jsonl(out_dir / "preds_val.jsonl", val_preds)
    write_predictions_jsonl(out_dir / "preds_test.jsonl", test_preds)
    with open(out_dir / "error_examples_test.jsonl", "w") as f:
        for row in error_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_payload = {
        "config": vars(args),
        "label_count": len(label_list),
        "split_sizes_raw": {k: len(v) for k, v in split.items()},
        "split_sizes_used": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "best_decode_mode": best_mode,
        "best_decode_param": best_param,
        "val_metrics_best_decode": best_val_metrics,
        "test_metrics_best_decode": test_metrics,
        "num_error_examples_test": len(error_rows),
    }
    write_json(out_dir / "metrics.json", metrics_payload)
    write_json(out_dir / "label_index.json", {"labels": label_list})
    write_json(
        out_dir / "split_ids.json",
        {
            "train": list(train_ds["problem_id"]),
            "val": list(val_ds["problem_id"]),
            "test": list(test_ds["problem_id"]),
        },
    )

    print(json.dumps(metrics_payload, indent=2))
    if args.save_model:
        print(f"\nSaved model to:      {out_dir / 'model'}")
    print(f"Saved metrics to:    {out_dir / 'metrics.json'}")
    print(f"Saved test preds to: {out_dir / 'preds_test.jsonl'}")


if __name__ == "__main__":
    main()
