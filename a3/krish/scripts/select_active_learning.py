#!/usr/bin/env python3
"""
Select high-uncertainty MathFish problems for annotation expansion.

Usage:
  python3 select_active_learning.py \
      --run-dir results/roberta_ft \
      --top-n 150 \
      --output annotations/active_learning_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.utils import PUBLISHER_RELATIONS, normalize_problem_text


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def batch_predict_probs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        all_probs.append(sigmoid_np(logits))
    if not all_probs:
        return np.zeros((0, model.num_labels), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def load_unlabeled_pool(path: str, min_text_len: int, max_text_len: int) -> List[Dict]:
    pool: List[Dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("has_image", False) or item.get("is_duplicate", False):
                continue
            direct = [code for rel, code in item.get("standards", []) if rel in PUBLISHER_RELATIONS]
            if direct:
                continue
            text = normalize_problem_text(item.get("text", ""), item.get("elements", {}))
            if len(text) < min_text_len or len(text) > max_text_len:
                continue
            pool.append(
                {
                    "id": item["id"],
                    "text": text,
                    "source": item.get("source", ""),
                    "metadata": item.get("metadata", {}),
                }
            )
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Uncertainty sampling from MathFish")
    parser.add_argument("--run-dir", required=True, help="Directory from train_roberta_ft.py output")
    parser.add_argument("--data-path", default="mathfish_train.jsonl")
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-text-len", type=int, default=20)
    parser.add_argument("--max-text-len", type=int, default=4000)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument(
        "--output",
        default="annotations/active_learning_candidates.jsonl",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "model"
    label_path = run_dir / "label_index.json"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label index not found: {label_path}")

    with open(label_path) as f:
        labels = json.load(f)["labels"]

    pool = load_unlabeled_pool(args.data_path, args.min_text_len, args.max_text_len)
    if not pool:
        raise RuntimeError("No unlabeled candidate problems found.")

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    texts = [x["text"] for x in pool]
    probs = batch_predict_probs(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    ent = binary_entropy(probs).mean(axis=1)
    margin = np.abs(probs - 0.5).mean(axis=1)
    uncertainty = ent + (0.5 - margin)

    order = np.argsort(uncertainty)[::-1]
    top_idx = order[: args.top_n]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in top_idx:
            p = pool[i]
            row_probs = probs[i]
            pred_idx = np.where(row_probs >= args.threshold)[0].tolist()
            if not pred_idx:
                pred_idx = [int(np.argmax(row_probs))]
            pred_labels = [labels[j] for j in pred_idx]
            row = {
                "problem_id": p["id"],
                "text": p["text"],
                "source": p["source"],
                "metadata": p["metadata"],
                "uncertainty_score": float(uncertainty[i]),
                "mean_entropy": float(ent[i]),
                "mean_margin_to_0.5": float(margin[i]),
                "predicted_labels": pred_labels,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(top_idx)} candidates to {out_path}")


if __name__ == "__main__":
    main()
