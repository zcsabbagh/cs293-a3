#!/usr/bin/env python3
"""
RoBERTa embedding-similarity baseline for CCSS prediction.

This is a non-fine-tuned baseline:
  - Embed problem text with RoBERTa
  - Embed CCSS standard descriptions with the same encoder
  - Predict top-k most similar standards

Usage:
  python3 roberta_similarity.py \
      --dataset publisher_full \
      --top-k 3 \
      --out-preds preds/roberta_similarity_k3.jsonl \
      --out-metrics results/roberta_similarity_k3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.utils import (
    build_gold_dict,
    evaluate_all_levels,
    is_standard_in_grade_scope,
    load_assigned_problem_examples,
    load_publisher_examples,
    load_standard_descriptions,
    parse_relations_arg,
    parse_grade_key,
    split_examples,
    write_json,
    write_predictions_jsonl,
)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def embed_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        masked = out * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        pooled = F.normalize(pooled, p=2, dim=1)
        rows.append(pooled.detach().cpu().numpy())
    if not rows:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def topk_predict(
    query_vec: np.ndarray,
    label_vecs: np.ndarray,
    label_ids: Sequence[str],
    k: int,
) -> List[str]:
    sims = np.dot(label_vecs, query_vec)
    idx = np.argsort(sims)[::-1][:k]
    return [label_ids[i] for i in idx]


def load_examples(dataset: str, relations_csv: str):
    relations = parse_relations_arg(relations_csv)
    if dataset == "publisher_full":
        return load_publisher_examples("mathfish_train.jsonl", relations=relations)
    if dataset == "assigned":
        return load_assigned_problem_examples("annotations/problems.json", relations=relations)
    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RoBERTa embedding baseline")
    parser.add_argument("--dataset", choices=["publisher_full", "assigned"], default="publisher_full")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--relations",
        default="Addressing,Alignment",
        help="Comma-separated relation labels to use as gold targets.",
    )
    parser.add_argument("--grade-filter", action="store_true")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--out-preds", default="preds/roberta_similarity.jsonl")
    parser.add_argument("--out-metrics", default="results/roberta_similarity.json")
    args = parser.parse_args()

    examples = load_examples(args.dataset, args.relations)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if len(examples) < 10:
        raise RuntimeError(f"Not enough examples for split/eval: {len(examples)}")

    split = split_examples(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by_source=True,
    )
    eval_examples = split[args.eval_split]
    gold = build_gold_dict(eval_examples)

    std_desc = load_standard_descriptions("standards.jsonl")
    label_ids = sorted(std_desc.keys())
    label_texts = [f"{sid}. {std_desc[sid]}".strip() for sid in label_ids]

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    label_vecs = embed_texts(
        label_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    problem_texts = [ex.text for ex in eval_examples]
    problem_vecs = embed_texts(
        problem_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    preds: Dict[str, List[str]] = {}
    label_id_to_idx = {sid: i for i, sid in enumerate(label_ids)}
    for ex, vec in zip(eval_examples, problem_vecs):
        if args.grade_filter:
            grade_key = parse_grade_key(ex.metadata)
            candidate_ids = [sid for sid in label_ids if is_standard_in_grade_scope(sid, grade_key)]
            if not candidate_ids:
                candidate_ids = label_ids
            candidate_idx = [label_id_to_idx[sid] for sid in candidate_ids]
            c_vecs = label_vecs[candidate_idx]
            pred = topk_predict(vec, c_vecs, candidate_ids, args.top_k)
        else:
            pred = topk_predict(vec, label_vecs, label_ids, args.top_k)
        preds[ex.problem_id] = pred

    pred_sets = {pid: set(labels) for pid, labels in preds.items()}
    metrics = evaluate_all_levels(pred_sets, gold)

    metrics_out = {
        "config": vars(args),
        "dataset_size": len(examples),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "metrics": metrics,
    }
    write_predictions_jsonl(args.out_preds, preds)
    write_json(args.out_metrics, metrics_out)

    print(json.dumps(metrics_out, indent=2))
    print(f"\nPredictions: {args.out_preds}")
    print(f"Metrics:     {args.out_metrics}")


if __name__ == "__main__":
    main()
