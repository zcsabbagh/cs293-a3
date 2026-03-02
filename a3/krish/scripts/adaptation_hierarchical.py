#!/usr/bin/env python3
"""
Adaptation 1: Hierarchical prediction (domain -> cluster -> standard).

This script builds a hierarchical predictor using RoBERTa embeddings over
standards descriptions, with stage-wise decoding tuned on validation data.

Usage:
  python3 a3/krish/scripts/adaptation_hierarchical.py \
    --output-dir a3/krish/results/adaptations/hierarchical \
    --epochs-note "description-similarity adaptation"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.utils import (
    Example,
    build_gold_dict,
    evaluate,
    load_publisher_examples,
    load_standards,
    map_level,
    parse_grade_key,
    split_examples,
    standard_levels,
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
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=1)
        rows.append(pooled.detach().cpu().numpy())
    if not rows:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def select_topk_with_scores(ids: Sequence[str], scores: np.ndarray, k: int) -> List[str]:
    if len(ids) == 0:
        return []
    k = max(1, min(k, len(ids)))
    idx = np.argsort(scores)[::-1][:k]
    return [ids[i] for i in idx]


def build_hierarchy(entries: Dict[str, Dict]) -> Dict:
    domain_to_clusters: Dict[str, List[str]] = defaultdict(list)
    cluster_to_standards: Dict[str, List[str]] = defaultdict(list)
    domain_desc: Dict[str, str] = {}
    cluster_desc: Dict[str, str] = {}
    standard_desc: Dict[str, str] = {}
    standard_to_cluster: Dict[str, str] = {}
    cluster_to_domain: Dict[str, str] = {}

    for sid, item in entries.items():
        lvl = item.get("level")
        if lvl == "Domain":
            domain_desc[sid] = item.get("description", "")
        elif lvl == "Cluster":
            cluster_desc[sid] = item.get("description", "")
            parent = item.get("parent", "")
            if parent:
                cluster_to_domain[sid] = parent
                domain_to_clusters[parent].append(sid)
        elif lvl in {"Standard", "Sub-standard"}:
            standard_desc[sid] = item.get("description", "")
            parent = item.get("parent", "")
            if parent:
                standard_to_cluster[sid] = parent
                cluster_to_standards[parent].append(sid)

    for d in domain_to_clusters:
        domain_to_clusters[d] = sorted(set(domain_to_clusters[d]))
    for c in cluster_to_standards:
        cluster_to_standards[c] = sorted(set(cluster_to_standards[c]))

    return {
        "domain_to_clusters": domain_to_clusters,
        "cluster_to_standards": cluster_to_standards,
        "domain_desc": domain_desc,
        "cluster_desc": cluster_desc,
        "standard_desc": standard_desc,
        "standard_to_cluster": standard_to_cluster,
        "cluster_to_domain": cluster_to_domain,
    }


def in_grade_scope(code: str, grade_key):
    if grade_key is None:
        return True
    if isinstance(grade_key, list):
        return any(code.startswith(f"{p}-") for p in grade_key)
    if grade_key == "K":
        return code.startswith("K.")
    return code.startswith(f"{grade_key}.")


def predict_hierarchical(
    examples: Sequence[Example],
    hierarchy: Dict,
    embeddings: Dict[str, Dict[str, np.ndarray]],
    ids: Dict[str, List[str]],
    k_domain: int,
    k_cluster: int,
    k_standard: int,
    grade_filter: bool = True,
) -> Dict[str, Dict[str, List[str]]]:
    domain_ids = ids["domain"]
    cluster_ids = ids["cluster"]
    standard_ids = ids["standard"]

    domain_mat = embeddings["domain"]["matrix"]
    cluster_mat = embeddings["cluster"]["matrix"]
    standard_mat = embeddings["standard"]["matrix"]
    cluster_to_idx = embeddings["cluster"]["id_to_idx"]
    standard_to_idx = embeddings["standard"]["id_to_idx"]

    out: Dict[str, Dict[str, List[str]]] = {}
    for ex, q in zip(examples, embeddings["query"]["matrix"]):
        grade_key = parse_grade_key(ex.metadata) if grade_filter else None

        domain_candidates = [d for d in domain_ids if in_grade_scope(d, grade_key)]
        if not domain_candidates:
            domain_candidates = domain_ids
        d_idx = [embeddings["domain"]["id_to_idx"][d] for d in domain_candidates]
        d_scores = np.dot(domain_mat[d_idx], q)
        pred_domains = select_topk_with_scores(domain_candidates, d_scores, k_domain)

        cluster_candidates: List[str] = []
        for d in pred_domains:
            cluster_candidates.extend(hierarchy["domain_to_clusters"].get(d, []))
        cluster_candidates = sorted(set(cluster_candidates))
        if grade_filter:
            cluster_candidates = [c for c in cluster_candidates if in_grade_scope(c, grade_key)]
        if not cluster_candidates:
            cluster_candidates = [c for c in cluster_ids if in_grade_scope(c, grade_key)] or cluster_ids

        c_idx = [cluster_to_idx[c] for c in cluster_candidates]
        c_scores = np.dot(cluster_mat[c_idx], q)
        pred_clusters = select_topk_with_scores(cluster_candidates, c_scores, k_cluster)

        standard_candidates: List[str] = []
        for c in pred_clusters:
            standard_candidates.extend(hierarchy["cluster_to_standards"].get(c, []))
        standard_candidates = sorted(set(standard_candidates))
        if grade_filter:
            standard_candidates = [s for s in standard_candidates if in_grade_scope(s, grade_key)]
        if not standard_candidates:
            standard_candidates = [s for s in standard_ids if in_grade_scope(s, grade_key)] or standard_ids

        s_idx = [standard_to_idx[s] for s in standard_candidates]
        s_scores = np.dot(standard_mat[s_idx], q)
        pred_standards = select_topk_with_scores(standard_candidates, s_scores, k_standard)

        out[ex.problem_id] = {
            "domain": pred_domains,
            "cluster": pred_clusters,
            "standard": pred_standards,
        }
    return out


def gold_by_level(examples: Sequence[Example]) -> Dict[str, Dict[str, Set[str]]]:
    gold_std = {ex.problem_id: set(ex.labels) for ex in examples}
    gold_cluster = {pid: map_level(labels, "cluster") for pid, labels in gold_std.items()}
    gold_domain = {pid: map_level(labels, "domain") for pid, labels in gold_std.items()}
    return {"standard": gold_std, "cluster": gold_cluster, "domain": gold_domain}


def evaluate_stage_preds(preds: Dict[str, Dict[str, List[str]]], gold: Dict[str, Dict[str, Set[str]]]) -> Dict:
    domain_pred = {pid: set(v["domain"]) for pid, v in preds.items()}
    cluster_pred = {pid: set(v["cluster"]) for pid, v in preds.items()}
    standard_pred = {pid: set(v["standard"]) for pid, v in preds.items()}
    return {
        "domain": evaluate(domain_pred, gold["domain"], "standard"),
        "cluster": evaluate(cluster_pred, gold["cluster"], "standard"),
        "standard": evaluate(standard_pred, gold["standard"], "standard"),
    }


def error_propagation_stats(preds: Dict[str, Dict[str, List[str]]], gold: Dict[str, Dict[str, Set[str]]]) -> Dict:
    n = len(preds)
    wrong_domain = 0
    wrong_standard = 0
    wrong_standard_given_wrong_domain = 0
    for pid, row in preds.items():
        gd = gold["domain"].get(pid, set())
        gs = gold["standard"].get(pid, set())
        pd = set(row["domain"])
        ps = set(row["standard"])
        wd = pd != gd
        ws = ps != gs
        if wd:
            wrong_domain += 1
        if ws:
            wrong_standard += 1
        if wd and ws:
            wrong_standard_given_wrong_domain += 1
    return {
        "n": n,
        "wrong_domain_count": wrong_domain,
        "wrong_standard_count": wrong_standard,
        "wrong_standard_and_wrong_domain_count": wrong_standard_given_wrong_domain,
        "p_wrong_domain": wrong_domain / n if n else 0.0,
        "p_wrong_standard": wrong_standard / n if n else 0.0,
        "p_wrong_standard_given_wrong_domain": (
            wrong_standard_given_wrong_domain / wrong_domain if wrong_domain else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 adaptation: hierarchical prediction")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--disable-grade-filter", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="a3/krish/results/adaptations/hierarchical",
    )
    parser.add_argument("--epochs-note", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_publisher_examples("mathfish_train.jsonl")
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    split = split_examples(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by_source=True,
    )
    val_examples = split["val"]
    test_examples = split["test"]

    entries = load_standards("standards.jsonl")
    hierarchy = build_hierarchy(entries)

    domain_ids = sorted(hierarchy["domain_desc"].keys())
    cluster_ids = sorted(hierarchy["cluster_desc"].keys())
    standard_ids = sorted(hierarchy["standard_desc"].keys())

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    domain_texts = [f"{sid}. {hierarchy['domain_desc'][sid]}" for sid in domain_ids]
    cluster_texts = [f"{sid}. {hierarchy['cluster_desc'][sid]}" for sid in cluster_ids]
    standard_texts = [f"{sid}. {hierarchy['standard_desc'][sid]}" for sid in standard_ids]

    domain_mat = embed_texts(
        domain_texts, tokenizer=tokenizer, model=model, device=device, batch_size=args.batch_size, max_length=args.max_length
    )
    cluster_mat = embed_texts(
        cluster_texts, tokenizer=tokenizer, model=model, device=device, batch_size=args.batch_size, max_length=args.max_length
    )
    standard_mat = embed_texts(
        standard_texts, tokenizer=tokenizer, model=model, device=device, batch_size=args.batch_size, max_length=args.max_length
    )

    val_q = embed_texts(
        [x.text for x in val_examples],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    test_q = embed_texts(
        [x.text for x in test_examples],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    base_embeddings = {
        "domain": {
            "matrix": domain_mat,
            "id_to_idx": {sid: i for i, sid in enumerate(domain_ids)},
        },
        "cluster": {
            "matrix": cluster_mat,
            "id_to_idx": {sid: i for i, sid in enumerate(cluster_ids)},
        },
        "standard": {
            "matrix": standard_mat,
            "id_to_idx": {sid: i for i, sid in enumerate(standard_ids)},
        },
    }

    ids = {"domain": domain_ids, "cluster": cluster_ids, "standard": standard_ids}
    val_gold = gold_by_level(val_examples)
    test_gold = gold_by_level(test_examples)

    best = {"k_domain": 1, "k_cluster": 1, "k_standard": 1, "f1": -1.0, "metrics": {}}
    for kd in [1, 2, 3]:
        for kc in [1, 2, 3]:
            for ks in [1, 2, 3]:
                pred = predict_hierarchical(
                    val_examples,
                    hierarchy=hierarchy,
                    embeddings={**base_embeddings, "query": {"matrix": val_q}},
                    ids=ids,
                    k_domain=kd,
                    k_cluster=kc,
                    k_standard=ks,
                    grade_filter=not args.disable_grade_filter,
                )
                m = evaluate_stage_preds(pred, val_gold)
                f1 = m["standard"]["f1"]
                if f1 > best["f1"]:
                    best = {
                        "k_domain": kd,
                        "k_cluster": kc,
                        "k_standard": ks,
                        "f1": f1,
                        "metrics": m,
                    }

    test_pred = predict_hierarchical(
        test_examples,
        hierarchy=hierarchy,
        embeddings={**base_embeddings, "query": {"matrix": test_q}},
        ids=ids,
        k_domain=best["k_domain"],
        k_cluster=best["k_cluster"],
        k_standard=best["k_standard"],
        grade_filter=not args.disable_grade_filter,
    )
    test_metrics = evaluate_stage_preds(test_pred, test_gold)
    prop = error_propagation_stats(test_pred, test_gold)

    metrics = {
        "config": vars(args),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "best_val_decoding": {
            "k_domain": best["k_domain"],
            "k_cluster": best["k_cluster"],
            "k_standard": best["k_standard"],
            "metrics": best["metrics"],
        },
        "test_metrics": test_metrics,
        "error_propagation": prop,
    }
    write_json(out_dir / "metrics.json", metrics)

    standard_preds = {pid: row["standard"] for pid, row in test_pred.items()}
    write_predictions_jsonl(out_dir / "preds_test_standard.jsonl", standard_preds)
    write_json(out_dir / "preds_test_full.json", test_pred)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
