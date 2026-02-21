#!/usr/bin/env python3
"""
Benchmark models against publisher labels for MathFish.

Usage examples:
  python3 benchmark.py tfidf --k 3 --output preds/tfidf_k3.jsonl
  python3 benchmark.py eval --preds preds/tfidf_k3.jsonl

Prediction JSONL format:
  {"problem_id": "...", "predicted": ["4.NBT.A.1", "4.NBT.A.2"]}
"""

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Set, Tuple


PUBLISHER_RELATIONS = {"Addressing", "Alignment"}


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def normalize_problem_text(text: str, elements: Dict[str, str]) -> str:
    if not text:
        return ""
    out = text
    if elements:
        for placeholder, html in elements.items():
            out = out.replace(placeholder, strip_html(html))
    # Remove any leftover placeholders
    out = re.sub(r"###[A-Z0-9_]+###", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def load_problems(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        data = json.load(f)
    problems = {}
    for pid, p in data.items():
        problems[pid] = {
            "id": p["id"],
            "text": normalize_problem_text(p.get("text", ""), p.get("elements", {})),
            "standards": p.get("standards", []),
        }
    return problems


def load_standards(path: str) -> List[Tuple[str, str]]:
    standards = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("level") in {"Standard", "Sub-standard"}:
                standards.append((item["id"], item.get("description", "")))
    return standards


def build_gold_labels(problems: Dict[str, Dict]) -> Dict[str, Set[str]]:
    gold = {}
    for pid, p in problems.items():
        labels = set()
        for rel, code in p.get("standards", []):
            if rel in PUBLISHER_RELATIONS:
                labels.add(code)
        gold[pid] = labels
    return gold


def standard_levels(code: str) -> Tuple[str, str, str]:
    parts = code.split(".")
    if len(parts) >= 2:
        domain = ".".join(parts[:2])
    else:
        domain = code
    if len(parts) >= 3:
        cluster = ".".join(parts[:3])
    else:
        cluster = domain
    standard = code
    return domain, cluster, standard


def map_level(codes: Iterable[str], level: str) -> Set[str]:
    mapped = set()
    for code in codes:
        domain, cluster, standard = standard_levels(code)
        if level == "domain":
            mapped.add(domain)
        elif level == "cluster":
            mapped.add(cluster)
        else:
            mapped.add(standard)
    return mapped


def evaluate(
    preds: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
    level: str,
) -> Dict[str, float]:
    tp = fp = fn = 0
    exact = 0
    total = 0
    for pid, gold_labels in gold.items():
        total += 1
        pred_labels = preds.get(pid, set())
        pred_level = map_level(pred_labels, level)
        gold_level = map_level(gold_labels, level)
        tp += len(pred_level & gold_level)
        fp += len(pred_level - gold_level)
        fn += len(gold_level - pred_level)
        if pred_level == gold_level:
            exact += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / total if total else 0.0,
        "total": total,
    }


def load_preds(path: str) -> Dict[str, Set[str]]:
    preds = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            preds[item["problem_id"]] = set(item.get("predicted", []))
    return preds


def write_preds(path: str, preds: Dict[str, List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for pid, labels in preds.items():
            f.write(json.dumps({"problem_id": pid, "predicted": labels}) + "\n")


def run_tfidf(
    problems: Dict[str, Dict],
    standards: List[Tuple[str, str]],
    k: int,
) -> Dict[str, List[str]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for the TF-IDF baseline. "
            "Install with: pip install scikit-learn"
        ) from e

    std_ids = [sid for sid, _ in standards]
    std_texts = [f"{sid} {desc}".strip() for sid, desc in standards]
    problem_texts = [p["text"] for p in problems.values()]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(std_texts + problem_texts)

    std_matrix = vectorizer.transform(std_texts)

    preds = {}
    for pid, p in problems.items():
        vec = vectorizer.transform([p["text"]])
        sims = cosine_similarity(vec, std_matrix).flatten()
        topk_idx = sims.argsort()[::-1][:k]
        preds[pid] = [std_ids[i] for i in topk_idx]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="MathFish benchmarking script")
    sub = parser.add_subparsers(dest="command", required=True)

    tfidf = sub.add_parser("tfidf", help="Run TF-IDF baseline")
    tfidf.add_argument("--k", type=int, default=3)
    tfidf.add_argument("--output", default="preds/tfidf_k3.jsonl")

    ev = sub.add_parser("eval", help="Evaluate predictions vs publisher labels")
    ev.add_argument("--preds", required=True, help="Predictions JSONL file")

    args = parser.parse_args()

    problems = load_problems("annotations/problems.json")
    gold = build_gold_labels(problems)

    if args.command == "tfidf":
        standards = load_standards("standards.jsonl")
        preds = run_tfidf(problems, standards, args.k)
        write_preds(args.output, preds)
        print(f"Wrote predictions to {args.output}")
        print("Run: python3 benchmark.py eval --preds", args.output)
        return

    if args.command == "eval":
        preds = load_preds(args.preds)
        for level in ["standard", "cluster", "domain"]:
            metrics = evaluate(preds, gold, level)
            print(f"\nLevel: {level}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1:        {metrics['f1']:.3f}")
            print(f"  Exact:     {metrics['exact_match']:.3f}")
            print(f"  N:         {metrics['total']}")


if __name__ == "__main__":
    main()
