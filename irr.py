#!/usr/bin/env python3
"""
Compute Krippendorff's alpha for shared problems.

Usage:
  python3 irr.py --output results/irr.json
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Set, Tuple


def load_shared_ids(path: str) -> List[str]:
    with open(path) as f:
        data = json.load(f)
    return data.get("shared_ids", [])


def load_annotations(paths: Iterable[str]) -> Dict[str, Dict[str, Dict]]:
    annotations = {}
    for path in paths:
        base = os.path.basename(path)
        annotator = base.replace("_annotations.jsonl", "").replace("_annotations.json", "")
        annotations[annotator] = {}
        with open(path) as f:
            content = f.read().strip()
            if not content:
                continue
            if content.startswith("["):
                items = json.loads(content)
                for a in items:
                    annotations[annotator][a["problem_id"]] = a
            else:
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    a = json.loads(line)
                    annotations[annotator][a["problem_id"]] = a
    return annotations


def standard_levels(code: str) -> Tuple[str, str, str]:
    parts = code.split(".")
    domain = ".".join(parts[:2]) if len(parts) >= 2 else code
    cluster = ".".join(parts[:3]) if len(parts) >= 3 else domain
    return domain, cluster, code


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


def build_reliability_matrix(
    annotations: Dict[str, Dict[str, Dict]],
    shared_ids: List[str],
    level: str,
) -> List[List[float]]:
    all_standards = set()
    for ann in annotations.values():
        for pid in shared_ids:
            a = ann.get(pid)
            if not a or a.get("skipped"):
                continue
            standards = [s["id"] if isinstance(s, dict) else s for s in a.get("standards", [])]
            all_standards.update(map_level(standards, level))

    items = [(pid, code) for pid in shared_ids for code in sorted(all_standards)]
    matrix = []
    for annotator, ann in annotations.items():
        row = []
        for pid, code in items:
            a = ann.get(pid)
            if not a or a.get("skipped"):
                row.append(float("nan"))
                continue
            standards = [s["id"] if isinstance(s, dict) else s for s in a.get("standards", [])]
            mapped = map_level(standards, level)
            row.append(1.0 if code in mapped else 0.0)
        matrix.append(row)
    return matrix


def compute_alpha(matrix: List[List[float]]) -> float:
    try:
        import numpy as np
        import krippendorff
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install with: pip install krippendorff numpy"
        ) from e
    arr = np.array(matrix, dtype=float)
    return float(krippendorff.alpha(reliability_data=arr, level_of_measurement="nominal"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IRR on shared problems")
    parser.add_argument("--output", default="results/irr.json")
    args = parser.parse_args()

    shared_ids = load_shared_ids("annotations/assignments.json")
    ann_files = [
        os.path.join("annotations", f)
        for f in os.listdir("annotations")
        if f.endswith("_annotations.jsonl") or f.endswith("_annotations.json")
    ]
    if not ann_files:
        raise RuntimeError("No annotation files found in annotations/")

    annotations = load_annotations(ann_files)

    results = {"annotators": sorted(annotations.keys()), "shared_count": len(shared_ids)}
    for level in ["standard", "cluster", "domain"]:
        matrix = build_reliability_matrix(annotations, shared_ids, level)
        alpha = compute_alpha(matrix)
        results[level] = {
            "alpha": alpha,
            "items": len(matrix[0]) if matrix else 0,
            "annotators": len(matrix),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote IRR results to {args.output}")


if __name__ == "__main__":
    main()
