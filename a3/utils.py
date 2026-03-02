#!/usr/bin/env python3
"""Shared utilities for A3 modeling and evaluation."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


PUBLISHER_RELATIONS = {"Addressing", "Alignment"}


def parse_relations_arg(relations_csv: Optional[str], default: Optional[Set[str]] = None) -> Set[str]:
    """Parse a comma-separated relation list into a normalized set."""
    if relations_csv is None:
        return set(default or PUBLISHER_RELATIONS)
    out = {x.strip() for x in relations_csv.split(",") if x.strip()}
    if not out:
        raise ValueError("No relations provided after parsing --relations.")
    return out


@dataclass(frozen=True)
class Example:
    """A single labeled math problem example."""

    problem_id: str
    text: str
    labels: Tuple[str, ...]
    source: str
    metadata: Dict


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def normalize_problem_text(text: str, elements: Optional[Dict[str, str]] = None) -> str:
    if not text:
        return ""
    out = text
    if elements:
        for placeholder, html_or_asset in elements.items():
            if isinstance(html_or_asset, str):
                out = out.replace(placeholder, strip_html(html_or_asset))
    out = re.sub(r"###[A-Z0-9_]+###", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def load_standards(path: Union[str, Path] = "standards.jsonl") -> Dict[str, Dict]:
    entries: Dict[str, Dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            entries[item["id"]] = item
    return entries


def load_standard_descriptions(
    path: Union[str, Path] = "standards.jsonl",
    levels: Optional[Set[str]] = None,
) -> Dict[str, str]:
    levels = levels or {"Standard", "Sub-standard"}
    descriptions: Dict[str, str] = {}
    for sid, item in load_standards(path).items():
        if item.get("level") in levels:
            descriptions[sid] = item.get("description", "").strip()
    return descriptions


def _extract_relation_labels(
    relations_and_codes: Sequence[Sequence[str]],
    relations: Set[str],
) -> List[str]:
    labels = []
    for pair in relations_and_codes:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        relation, code = pair
        if relation in relations:
            labels.append(str(code))
    return sorted(set(labels))


def load_publisher_examples(
    path: Union[str, Path] = "mathfish_train.jsonl",
    relations: Optional[Set[str]] = None,
    include_images: bool = False,
    include_duplicates: bool = False,
    min_text_len: int = 20,
    max_text_len: int = 4000,
) -> List[Example]:
    relations = relations or PUBLISHER_RELATIONS
    examples: List[Example] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if not include_images and item.get("has_image", False):
                continue
            if not include_duplicates and item.get("is_duplicate", False):
                continue
            labels = _extract_relation_labels(item.get("standards", []), relations)
            if not labels:
                continue
            text = normalize_problem_text(item.get("text", ""), item.get("elements", {}))
            if len(text) < min_text_len or len(text) > max_text_len:
                continue
            examples.append(
                Example(
                    problem_id=item["id"],
                    text=text,
                    labels=tuple(labels),
                    source=item.get("source", ""),
                    metadata=item.get("metadata", {}),
                )
            )
    return examples


def load_assigned_problem_examples(
    path: Union[str, Path] = "annotations/problems.json",
    relations: Optional[Set[str]] = None,
) -> List[Example]:
    relations = relations or PUBLISHER_RELATIONS
    with open(path) as f:
        data = json.load(f)

    examples: List[Example] = []
    for pid, item in data.items():
        labels = _extract_relation_labels(item.get("standards", []), relations)
        if not labels:
            continue
        text = normalize_problem_text(item.get("text", ""), item.get("elements", {}))
        if not text:
            continue
        examples.append(
            Example(
                problem_id=pid,
                text=text,
                labels=tuple(labels),
                source=item.get("source", ""),
                metadata=item.get("metadata", {}),
            )
        )
    return examples


def split_examples(
    examples: Sequence[Example],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_source: bool = False,
) -> Dict[str, List[Example]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = random.Random(seed)

    if not stratify_by_source:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        return _split_single_bucket(shuffled, train_ratio, val_ratio)

    by_source: Dict[str, List[Example]] = defaultdict(list)
    for ex in examples:
        by_source[ex.source].append(ex)

    train: List[Example] = []
    val: List[Example] = []
    test: List[Example] = []
    for source_examples in by_source.values():
        shuffled = list(source_examples)
        rng.shuffle(shuffled)
        parts = _split_single_bucket(shuffled, train_ratio, val_ratio)
        train.extend(parts["train"])
        val.extend(parts["val"])
        test.extend(parts["test"])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def _split_single_bucket(
    shuffled: List[Example],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Example]]:
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def build_label_space(examples: Sequence[Example], min_freq: int = 1) -> List[str]:
    counts: Counter[str] = Counter()
    for ex in examples:
        counts.update(set(ex.labels))
    return sorted([sid for sid, c in counts.items() if c >= min_freq])


def standard_levels(code: str) -> Tuple[str, str, str]:
    parts = code.split(".")
    domain = ".".join(parts[:2]) if len(parts) >= 2 else code
    cluster = ".".join(parts[:3]) if len(parts) >= 3 else domain
    return domain, cluster, code


def map_level(codes: Iterable[str], level: str) -> Set[str]:
    mapped: Set[str] = set()
    for code in codes:
        domain, cluster, standard = standard_levels(code)
        if level == "domain":
            mapped.add(domain)
        elif level == "cluster":
            mapped.add(cluster)
        elif level == "standard":
            mapped.add(standard)
        else:
            raise ValueError(f"Unknown level: {level}")
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


def evaluate_all_levels(
    preds: Dict[str, Set[str]],
    gold: Dict[str, Set[str]],
) -> Dict[str, Dict[str, float]]:
    return {level: evaluate(preds, gold, level) for level in ("standard", "cluster", "domain")}


def build_gold_dict(examples: Sequence[Example]) -> Dict[str, Set[str]]:
    return {ex.problem_id: set(ex.labels) for ex in examples}


def write_json(path: Union[str, Path], data: Dict) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_predictions_jsonl(path: Union[str, Path], preds: Dict[str, Sequence[str]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for problem_id, labels in preds.items():
            f.write(
                json.dumps({"problem_id": problem_id, "predicted": list(labels)}, ensure_ascii=False)
                + "\n"
            )


def load_predictions_jsonl(path: Union[str, Path]) -> Dict[str, Set[str]]:
    preds: Dict[str, Set[str]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            preds[item["problem_id"]] = set(item.get("predicted", []))
    return preds


def load_annotation_file_latest(path: Union[str, Path]) -> Dict[str, Dict]:
    path = Path(path)
    by_problem: Dict[str, Dict] = {}
    with open(path) as f:
        content = f.read().strip()
        if not content:
            return by_problem
        if content.startswith("["):
            entries = json.loads(content)
            for row in entries:
                by_problem[row["problem_id"]] = row
            return by_problem
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_problem[row["problem_id"]] = row
    return by_problem


def load_all_annotation_files_latest(annotations_dir: Union[str, Path] = "annotations") -> Dict[str, Dict[str, Dict]]:
    ann_dir = Path(annotations_dir)
    annotator_to_rows: Dict[str, Dict[str, Dict]] = {}
    for path in sorted(ann_dir.glob("*_annotations.json*")):
        annotator = (
            path.name.replace("_annotations.jsonl", "").replace("_annotations.json", "")
        )
        annotator_to_rows[annotator] = load_annotation_file_latest(path)
    return annotator_to_rows


def extract_annotation_standard_ids(row: Dict) -> List[str]:
    if row.get("skipped"):
        return []
    ids = []
    for s in row.get("standards", []):
        if isinstance(s, dict):
            sid = s.get("id")
            if sid:
                ids.append(sid)
        else:
            ids.append(str(s))
    return sorted(set(ids))


def majority_consensus_labels(
    annotator_rows: Dict[str, Dict[str, Dict]],
    problem_ids: Sequence[str],
    required_votes: Optional[int] = None,
) -> Dict[str, Dict]:
    annotators = sorted(annotator_rows.keys())
    n_annotators = len(annotators)
    if required_votes is None:
        required_votes = (n_annotators // 2) + 1
    result: Dict[str, Dict] = {}

    for pid in problem_ids:
        votes: Counter[str] = Counter()
        by_annotator: Dict[str, List[str]] = {}
        for annotator in annotators:
            row = annotator_rows.get(annotator, {}).get(pid, {})
            labels = extract_annotation_standard_ids(row) if row else []
            by_annotator[annotator] = labels
            votes.update(labels)
        consensus = sorted([sid for sid, c in votes.items() if c >= required_votes])
        result[pid] = {
            "problem_id": pid,
            "consensus_labels": consensus,
            "vote_counts": dict(sorted(votes.items())),
            "by_annotator": by_annotator,
            "required_votes": required_votes,
            "needs_adjudication": len(consensus) == 0 or any(c == required_votes - 1 for c in votes.values()),
        }
    return result


def parse_grade_key(meta: Dict) -> Optional[Union[str, List[str]]]:
    raw = meta.get("grade / subject") or meta.get("grade") or meta.get("subject")
    if not raw:
        return None
    val = str(raw).strip().lower()
    if "kindergarten" in val or val == "k":
        return "K"
    m = re.search(r"grade-(\d)", val)
    if m:
        return m.group(1)
    m = re.search(r"(\d)(st|nd|rd|th)-grade", val)
    if m:
        return m.group(1)
    if "algebra-1" in val or "algebra 1" in val or "algebra-2" in val or "algebra 2" in val:
        return ["A", "F", "N", "S"]
    if "geometry" in val:
        return ["G", "N"]
    if "high school" in val or val.startswith("hs"):
        return ["A", "F", "G", "N", "S"]
    return None


def is_standard_in_grade_scope(
    standard_id: str,
    grade_key: Optional[Union[str, List[str]]],
) -> bool:
    if grade_key is None:
        return True
    if isinstance(grade_key, list):
        return any(standard_id.startswith(f"{prefix}-") for prefix in grade_key)
    if grade_key == "K":
        return standard_id.startswith("K.")
    return standard_id.startswith(f"{grade_key}.")
