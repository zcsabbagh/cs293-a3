#!/usr/bin/env python3
"""
Adaptation 3: Standard description conditioning.

Compares two prompting setups on the same split:
1) Codes-only candidate list
2) Codes + full standard descriptions
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_benchmark
from a3.utils import (
    build_gold_dict,
    evaluate_all_levels,
    is_standard_in_grade_scope,
    load_assigned_problem_examples,
    load_publisher_examples,
    load_standards,
    parse_grade_key,
    split_examples,
    write_json,
    write_predictions_jsonl,
)


def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def build_prompt(problem_text: str, candidates: List[Tuple[str, str]], use_descriptions: bool) -> str:
    if use_descriptions:
        cands = "\n".join([f"- {sid}: {desc}" for sid, desc in candidates])
    else:
        cands = "\n".join([f"- {sid}" for sid, _ in candidates])
    mode = "codes and descriptions" if use_descriptions else "codes only"
    return (
        "You are a K-12 math curriculum expert.\n"
        "Predict CCSS standards directly addressed by the target problem.\n"
        "Return ONLY a JSON array of standard codes.\n\n"
        f"Candidate standards ({mode}):\n{cands}\n\n"
        f"Problem:\n{problem_text}\n\n"
        "Return format example: [\"4.NBT.A.1\", \"4.OA.A.3\"]"
    )


def call_model(provider: str, model: str, prompt: str) -> str:
    def do_call():
        if provider == "openai":
            return llm_benchmark.call_openai(prompt, model)
        if provider == "anthropic":
            return llm_benchmark.call_anthropic(prompt, model)
        if provider == "google":
            return llm_benchmark.call_gemini(prompt, model)
        raise ValueError(f"Unknown provider: {provider}")

    return llm_benchmark.request_with_retries(do_call)


def load_examples(dataset: str):
    if dataset == "publisher_full":
        return load_publisher_examples("mathfish_train.jsonl")
    if dataset == "assigned":
        return load_assigned_problem_examples("annotations/problems.json")
    raise ValueError(f"Unknown dataset: {dataset}")


def run_mode(
    mode_name: str,
    use_descriptions: bool,
    test_examples,
    all_standards: List[Tuple[str, str]],
    provider: str,
    model: str,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    preds: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    for ex in test_examples:
        grade_key = parse_grade_key(ex.metadata)
        candidates = [(sid, desc) for sid, desc in all_standards if is_standard_in_grade_scope(sid, grade_key)]
        if not candidates:
            candidates = all_standards
        prompt = build_prompt(ex.text, candidates, use_descriptions=use_descriptions)
        try:
            raw = call_model(provider, model, prompt)
            labels = llm_benchmark.extract_json_array(raw)
            valid = {sid for sid, _ in candidates}
            filtered = []
            for sid in labels:
                sid = sid.strip()
                if sid in valid and sid not in filtered:
                    filtered.append(sid)
            preds[ex.problem_id] = filtered
        except Exception as exc:
            preds[ex.problem_id] = []
            errors[ex.problem_id] = str(exc)
        time.sleep(0.1)
    return preds, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 adaptation: standard description conditioning")
    parser.add_argument("--dataset", choices=["publisher_full", "assigned"], default="publisher_full")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-test", type=int, default=0, help="0 = full test split")
    parser.add_argument(
        "--output-dir",
        default="a3/krish/results/adaptations/standard_conditioning",
    )
    args = parser.parse_args()

    load_env_file(".env")
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (loaded from env/.env).")
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set (loaded from env/.env).")
    if args.provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set (loaded from env/.env).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.dataset)
    split = split_examples(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by_source=True,
    )
    test = split["test"][: args.max_test] if args.max_test > 0 else split["test"]
    if not test:
        raise RuntimeError("Test split is empty; adjust split ratios or max-test.")
    gold = build_gold_dict(test)

    entries = load_standards("standards.jsonl")
    all_standards = [
        (sid, item.get("description", ""))
        for sid, item in entries.items()
        if item.get("level") in {"Standard", "Sub-standard"}
    ]

    codes_only_preds, codes_only_errors = run_mode(
        "codes_only",
        use_descriptions=False,
        test_examples=test,
        all_standards=all_standards,
        provider=args.provider,
        model=args.model,
    )
    write_predictions_jsonl(out_dir / "preds_codes_only.partial.jsonl", codes_only_preds)
    write_json(out_dir / "errors_codes_only.partial.json", codes_only_errors)

    desc_preds, desc_errors = run_mode(
        "codes_plus_descriptions",
        use_descriptions=True,
        test_examples=test,
        all_standards=all_standards,
        provider=args.provider,
        model=args.model,
    )

    m_codes = evaluate_all_levels({k: set(v) for k, v in codes_only_preds.items()}, gold)
    m_desc = evaluate_all_levels({k: set(v) for k, v in desc_preds.items()}, gold)

    payload = {
        "config": vars(args),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "evaluated_test_items": len(test),
        "error_count_codes_only": len(codes_only_errors),
        "error_count_codes_plus_descriptions": len(desc_errors),
        "metrics_codes_only": m_codes,
        "metrics_codes_plus_descriptions": m_desc,
    }
    write_json(out_dir / "metrics.json", payload)
    write_predictions_jsonl(out_dir / "preds_codes_only.jsonl", codes_only_preds)
    write_predictions_jsonl(out_dir / "preds_codes_plus_descriptions.jsonl", desc_preds)
    write_json(out_dir / "errors_codes_only.json", codes_only_errors)
    write_json(out_dir / "errors_codes_plus_descriptions.json", desc_errors)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
