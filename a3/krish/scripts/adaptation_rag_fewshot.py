#!/usr/bin/env python3
"""
Adaptation 2: Retrieval-augmented few-shot prompting.

This evaluates an LLM with retrieved in-context examples from annotated data.
Default split uses publisher labels for direct comparability with Section 2.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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
    parse_relations_arg,
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


def retrieve_topk(train_vecs: np.ndarray, query_vec: np.ndarray, k: int) -> List[int]:
    sims = np.dot(train_vecs, query_vec)
    idx = np.argsort(sims)[::-1][:k]
    return idx.tolist()


def build_prompt(
    problem_text: str,
    retrieved_examples: List[Dict],
    candidate_standards: List[tuple],
) -> str:
    candidates = "\n".join([f"- {sid}: {desc}" for sid, desc in candidate_standards])
    fewshot_chunks = []
    for i, ex in enumerate(retrieved_examples, start=1):
        fewshot_chunks.append(
            f"Example {i}\n"
            f"Problem: {ex['text']}\n"
            f"Labels: {json.dumps(ex['labels'])}"
        )
    fewshot_text = "\n\n".join(fewshot_chunks) if fewshot_chunks else "(none)"
    return (
        "You are a K-12 math curriculum expert.\n"
        "Task: predict the CCSS standards directly addressed by the target problem.\n"
        "Return ONLY a JSON array of codes.\n\n"
        "Candidate standards:\n"
        f"{candidates}\n\n"
        "Retrieved solved examples:\n"
        f"{fewshot_text}\n\n"
        f"Target problem:\n{problem_text}\n\n"
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


def load_examples(dataset: str, relations_csv: str):
    relations = parse_relations_arg(relations_csv)
    if dataset == "publisher_full":
        return load_publisher_examples("mathfish_train.jsonl", relations=relations)
    if dataset == "assigned":
        return load_assigned_problem_examples("annotations/problems.json", relations=relations)
    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 adaptation: retrieval-augmented few-shot prompting")
    parser.add_argument("--dataset", choices=["publisher_full", "assigned"], default="publisher_full")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--relations",
        default="Addressing,Alignment",
        help="Comma-separated relation labels to use as gold targets.",
    )
    parser.add_argument("--k-retrieval", type=int, default=3)
    parser.add_argument("--max-test", type=int, default=0, help="0 = full test split")
    parser.add_argument("--embed-model", default="roberta-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default="a3/krish/results/adaptations/rag_fewshot",
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

    examples = load_examples(args.dataset, args.relations)
    split = split_examples(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by_source=True,
    )
    train = split["train"]
    test = split["test"][: args.max_test] if args.max_test > 0 else split["test"]
    if not test:
        raise RuntimeError("Test split is empty; adjust split ratios or max-test.")
    gold = build_gold_dict(test)

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    embed_model = AutoModel.from_pretrained(args.embed_model).to(device)
    embed_model.eval()

    train_vecs = embed_texts(
        [x.text for x in train],
        tokenizer=tokenizer,
        model=embed_model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    test_vecs = embed_texts(
        [x.text for x in test],
        tokenizer=tokenizer,
        model=embed_model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    std_entries = load_standards("standards.jsonl")
    all_standards = [
        (sid, item.get("description", ""))
        for sid, item in std_entries.items()
        if item.get("level") in {"Standard", "Sub-standard"}
    ]

    preds: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    retrieval_log = {}
    for i, (ex, q) in enumerate(zip(test, test_vecs), start=1):
        idx = retrieve_topk(train_vecs, q, args.k_retrieval)
        retrieved = [train[i] for i in idx]
        grade_key = parse_grade_key(ex.metadata)
        candidates = [(sid, desc) for sid, desc in all_standards if is_standard_in_grade_scope(sid, grade_key)]
        if not candidates:
            candidates = all_standards
        prompt = build_prompt(
            problem_text=ex.text,
            retrieved_examples=[{"problem_id": r.problem_id, "text": r.text, "labels": list(r.labels)} for r in retrieved],
            candidate_standards=candidates,
        )
        try:
            raw = call_model(args.provider, args.model, prompt)
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
        retrieval_log[ex.problem_id] = [r.problem_id for r in retrieved]
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        if i % 25 == 0:
            write_predictions_jsonl(out_dir / "preds_test.partial.jsonl", preds)
            write_json(out_dir / "errors.partial.json", errors)
            print(f"Processed {i}/{len(test)} test items")

    metrics = evaluate_all_levels({k: set(v) for k, v in preds.items()}, gold)
    payload = {
        "config": vars(args),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "evaluated_test_items": len(test),
        "error_count": len(errors),
        "metrics": metrics,
    }
    write_json(out_dir / "metrics.json", payload)
    write_predictions_jsonl(out_dir / "preds_test.jsonl", preds)
    write_json(out_dir / "retrieval_log.json", retrieval_log)
    write_json(out_dir / "errors.json", errors)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
