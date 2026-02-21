#!/usr/bin/env python3
"""
Run LLM benchmarks against publisher labels and write predictions + metrics.

Usage:
  python3 llm_benchmark.py all --output-dir preds --results results/llm_results.json
"""

import argparse
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import requests

import benchmark


MODELS = {
    "google": "gemini-3.1-pro-preview",
    "openai": "gpt-5.2",
    "anthropic": "claude-sonnet-4-6",
}


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def normalize_problem_text(text: str, elements: Dict[str, str]) -> str:
    if not text:
        return ""
    out = text
    if elements:
        for placeholder, html in elements.items():
            out = out.replace(placeholder, strip_html(html))
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
            "metadata": p.get("metadata", {}),
        }
    return problems


def load_standards(path: str) -> Dict[str, Dict]:
    entries = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            entries[item["id"]] = item
    return entries


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


def is_domain_in_scope(domain_id: str, grade_key: Union[str, List[str], None]) -> bool:
    if grade_key is None:
        return True
    if isinstance(grade_key, list):
        return any(domain_id.startswith(f"{p}-") for p in grade_key)
    if grade_key == "K":
        return domain_id.startswith("K.")
    return domain_id.startswith(f"{grade_key}.")


def candidate_standards(entries: Dict[str, Dict], grade_key: Union[str, List[str], None]) -> List[Tuple[str, str]]:
    out = []
    for sid, item in entries.items():
        if item.get("level") not in {"Standard", "Sub-standard"}:
            continue
        if grade_key is None:
            out.append((sid, item.get("description", "")))
            continue
        if isinstance(grade_key, list):
            if any(sid.startswith(f"{p}-") for p in grade_key):
                out.append((sid, item.get("description", "")))
        else:
            if grade_key == "K":
                if sid.startswith("K."):
                    out.append((sid, item.get("description", "")))
            elif sid.startswith(f"{grade_key}."):
                out.append((sid, item.get("description", "")))
    return out


def build_hierarchy(entries: Dict[str, Dict], grade_key: Union[str, List[str], None]) -> str:
    lines = []
    domains = [e for e in entries.values() if e.get("level") == "Domain" and is_domain_in_scope(e["id"], grade_key)]
    domains = sorted(domains, key=lambda d: d["id"])
    for domain in domains:
        lines.append(f"Domain {domain['id']}: {domain.get('description','')}")
        for cluster_id in domain.get("children", []):
            cluster = entries.get(cluster_id)
            if not cluster:
                continue
            ctype = cluster.get("cluster_type", "").strip()
            ctype_txt = f" ({ctype})" if ctype else ""
            lines.append(f"  Cluster {cluster_id}{ctype_txt}: {cluster.get('description','')}")
            for std_id in cluster.get("children", []):
                std = entries.get(std_id)
                if not std:
                    continue
                lines.append(f"    Standard {std_id}: {std.get('description','')}")
                for sub_id in std.get("children", []):
                    sub = entries.get(sub_id)
                    if not sub:
                        continue
                    lines.append(f"      Sub-standard {sub_id}: {sub.get('description','')}")
    return "\n".join(lines)


def extract_json_array(text: str) -> List[str]:
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        return []
    return []


def build_prompt(problem_text: str, hierarchy: str) -> str:
    return (
        "You are a K-12 math curriculum expert. Given this math problem, identify "
        "which Common Core standard(s) it directly addresses (the \"Addressing\" relation).\n\n"
        "Use this standards hierarchy to narrow your answer:\n"
        f"{hierarchy}\n\n"
        f"Problem: {problem_text}\n\n"
        "Return ONLY a JSON array of standard codes, e.g. [\"4.NBT.A.1\", \"4.OA.A.3\"]."
    )


def request_with_retries(fn, retries: int = 3, base_sleep: float = 1.5):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** i))
    raise last_err


def call_openai(prompt: str, model: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON arrays of CCSS codes."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = 300
    else:
        payload["max_tokens"] = 300
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_anthropic(prompt: str, model: str) -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 300,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "system": "Return only JSON arrays of CCSS codes.",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic error {resp.status_code}: {resp.text}")
    data = resp.json()
    parts = data.get("content", [])
    text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return "\n".join(text_parts).strip()


def call_gemini(prompt: str, model: str) -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini error {resp.status_code}: {resp.text}")
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "\n".join(text_parts).strip()


def run_model(
    provider: str,
    model: str,
    problems: Dict[str, Dict],
    entries: Dict[str, Dict],
    standards_set: Set[str],
) -> Dict[str, List[str]]:
    preds = {}
    for pid, p in problems.items():
        grade_key = parse_grade_key(p.get("metadata", {}))
        hierarchy = build_hierarchy(entries, grade_key)
        prompt = build_prompt(p["text"], hierarchy)

        def do_call():
            if provider == "openai":
                return call_openai(prompt, model)
            if provider == "anthropic":
                return call_anthropic(prompt, model)
            if provider == "google":
                return call_gemini(prompt, model)
            raise ValueError(f"Unknown provider: {provider}")

        raw = request_with_retries(do_call)
        labels = extract_json_array(raw)
        filtered = []
        for code in labels:
            code = code.strip()
            if code in standards_set and code not in filtered:
                filtered.append(code)
        preds[pid] = filtered
        time.sleep(0.1)
    return preds


def write_preds(path: str, preds: Dict[str, List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for pid, labels in preds.items():
            f.write(json.dumps({"problem_id": pid, "predicted": labels}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    all_cmd = sub.add_parser("all", help="Run all providers")
    all_cmd.add_argument("--output-dir", default="preds")
    all_cmd.add_argument("--results", default="results/llm_results.json")
    all_cmd.add_argument(
        "--providers",
        default="google,openai,anthropic",
        help="Comma-separated list: google,openai,anthropic",
    )
    args = parser.parse_args()

    problems = load_problems("annotations/problems.json")
    entries = load_standards("standards.jsonl")
    standards_set = {sid for sid, item in entries.items() if item.get("level") in {"Standard", "Sub-standard"}}
    gold = benchmark.build_gold_labels(benchmark.load_problems("annotations/problems.json"))

    want = [p.strip() for p in args.providers.split(",") if p.strip()]
    results = {}
    for provider, model in MODELS.items():
        if provider not in want:
            continue
        try:
            preds = run_model(provider, model, problems, entries, standards_set)
            out_path = os.path.join(args.output_dir, f"{provider}_{model}.jsonl")
            write_preds(out_path, preds)
            metrics = {}
            for level in ["standard", "cluster", "domain"]:
                metrics[level] = benchmark.evaluate(
                    {k: set(v) for k, v in preds.items()},
                    gold,
                    level,
                )
            results[f"{provider}:{model}"] = {"preds": out_path, "metrics": metrics}
        except Exception as e:
            results[f"{provider}:{model}"] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {args.results}")


if __name__ == "__main__":
    main()
