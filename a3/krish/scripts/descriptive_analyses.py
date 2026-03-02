#!/usr/bin/env python3
"""
Section 4 descriptive analyses at scale.

Generates:
1) Standards coverage heatmap (grade x domain)
2) Prerequisite chain graph (heuristic Building-On predictions)
3) Publisher comparison figure

By default this script predicts standards across full MathFish using a
RoBERTa description-similarity model (scalable local baseline).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.utils import (
    is_standard_in_grade_scope,
    normalize_problem_text,
    parse_grade_key,
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
    batch_size: int = 32,
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


def load_standards(path: str = "standards.jsonl") -> Dict[str, Dict]:
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def load_mathfish(path: str = "mathfish_train.jsonl") -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def predict_with_roberta_similarity(
    rows: List[Dict],
    standards: Dict[str, Dict],
    model_name: str,
    top_k: int,
    sim_threshold: float,
    rel_margin: float,
    grade_filter: bool,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Dict[str, List[str]]:
    standard_ids = sorted(
        [sid for sid, item in standards.items() if item.get("level") in {"Standard", "Sub-standard"}]
    )
    standard_texts = [f"{sid}. {standards[sid].get('description','')}" for sid in standard_ids]
    sid_to_idx = {sid: i for i, sid in enumerate(standard_ids)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    std_mat = embed_texts(
        standard_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    texts = [normalize_problem_text(r.get("text", ""), r.get("elements", {})) for r in rows]
    q_mat = embed_texts(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    preds: Dict[str, List[str]] = {}
    for r, q in zip(rows, q_mat):
        grade_key = parse_grade_key(r.get("metadata", {})) if grade_filter else None
        if grade_filter:
            candidates = [sid for sid in standard_ids if is_standard_in_grade_scope(sid, grade_key)]
            if not candidates:
                candidates = standard_ids
        else:
            candidates = standard_ids
        c_idx = [sid_to_idx[sid] for sid in candidates]
        sims = np.dot(std_mat[c_idx], q)
        order = np.argsort(sims)[::-1]
        cutoff = max(sim_threshold, float(sims[order[0]]) - rel_margin)
        keep = [i for i in order if sims[i] >= cutoff]
        if not keep:
            keep = [int(order[0])]
        keep = keep[: max(1, top_k)]
        preds[r["id"]] = [candidates[i] for i in keep]
    return preds


def standard_to_domain(code: str) -> str:
    parts = code.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else code


def domain_to_grade(domain: str) -> str:
    head = domain.split(".")[0]
    if "-" in head:
        return f"HS-{head.split('-')[0]}"
    return head


def build_prereq_mapping_from_publisher(rows: List[Dict], top_n: int = 2) -> Dict[str, List[str]]:
    counts = defaultdict(Counter)
    for r in rows:
        addressing = [sid for rel, sid in r.get("standards", []) if rel in {"Addressing", "Alignment"}]
        building = [sid for rel, sid in r.get("standards", []) if rel == "Building On"]
        if not addressing or not building:
            continue
        for a in set(addressing):
            counts[a].update(set(building))
    mapping = {}
    for a, ctr in counts.items():
        mapping[a] = [sid for sid, _ in ctr.most_common(top_n)]
    return mapping


def plot_coverage_heatmap(
    domain_counts_by_grade: Dict[str, Counter],
    out_path: Path,
) -> Dict[str, List[str]]:
    grades = sorted(domain_counts_by_grade.keys(), key=lambda g: (g.startswith("HS"), g))
    domains = sorted({d for c in domain_counts_by_grade.values() for d in c.keys()})
    if not grades or not domains:
        raise RuntimeError("No data for coverage heatmap.")

    mat = np.zeros((len(grades), len(domains)), dtype=float)
    for i, g in enumerate(grades):
        for j, d in enumerate(domains):
            mat[i, j] = domain_counts_by_grade[g].get(d, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(domains) * 0.35), max(5, len(grades) * 0.55)))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(grades)))
    ax.set_yticklabels(grades)
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, rotation=90, fontsize=8)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Grade / HS Category")
    ax.set_title("Predicted Standards Coverage (Grade x Domain)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Predicted label count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {"grades": grades, "domains": domains}


def plot_prereq_graph(edge_counts: Counter, out_path: Path, top_edges: int = 40) -> List[Tuple[str, str, int]]:
    top = edge_counts.most_common(top_edges)
    if not top:
        raise RuntimeError("No prerequisite edges found.")
    G = nx.DiGraph()
    for (src, dst), w in top:
        G.add_edge(src, dst, weight=w)

    pos = nx.spring_layout(G, seed=42, k=1.1 / np.sqrt(max(G.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=450, node_color="#E3F2FD", edgecolors="#1565C0", ax=ax)
    widths = [0.5 + 3.0 * (G[u][v]["weight"] / top[0][1]) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        edge_color="#1976D2",
        alpha=0.7,
        arrows=True,
        arrowsize=12,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title("Predicted Prerequisite Chains (heuristic Building On)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return [(src, dst, int(w)) for (src, dst), w in top]


def plot_publisher_comparison(
    stats: Dict[str, Dict],
    top_domains: List[str],
    domain_share: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    pubs = sorted(stats.keys())
    avg_standards = [stats[p]["avg_standards_per_problem"] for p in pubs]
    pct_multi = [stats[p]["pct_multi_standard"] for p in pubs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    x = np.arange(len(pubs))
    ax1.bar(x, avg_standards, color=["#1E88E5", "#43A047"][: len(pubs)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(pubs, rotation=10)
    ax1.set_title("Avg Predicted Standards per Problem")
    ax1.set_ylabel("Average count")

    width = 0.35
    ax2.bar(x - width / 2, pct_multi, width=width, label="% Multi-standard", color="#8E24AA")
    ax2.bar(
        x + width / 2,
        [100 - v for v in pct_multi],
        width=width,
        label="% Single-standard",
        color="#FB8C00",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(pubs, rotation=10)
    ax2.set_title("Single vs Multi-standard Problems")
    ax2.set_ylabel("Percent")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    # Supplemental top-domain comparison chart.
    sup_path = out_path.with_name(out_path.stem + "_domains.png")
    fig2, ax = plt.subplots(figsize=(12, 5))
    bar_x = np.arange(len(top_domains))
    w = 0.38
    for i, p in enumerate(pubs):
        vals = [100.0 * domain_share[p].get(d, 0.0) for d in top_domains]
        ax.bar(bar_x + (i - (len(pubs) - 1) / 2) * w, vals, width=w, label=p)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(top_domains, rotation=45, ha="right")
    ax.set_ylabel("% of predicted domain tags")
    ax.set_title("Top Domain Emphasis by Publisher")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(sup_path, dpi=220)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 descriptive analyses at scale")
    parser.add_argument("--data-path", default="mathfish_train.jsonl")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum predicted standards per problem")
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.28,
        help="Cosine-similarity threshold for multi-label prediction (fallback is top-1).",
    )
    parser.add_argument(
        "--rel-margin",
        type=float,
        default=0.02,
        help="Keep labels within this margin of the top similarity score.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-grade-filter", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="a3/krish/results/descriptive",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("a3/krish/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = load_mathfish(args.data_path)
    standards = load_standards("standards.jsonl")
    device = pick_device(args.device)

    preds = predict_with_roberta_similarity(
        rows=rows,
        standards=standards,
        model_name=args.model_name,
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
        rel_margin=args.rel_margin,
        grade_filter=not args.no_grade_filter,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    full_pred_rows = {}
    for r in rows:
        pid = r["id"]
        full_pred_rows[pid] = {
            "problem_id": pid,
            "source": r.get("source", ""),
            "metadata": r.get("metadata", {}),
            "predicted_standards": preds.get(pid, []),
        }

    # Analysis 1: Standards coverage map.
    domain_counts_by_grade: Dict[str, Counter] = defaultdict(Counter)
    standard_counts = Counter()
    for row in full_pred_rows.values():
        for sid in row["predicted_standards"]:
            standard_counts[sid] += 1
            dom = standard_to_domain(sid)
            grade = domain_to_grade(dom)
            domain_counts_by_grade[grade][dom] += 1

    coverage_meta = plot_coverage_heatmap(
        domain_counts_by_grade,
        out_path=fig_dir / "analysis1_coverage_heatmap.png",
    )
    rare_or_zero = [sid for sid, item in standards.items() if item.get("level") in {"Standard", "Sub-standard"} and standard_counts.get(sid, 0) == 0]

    # Analysis 2: prerequisite chains via heuristic Building On mapping.
    prereq_map = build_prereq_mapping_from_publisher(rows, top_n=2)
    prereq_edges = Counter()
    prereq_predictions = {}
    for pid, row in full_pred_rows.items():
        pr = []
        for target in row["predicted_standards"]:
            for pre in prereq_map.get(target, []):
                prereq_edges[(pre, target)] += 1
                pr.append(pre)
        prereq_predictions[pid] = sorted(set(pr))

    top_chain_edges = plot_prereq_graph(
        prereq_edges,
        out_path=fig_dir / "analysis2_prereq_graph.png",
        top_edges=40,
    )

    # Analysis 3: publisher comparison.
    by_pub = defaultdict(list)
    pub_domain_counts = defaultdict(Counter)
    for pid, row in full_pred_rows.items():
        pub = row["source"] or "Unknown"
        labels = row["predicted_standards"]
        by_pub[pub].append(len(labels))
        for sid in labels:
            pub_domain_counts[pub][standard_to_domain(sid)] += 1

    pub_stats = {}
    for pub, lens in by_pub.items():
        arr = np.array(lens, dtype=float)
        pub_stats[pub] = {
            "n_problems": int(len(lens)),
            "avg_standards_per_problem": float(arr.mean()) if len(arr) else 0.0,
            "pct_multi_standard": float((arr >= 2).mean() * 100.0) if len(arr) else 0.0,
        }

    total_domain_counts = Counter()
    for c in pub_domain_counts.values():
        total_domain_counts.update(c)
    top_domains = [d for d, _ in total_domain_counts.most_common(10)]
    domain_share = {}
    for pub, cnt in pub_domain_counts.items():
        total = sum(cnt.values()) or 1
        domain_share[pub] = {d: cnt[d] / total for d in cnt}

    plot_publisher_comparison(
        stats=pub_stats,
        top_domains=top_domains,
        domain_share=domain_share,
        out_path=fig_dir / "analysis3_publisher_comparison.png",
    )

    # Persist outputs.
    write_predictions_jsonl(
        out_dir / "preds_full_roberta_similarity.jsonl",
        {pid: row["predicted_standards"] for pid, row in full_pred_rows.items()},
    )
    write_json(
        out_dir / "preds_full_with_metadata.json",
        full_pred_rows,
    )
    write_json(
        out_dir / "predicted_prerequisites.json",
        prereq_predictions,
    )
    summary = {
        "config": vars(args),
        "n_problems_scored": len(full_pred_rows),
        "analysis1": {
            "heatmap_file": str(fig_dir / "analysis1_coverage_heatmap.png"),
            "n_zero_count_standards": len(rare_or_zero),
            "example_zero_count_standards": rare_or_zero[:25],
            "top_predicted_standards": standard_counts.most_common(25),
        },
        "analysis2": {
            "graph_file": str(fig_dir / "analysis2_prereq_graph.png"),
            "n_predicted_edges": len(prereq_edges),
            "top_edges": top_chain_edges[:25],
        },
        "analysis3": {
            "figure_file": str(fig_dir / "analysis3_publisher_comparison.png"),
            "domain_figure_file": str(fig_dir / "analysis3_publisher_comparison_domains.png"),
            "publisher_stats": pub_stats,
            "top_domains_overall": top_domains,
        },
    }
    write_json(out_dir / "summary.json", summary)

    md_lines = [
        "# Descriptive Analyses Summary",
        "",
        f"- Problems scored: {summary['n_problems_scored']}",
        f"- Coverage heatmap: `{summary['analysis1']['heatmap_file']}`",
        f"- Prerequisite graph: `{summary['analysis2']['graph_file']}`",
        f"- Publisher comparison: `{summary['analysis3']['figure_file']}`",
        "",
        "## Analysis 1 Highlights",
        f"- Zero-count standards under model predictions: {summary['analysis1']['n_zero_count_standards']}",
        f"- Top predicted standards: {summary['analysis1']['top_predicted_standards'][:10]}",
        "",
        "## Analysis 2 Highlights",
        f"- Predicted prerequisite edges: {summary['analysis2']['n_predicted_edges']}",
        f"- Top edges: {summary['analysis2']['top_edges'][:10]}",
        "",
        "## Analysis 3 Highlights",
        f"- Publisher stats: {summary['analysis3']['publisher_stats']}",
        f"- Top domains overall: {summary['analysis3']['top_domains_overall']}",
    ]
    (out_dir / "summary.md").write_text("\n".join(md_lines))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
