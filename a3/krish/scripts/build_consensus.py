#!/usr/bin/env python3
"""
Build majority-vote consensus draft for shared annotation problems.

Usage:
  python3 build_consensus.py \
      --assignments annotations/assignments.json \
      --annotations-dir annotations \
      --output-jsonl annotations/shared_consensus_draft.jsonl \
      --output-md annotations/shared_adjudication_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.utils import load_all_annotation_files_latest, majority_consensus_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared-problem consensus draft")
    parser.add_argument("--assignments", default="annotations/assignments.json")
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument("--required-votes", type=int, default=0, help="0 = majority")
    parser.add_argument(
        "--output-jsonl",
        default="annotations/shared_consensus_draft.jsonl",
    )
    parser.add_argument(
        "--output-md",
        default="annotations/shared_adjudication_report.md",
    )
    args = parser.parse_args()

    with open(args.assignments) as f:
        config = json.load(f)
    shared_ids = config.get("shared_ids", [])
    if not shared_ids:
        raise RuntimeError("No shared_ids found in assignments file.")

    ann = load_all_annotation_files_latest(args.annotations_dir)
    if not ann:
        raise RuntimeError("No annotation files found.")

    required_votes = args.required_votes if args.required_votes > 0 else None
    consensus = majority_consensus_labels(
        annotator_rows=ann,
        problem_ids=shared_ids,
        required_votes=required_votes,
    )

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for pid in shared_ids:
            row = consensus[pid]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    needs = [row for row in consensus.values() if row["needs_adjudication"]]
    clean = len(shared_ids) - len(needs)
    annotators = sorted(ann.keys())
    rv = next(iter(consensus.values()))["required_votes"] if consensus else 0

    lines = [
        "# Shared Annotation Adjudication Report",
        "",
        f"- Shared problems: {len(shared_ids)}",
        f"- Annotators found: {len(annotators)} ({', '.join(annotators)})",
        f"- Vote threshold for consensus: {rv}",
        f"- Auto-resolved by majority: {clean}",
        f"- Needs adjudication discussion: {len(needs)}",
        "",
        "## Items Requiring Adjudication",
        "",
    ]
    if not needs:
        lines.append("None.")
    else:
        for row in needs:
            lines.append(f"### {row['problem_id']}")
            lines.append(f"- Consensus labels: {row['consensus_labels']}")
            lines.append(f"- Vote counts: {row['vote_counts']}")
            lines.append("")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))

    print(f"Wrote consensus draft: {out_jsonl}")
    print(f"Wrote adjudication report: {out_md}")
    print(f"Needs adjudication: {len(needs)} / {len(shared_ids)}")


if __name__ == "__main__":
    main()
