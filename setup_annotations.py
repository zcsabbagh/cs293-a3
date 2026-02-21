#!/usr/bin/env python3
"""
Setup annotation assignments for CS293 MathFish standards tagging.

Run once to create assignments for all annotators:
    python3 setup_annotations.py --annotators zane alice bob teacher_buddy

Each annotator gets:
  - 20 shared problems (same for everyone, used for IRR)
  - 5 unique problems (different per person, for extra validation data)
"""

import json
import random
import argparse
import os


def load_eligible_problems(path, max_length=2000):
    """Load labeled, non-image, non-duplicate problems."""
    problems = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if (
                item.get("standards")
                and not item.get("has_image", False)
                and not item.get("is_duplicate", False)
                and len(item.get("text", "")) <= max_length
                and len(item.get("text", "")) >= 20
            ):
                problems.append({
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "standards": item["standards"],
                    "source": item.get("source", ""),
                    "elements": item.get("elements", {}),
                    "num_problems": item.get("num_problems", 1),
                })
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Setup annotation assignments for CS293 MathFish"
    )
    parser.add_argument(
        "--annotators", nargs="+", required=True,
        help="Names of annotators (e.g., zane alice bob teacher)"
    )
    parser.add_argument(
        "--overlap", type=int, default=20,
        help="Shared problems for IRR (default: 20)"
    )
    parser.add_argument(
        "--unique", type=int, default=5,
        help="Unique problems per annotator (default: 5)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data", default="mathfish_train.jsonl", help="Path to training data"
    )
    args = parser.parse_args()

    if os.path.exists("annotations/assignments.json"):
        resp = input("annotations/assignments.json already exists. Overwrite? [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    print("Loading eligible problems...")
    problems = load_eligible_problems(args.data)
    print(f"  Found {len(problems)} eligible problems (labeled, no images, not duplicate)")

    random.seed(args.seed)
    random.shuffle(problems)

    total_needed = args.overlap + len(args.annotators) * args.unique
    if len(problems) < total_needed:
        print(f"Warning: only {len(problems)} problems, need {total_needed}")

    shared = problems[: args.overlap]
    remaining = problems[args.overlap :]

    assignments = {}
    for i, name in enumerate(args.annotators):
        start = i * args.unique
        unique = remaining[start : start + args.unique]
        assignments[name] = {
            "shared_ids": [p["id"] for p in shared],
            "unique_ids": [p["id"] for p in unique],
            "all_ids": [p["id"] for p in shared] + [p["id"] for p in unique],
        }

    os.makedirs("annotations", exist_ok=True)

    all_problem_ids = set()
    for a in assignments.values():
        all_problem_ids.update(a["all_ids"])

    assigned_problems = {p["id"]: p for p in problems if p["id"] in all_problem_ids}

    with open("annotations/problems.json", "w") as f:
        json.dump(assigned_problems, f, indent=2)

    with open("annotations/assignments.json", "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "overlap_count": args.overlap,
                "unique_count": args.unique,
                "annotators": args.annotators,
                "assignments": assignments,
                "shared_ids": [p["id"] for p in shared],
            },
            f,
            indent=2,
        )

    print(f"\nCreated assignments for {len(args.annotators)} annotators:")
    print(f"  {args.overlap} shared problems (for IRR)")
    print(f"  {args.unique} unique problems per annotator")
    print(f"  {len(all_problem_ids)} total unique problems")
    print(f"\nSaved to annotations/")
    for name in args.annotators:
        print(f"  {name}: {len(assignments[name]['all_ids'])} problems")
    print(f"\nEach annotator runs:")
    print(f"  python3 annotate.py --name <their_name>")


if __name__ == "__main__":
    main()
