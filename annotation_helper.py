"""
annotation_helper.py - Common Core Math Standards Annotation Lookup Tool

Usage:
  Interactive mode:  python annotation_helper.py
  Grade tree mode:   python annotation_helper.py --grade 4
  Search mode:       python annotation_helper.py --search "fraction"
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

STANDARDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standards.jsonl")

# Maps HS category IDs to friendly display names (and vice-versa for lookup)
HS_CATEGORY_DISPLAY = {
    "A": "Algebra",
    "N": "Number & Quantity",
    "F": "Functions",
    "G": "Geometry",
    "S": "Statistics & Probability",
}

HS_CATEGORY_BY_NAME = {v.lower(): k for k, v in HS_CATEGORY_DISPLAY.items()}
# Extra aliases the user might type
HS_CATEGORY_BY_NAME["statistics"] = "S"
HS_CATEGORY_BY_NAME["probability"] = "S"
HS_CATEGORY_BY_NAME["statistics & probability"] = "S"
HS_CATEGORY_BY_NAME["number and quantity"] = "N"
HS_CATEGORY_BY_NAME["number & quantity"] = "N"

CLUSTER_TYPE_LABELS = {
    "major cluster": "[Major]",
    "supporting cluster": "[Supporting]",
    "additional cluster": "[Additional]",
    "none": "",
    "": "",
}


def load_standards(filepath: str) -> dict[str, dict]:
    """Load all standards from a JSONL file into a dict keyed by standard ID."""
    standards: dict[str, dict] = {}
    try:
        with open(filepath, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    standards[entry["id"]] = entry
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"Warning: could not parse line {line_num}: {exc}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: standards file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    return standards


# ---------------------------------------------------------------------------
# Tree-printing helpers
# ---------------------------------------------------------------------------

INDENT = "    "


def cluster_label(entry: dict) -> str:
    """Return a short cluster-type label string, or empty string."""
    return CLUSTER_TYPE_LABELS.get(entry.get("cluster_type", ""), "")


def wrap_description(text: str, prefix: str, width: int = 100) -> str:
    """Wrap a long description so it aligns under the first character after the prefix."""
    subsequent = " " * len(prefix)
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=subsequent)


def print_standard_entry(entry: dict, indent_level: int, standards: dict[str, dict]) -> None:
    """Recursively print a standard and all its children."""
    prefix = INDENT * indent_level
    sid = entry["id"]
    desc = entry["description"]
    level = entry.get("level", "")
    ct = cluster_label(entry)
    ct_str = f"  {ct}" if ct else ""

    if level in ("Standard", "Sub-standard"):
        line_prefix = f"{prefix}{sid}: "
        print(wrap_description(desc, line_prefix))
    elif level == "Cluster":
        print(f"{prefix}--- Cluster {sid}{ct_str} ---")
        line_prefix = f"{prefix}    "
        print(wrap_description(desc, line_prefix))
    elif level == "Domain":
        print(f"\n{prefix}== Domain: {sid} - {desc} ==")
    elif level == "HS Category":
        print(f"\n{prefix}=== HS Category: {sid} - {desc} ===")
    elif level == "Grade":
        print(f"\n{'=' * 60}")
        print(f"  {desc} ({sid})")
        print(f"{'=' * 60}")

    for child_id in entry.get("children", []):
        if child_id in standards:
            print_standard_entry(standards[child_id], indent_level + 1, standards)


def print_grade_tree(grade_id: str, standards: dict[str, dict]) -> None:
    """Print the full hierarchy for a given grade ID."""
    entry = standards.get(grade_id)
    if entry is None:
        print(f"Error: grade '{grade_id}' not found in standards.", file=sys.stderr)
        sys.exit(1)
    print_standard_entry(entry, 0, standards)
    print()


# ---------------------------------------------------------------------------
# Search mode
# ---------------------------------------------------------------------------

SEARCHABLE_LEVELS = {"Standard", "Sub-standard", "Cluster", "Domain"}


def search_standards(keyword: str, standards: dict[str, dict]) -> None:
    """Search standard descriptions for a keyword (case-insensitive) and print matches."""
    keyword_lower = keyword.lower()
    matches = [
        entry for entry in standards.values()
        if entry.get("level") in SEARCHABLE_LEVELS
        and keyword_lower in entry.get("description", "").lower()
    ]

    if not matches:
        print(f"No standards found matching '{keyword}'.")
        return

    print(f"\nFound {len(matches)} result(s) for '{keyword}':\n")
    print("-" * 70)

    # Group by grade / HS category for readability
    grouped: dict[str, list[dict]] = {}
    for entry in matches:
        grade_key = _grade_key_for_entry(entry, standards)
        grouped.setdefault(grade_key, []).append(entry)

    # Sort grade keys in logical order: K, 1-8, HS
    def grade_sort_key(k: str) -> tuple:
        if k == "K":
            return (0,)
        if k.isdigit():
            return (int(k),)
        return (99, k)

    for grade_key in sorted(grouped.keys(), key=grade_sort_key):
        grade_entry = standards.get(grade_key)
        grade_label = grade_entry["description"] if grade_entry else grade_key
        print(f"\n[{grade_label}]")
        for entry in sorted(grouped[grade_key], key=lambda e: e["id"]):
            level = entry.get("level", "")
            indent = "  " if level in ("Cluster", "Domain") else "    "
            ct = cluster_label(entry)
            ct_str = f"  {ct}" if ct else ""
            line_prefix = f"{indent}{entry['id']}: "
            print(wrap_description(entry["description"], line_prefix) + ct_str)

    print()


def _grade_key_for_entry(entry: dict, standards: dict[str, dict]) -> str:
    """Walk up the parent chain to find the top-level grade ID."""
    current = entry
    while True:
        parent_id = current.get("parent", "")
        if not parent_id:
            # This node itself is the top
            return current["id"]
        parent = standards.get(parent_id)
        if parent is None:
            return current["id"]
        if parent.get("level") in ("Grade", "HS Category"):
            # For HS, return the HS category parent, then walk one more to get "HS" grade
            hs_parent = standards.get(parent.get("parent", ""))
            if hs_parent and hs_parent.get("level") == "Grade":
                return hs_parent["id"]
            return parent["id"]
        current = parent


# ---------------------------------------------------------------------------
# Interactive mode helpers
# ---------------------------------------------------------------------------

def prompt(message: str) -> str:
    """Print a prompt and return stripped input. Handles EOF gracefully."""
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        sys.exit(0)


def numbered_menu(items: list[tuple[str, str]], prompt_text: str) -> int:
    """
    Display a numbered menu and return the 0-based index chosen by the user.

    items is a list of (id, description) tuples.
    Returns the chosen index, or -1 if the user wants to quit.
    """
    print()
    for i, (sid, desc) in enumerate(items, start=1):
        line_prefix = f"  {i:2}. {sid}: "
        print(wrap_description(desc, line_prefix))
    print()

    while True:
        choice = prompt(prompt_text)
        if choice.lower() in ("q", "quit", "exit"):
            return -1
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return idx
        print(f"Please enter a number between 1 and {len(items)}, or 'q' to quit.")


def get_children_of(entry: dict, standards: dict[str, dict]) -> list[dict]:
    """Return child entries in the order listed in the parent's children field."""
    children = []
    for child_id in entry.get("children", []):
        child = standards.get(child_id)
        if child:
            children.append(child)
    return children


def display_cluster_and_standards(cluster: dict, standards: dict[str, dict]) -> None:
    """Print a cluster header and all its standards (and sub-standards)."""
    ct = cluster_label(cluster)
    ct_str = f"  {ct}" if ct else ""
    print(f"\n  --- Cluster {cluster['id']}{ct_str} ---")
    print(f"      {cluster['description']}")

    def print_std(entry: dict, depth: int) -> None:
        prefix = "  " * (depth + 2)
        sid = entry["id"]
        desc = entry["description"]
        line_prefix = f"{prefix}{sid}: "
        print(wrap_description(desc, line_prefix))
        for child_id in entry.get("children", []):
            child = standards.get(child_id)
            if child:
                print_std(child, depth + 1)

    for child in get_children_of(cluster, standards):
        print_std(child, 0)


def run_interactive(standards: dict[str, dict]) -> None:
    """Run the interactive annotation helper."""
    print()
    print("=" * 60)
    print("  Common Core Math Standards Annotation Helper")
    print("=" * 60)
    print("  Type 'q' at any prompt to quit.\n")

    # ---- Step 1: Choose grade ----
    grade_options = [
        ("K",  "Kindergarten"),
        ("1",  "Grade 1"),
        ("2",  "Grade 2"),
        ("3",  "Grade 3"),
        ("4",  "Grade 4"),
        ("5",  "Grade 5"),
        ("6",  "Grade 6"),
        ("7",  "Grade 7"),
        ("8",  "Grade 8"),
        ("HS", "High School"),
    ]
    idx = numbered_menu(grade_options, "Select a grade level (enter number): ")
    if idx == -1:
        return
    grade_id, grade_label = grade_options[idx]

    # ---- Step 2: If HS, choose category ----
    if grade_id == "HS":
        hs_entry = standards.get("HS")
        if hs_entry is None:
            print("Error: HS entry not found in standards file.", file=sys.stderr)
            return

        hs_categories = get_children_of(hs_entry, standards)
        # Exclude "Modeling" (id="M") since it has no children
        hs_categories = [c for c in hs_categories if c.get("children")]
        cat_options = [(c["id"], c["description"]) for c in hs_categories]

        print(f"\n-- High School categories --")
        cat_idx = numbered_menu(cat_options, "Select a category (enter number): ")
        if cat_idx == -1:
            return
        category_entry = hs_categories[cat_idx]
        domains = get_children_of(category_entry, standards)
        parent_label = f"{grade_label} > {category_entry['description']}"
    else:
        grade_entry = standards.get(grade_id)
        if grade_entry is None:
            print(f"Error: grade '{grade_id}' not found.", file=sys.stderr)
            return
        domains = get_children_of(grade_entry, standards)
        parent_label = grade_label

    # ---- Step 3: Choose domain ----
    if not domains:
        print("No domains found for the selected grade/category.")
        return

    domain_options = [(d["id"], d["description"]) for d in domains]
    print(f"\n-- Domains for {parent_label} --")
    dom_idx = numbered_menu(domain_options, "Select a domain (enter number): ")
    if dom_idx == -1:
        return
    domain_entry = domains[dom_idx]

    # ---- Step 4: Show all clusters and standards ----
    clusters = get_children_of(domain_entry, standards)
    if not clusters:
        print(f"\nNo clusters found under domain {domain_entry['id']}.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Domain: {domain_entry['id']} - {domain_entry['description']}")
    print(f"{'=' * 60}")

    for cluster in clusters:
        display_cluster_and_standards(cluster, standards)

    # ---- Step 5: Ask the annotator to select standard(s) ----
    print()
    print("-" * 60)
    print("Enter the standard code(s) that apply to your item.")
    print("You can enter multiple codes separated by commas.")
    print("Example:  4.NF.B.3,  4.NF.B.3a")
    print("-" * 60)

    # Build a flat lookup of valid IDs under this domain for validation
    valid_ids: set[str] = set()

    def collect_ids(entry: dict) -> None:
        valid_ids.add(entry["id"])
        for child_id in entry.get("children", []):
            child = standards.get(child_id)
            if child:
                collect_ids(child)

    for cluster in clusters:
        collect_ids(cluster)

    while True:
        raw = prompt("\nEnter standard code(s) (or 'q' to quit): ")
        if raw.lower() in ("q", "quit", "exit"):
            print("Exiting without selection.")
            return
        if not raw:
            continue

        codes = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in codes if c not in valid_ids]

        if unknown:
            print(f"  Warning: these codes were not found under {domain_entry['id']}:")
            for u in unknown:
                print(f"    - {u}")
            confirm = prompt("  Use them anyway? (yes/no): ")
            if confirm.lower() not in ("yes", "y"):
                continue

        print()
        print("=" * 60)
        print("  Selected standard(s):")
        print("=" * 60)
        for code in codes:
            entry = standards.get(code)
            if entry:
                # Normalize any embedded newlines in the description before wrapping
                desc = entry["description"].replace("\n", " ")
                line_prefix = f"  {code}: "
                print(wrap_description(desc, line_prefix))
            else:
                print(f"  {code}: (custom / not in database)")
        print("=" * 60)
        print()
        return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotation_helper",
        description=(
            "Common Core Math Standards Annotation Helper.\n\n"
            "Run without arguments for interactive mode.\n"
            "Use --grade to print a full grade tree.\n"
            "Use --search to find standards by keyword."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grade",
        metavar="GRADE",
        help=(
            "Print the full hierarchy for a grade. "
            "Values: K, 1-8, HS, or an HS category prefix "
            "(A=Algebra, N=Number & Quantity, F=Functions, G=Geometry, S=Statistics & Probability)."
        ),
    )
    parser.add_argument(
        "--search",
        metavar="KEYWORD",
        help="Search all standard descriptions for a keyword.",
    )
    parser.add_argument(
        "--standards-file",
        metavar="PATH",
        default=STANDARDS_FILE,
        help=f"Path to the standards JSONL file (default: {STANDARDS_FILE}).",
    )
    return parser


def resolve_grade_arg(grade_arg: str, standards: dict[str, dict]) -> Optional[str]:
    """
    Resolve the --grade argument to a standards ID.
    Accepts: K, 1-8, HS, or HS category IDs/names.
    """
    normalized = grade_arg.strip().upper()

    # Direct match (K, 1-8, HS, A, N, F, G, S)
    if normalized in standards:
        return normalized

    # Try lowercase name lookup for HS categories
    lower = grade_arg.strip().lower()
    if lower in HS_CATEGORY_BY_NAME:
        return HS_CATEGORY_BY_NAME[lower]

    return None


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    standards = load_standards(args.standards_file)

    if args.search:
        search_standards(args.search, standards)
        return

    if args.grade:
        grade_id = resolve_grade_arg(args.grade, standards)
        if grade_id is None:
            print(
                f"Error: '{args.grade}' is not a recognized grade or HS category.\n"
                "Valid grades: K, 1, 2, 3, 4, 5, 6, 7, 8, HS\n"
                "Valid HS categories: A (Algebra), N (Number & Quantity), "
                "F (Functions), G (Geometry), S (Statistics & Probability)",
                file=sys.stderr,
            )
            sys.exit(1)
        print_grade_tree(grade_id, standards)
        return

    # No flags: run interactive mode
    run_interactive(standards)


if __name__ == "__main__":
    main()
