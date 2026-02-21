#!/usr/bin/env python3
"""
Annotation server for CS293 MathFish standards tagging.

Usage:
    python3 annotate.py --name zane

Opens a browser-based annotation tool at http://localhost:8000
"""

import json
import os
import sys
import argparse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


class AnnotationHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_html()
        elif parsed.path == "/api/config":
            self._json_response({
                "annotator": self.server.annotator_name,
                "total_problems": len(self.server.problem_ids),
                "shared_count": len(self.server.shared_ids),
            })
        elif parsed.path == "/api/problems":
            problems = []
            for pid in self.server.problem_ids:
                if pid in self.server.problems:
                    p = self.server.problems[pid]
                    problems.append({
                        "id": p["id"],
                        "text": p["text"],
                        "source": p.get("source", ""),
                        "metadata": p.get("metadata", {}),
                        "elements": p.get("elements", {}),
                        "num_problems": p.get("num_problems", 1),
                        "is_shared": pid in self.server.shared_ids,
                    })
            self._json_response(problems)
        elif parsed.path == "/api/standards":
            self._json_response(self.server.standards_hierarchy)
        elif parsed.path == "/api/annotations":
            self._json_response(self.server.saved_annotations)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            annotation = {
                "problem_id": body["problem_id"],
                "annotator": self.server.annotator_name,
                "standards": body["standards"],
                "notes": body.get("notes", ""),
                "skipped": body.get("skipped", False),
            }

            self.server.saved_annotations[body["problem_id"]] = annotation

            ann_file = f"annotations/{self.server.annotator_name}_annotations.jsonl"
            with open(ann_file, "a") as f:
                f.write(json.dumps(annotation) + "\n")

            self._json_response({
                "ok": True,
                "saved": len(self.server.saved_annotations),
            })
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_html(self):
        html_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static", "index.html"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        with open(html_path, "rb") as f:
            self.wfile.write(f.read())

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


def load_standards_hierarchy(path):
    """Build nested grade > domain > cluster > standard hierarchy."""
    entries = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            entries[item["id"]] = item

    hs_cat_names = {
        "A": "Algebra",
        "F": "Functions",
        "G": "Geometry",
        "N": "Number & Quantity",
        "S": "Statistics & Probability",
    }

    grade_names = {
        "K": "Kindergarten",
    }
    for i in range(1, 9):
        grade_names[str(i)] = f"Grade {i}"

    hierarchy = {}

    for eid, entry in entries.items():
        if entry["level"] != "Domain":
            continue

        parts = eid.split(".")
        first = parts[0]

        if first in grade_names or first == "K":
            grade_key = first
            grade_name = grade_names.get(first, f"Grade {first}")
            sort_key = 0 if first == "K" else int(first)
        else:
            hs_prefix = eid.split("-")[0]
            grade_key = f"HS-{hs_prefix}"
            grade_name = f"HS: {hs_cat_names.get(hs_prefix, hs_prefix)}"
            hs_order = {"N": 0, "A": 1, "F": 2, "G": 3, "S": 4}
            sort_key = 100 + hs_order.get(hs_prefix, 9)

        if grade_key not in hierarchy:
            hierarchy[grade_key] = {
                "name": grade_name,
                "sort_key": sort_key,
                "domains": {},
            }

        domain = {
            "id": eid,
            "description": entry["description"],
            "clusters": {},
        }

        for cluster_id in entry.get("children", []):
            if cluster_id not in entries:
                continue
            cluster = entries[cluster_id]
            cluster_data = {
                "id": cluster_id,
                "description": cluster["description"],
                "cluster_type": cluster.get("cluster_type", ""),
                "standards": {},
            }

            for std_id in cluster.get("children", []):
                if std_id not in entries:
                    continue
                std = entries[std_id]
                std_data = {
                    "id": std_id,
                    "description": std["description"],
                    "sub_standards": {},
                }

                for sub_id in std.get("children", []):
                    if sub_id not in entries:
                        continue
                    sub = entries[sub_id]
                    std_data["sub_standards"][sub_id] = {
                        "id": sub_id,
                        "description": sub["description"],
                    }

                cluster_data["standards"][std_id] = std_data

            domain["clusters"][cluster_id] = cluster_data

        hierarchy[grade_key]["domains"][eid] = domain

    return hierarchy


def main():
    parser = argparse.ArgumentParser(
        description="MathFish annotation server"
    )
    parser.add_argument("--name", required=True, help="Your annotator name")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--standards", default="standards.jsonl")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("annotations/assignments.json"):
        print("Error: Run setup_annotations.py first to create assignments.")
        sys.exit(1)

    with open("annotations/assignments.json") as f:
        config = json.load(f)

    if args.name not in config["assignments"]:
        print(f"Error: '{args.name}' not found. Available: {config['annotators']}")
        sys.exit(1)

    with open("annotations/problems.json") as f:
        problems = json.load(f)

    ann_file = f"annotations/{args.name}_annotations.jsonl"
    saved = {}
    if os.path.exists(ann_file):
        with open(ann_file) as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    saved[a["problem_id"]] = a

    hierarchy = load_standards_hierarchy(args.standards)

    server = HTTPServer(("localhost", args.port), AnnotationHandler)
    server.annotator_name = args.name
    server.problems = problems
    server.problem_ids = config["assignments"][args.name]["all_ids"]
    server.shared_ids = set(config["shared_ids"])
    server.standards_hierarchy = hierarchy
    server.saved_annotations = saved

    done = len(saved)
    total = len(server.problem_ids)
    print(f"\n  MathFish Annotator")
    print(f"  Annotator: {args.name}")
    print(f"  Progress:  {done}/{total} problems")
    print(f"  Server:    http://localhost:{args.port}")
    print(f"\n  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(server.saved_annotations)}/{total} annotations saved.")
        server.server_close()


if __name__ == "__main__":
    main()
