# CS293 A3 - MathFish Standards Alignment

This repo contains the annotation tool, baseline scripts, and benchmarking utilities for Assignment 3.

## Quick Start

```bash
python3 annotate.py --name <your_name>
```

## Benchmarks

TF-IDF baseline (publisher labels):

```bash
python3 benchmark.py tfidf --k 1 --output preds/tfidf_k1.jsonl
python3 benchmark.py eval --preds preds/tfidf_k1.jsonl
```

LLM baselines (publisher labels):

```bash
python3 llm_benchmark.py all --output-dir preds --results results/llm_results.json
```

IRR (Krippendorff’s alpha):

```bash
python3 irr.py --output results/irr.json
```

## Results

All benchmark results are summarized in `RESULTS.md`.

## Status Checklist

- TF-IDF baseline: complete (k=1,3,5)
- LLM benchmarks: complete (Gemini, OpenAI, Anthropic)
- IRR: computed for `krish` + `zane`; rerun after more annotators finish
- After Sera + teacher_buddy add their annotations then run:
  ```bash
  python3 irr.py --output results/irr.json
  ```
  and update `RESULTS.md` with updated IRR values.
