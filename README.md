# CS293 A3 - MathFish Standards Alignment

This repo contains:

- A2 annotation + baseline tooling (root scripts)
- A3 model/adaptation/descriptive-analysis work

## Where Krish's A3 Work Is

Primary A3 deliverables for Krish are organized under:

- [a3/krish/README.md](/Users/krishm/Desktop/School/cs293-a3/a3/krish/README.md)
- `a3/krish/scripts/`
- `a3/krish/results/`
- `a3/krish/figures/`

## Krish A3 Results Snapshot

Latest refresh: all Section 3 adaptation runs were executed on `publisher_full` with `train/val/test = 0.8/0.1/0.1`, and Section 4 was run on all `13,065` problems.

Performance Analysis (publisher-label evaluation):

- RoBERTa similarity baseline standard F1: **0.0675**
  - [roberta_similarity_publisher_k3.json](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/core/roberta_similarity_publisher_k3.json)
- RoBERTa fine-tuned standard F1: **0.0741**
  - [metrics.json](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/core/roberta_ft_publisher_e2_light/metrics.json)

Note: fine-tuned test size is `198` (post label-frequency filtering); baseline test size is `216`.

Adaptations completed:

- Hierarchical prediction:
  - Standard/Cluster/Domain F1: **0.0427 / 0.0859 / 0.1706** (`n=216`)
  - [metrics.json](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/adaptations/hierarchical/metrics.json)
- Retrieval-augmented few-shot prompting:
  - Standard/Cluster/Domain F1: **0.4924 / 0.5905 / 0.6940** (`n=216`)
  - [metrics.json](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/adaptations/rag_fewshot/metrics.json)
- Standard description conditioning:
  - Codes-only standard F1: **0.3443** (`n=216`)
  - Codes+descriptions standard F1: **0.4383** (`n=216`)
  - [metrics.json](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/adaptations/standard_conditioning/metrics.json)

Descriptive analyses (full MathFish, figures generated):

- Coverage heatmap: [analysis1_coverage_heatmap.png](/Users/krishm/Desktop/School/cs293-a3/a3/krish/figures/analysis1_coverage_heatmap.png)
- Prerequisite graph: [analysis2_prereq_graph.png](/Users/krishm/Desktop/School/cs293-a3/a3/krish/figures/analysis2_prereq_graph.png)
- Publisher comparison: [analysis3_publisher_comparison.png](/Users/krishm/Desktop/School/cs293-a3/a3/krish/figures/analysis3_publisher_comparison.png)
- Summary: [summary.json](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/descriptive/summary.json)
- Aggregate highlights:
  - Problems scored: **13,065**
  - Predicted zero-count standards: **51**
  - Predicted prerequisite edge types: **554**

Building On-only follow-up runs are also included at:
- [RESULTS_SUMMARY.md](/Users/krishm/Desktop/School/cs293-a3/a3/krish/results/building_on/RESULTS_SUMMARY.md)

## Setup

```bash
python3 -m pip install --user torch transformers datasets scikit-learn accelerate matplotlib networkx pandas
```

## Legacy Root Workflows

A2/annotation scripts at repo root are still available:

```bash
python3 annotate.py --name <your_name>
python3 irr.py --output results/irr.json
python3 benchmark.py tfidf --k 3 --output preds/tfidf_k3.jsonl
python3 llm_benchmark.py all --output-dir preds --results results/llm_results.json
```
