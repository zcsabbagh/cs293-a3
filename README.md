# CS293 A3 - MathFish Standards and Prerequisites

This repository is organized around two main docs:

- `README.md` (project status, structure, rerun instructions)
- `RESULTS.md` (full detailed metrics and outputs)

## Project Scope (What Is Done)

### Section 2: Performance Analysis

Completed for:
- `Addressing,Alignment` labels
- `Building On` labels (follow-up run)

Implemented and executed:
- RoBERTa similarity baseline
- RoBERTa fine-tuning pipeline

### Section 3: Performance Enhancement

Completed for both relation settings above:
- Adaptation 1: Hierarchical prediction (`domain -> cluster -> standard`)
- Adaptation 2: Retrieval-augmented few-shot prompting
- Adaptation 3: Standard-description conditioning (codes-only vs codes+descriptions)

### Section 4: Descriptive Analyses at Scale

Completed over full MathFish (`13,065` problems):
- Standards coverage map
- Prerequisite chain graph
- Publisher comparison

## Result Snapshot

Detailed precision/recall/F1/exact tables are in `RESULTS.md`.

### Addressing/Alignment (Standard F1)

- RoBERTa similarity baseline: `0.0675`
- RoBERTa FT (`e2_light`): `0.0741`
- Hierarchical: `0.0427`
- RAG few-shot: `0.4924`
- Conditioning (codes only): `0.3443`
- Conditioning (codes+descriptions): `0.4383`

### Building On (Standard F1)

- RoBERTa similarity baseline: `0.0167`
- RoBERTa FT (`e2_light`): `0.0403`
- Hierarchical: `0.0123`
- RAG few-shot: `0.0809`
- Conditioning (codes only): `0.0568`
- Conditioning (codes+descriptions): `0.0658`

### Section 4 Highlights

- Problems scored: `13,065`
- Predicted zero-count standards: `51`
- Predicted prerequisite edge types: `554`
- Avg predicted standards/problem:
  - Illustrative Mathematics: `4.7929`
  - Fishtank Learning: `4.6977`

## Main Artifacts

### Scripts

- `a3/krish/scripts/roberta_similarity.py`
- `a3/krish/scripts/train_roberta_ft.py`
- `a3/krish/scripts/adaptation_hierarchical.py`
- `a3/krish/scripts/adaptation_rag_fewshot.py`
- `a3/krish/scripts/adaptation_standard_conditioning.py`
- `a3/krish/scripts/descriptive_analyses.py`

### Addressing/Alignment Results

- `a3/krish/results/core/`
- `a3/krish/results/adaptations/`
- `a3/krish/results/descriptive/`

### Building On Results

- `a3/krish/results/building_on/core/`
- `a3/krish/results/building_on/adaptations/`

### Figures

- `a3/krish/figures/analysis1_coverage_heatmap.png`
- `a3/krish/figures/analysis2_prereq_graph.png`
- `a3/krish/figures/analysis3_publisher_comparison.png`
- `a3/krish/figures/analysis3_publisher_comparison_domains.png`

## Reproduction

### Addressing/Alignment (`relations="Addressing,Alignment"`)

```bash
python3 a3/krish/scripts/roberta_similarity.py --dataset publisher_full --relations "Addressing,Alignment" --top-k 3 --grade-filter --out-preds a3/krish/results/core/roberta_similarity_publisher_k3.jsonl --out-metrics a3/krish/results/core/roberta_similarity_publisher_k3.json
python3 a3/krish/scripts/train_roberta_ft.py --dataset publisher_full --relations "Addressing,Alignment" --epochs 2 --output-dir a3/krish/results/core/roberta_ft_publisher_e2_light
python3 a3/krish/scripts/adaptation_hierarchical.py --relations "Addressing,Alignment" --output-dir a3/krish/results/adaptations/hierarchical
python3 a3/krish/scripts/adaptation_rag_fewshot.py --dataset publisher_full --relations "Addressing,Alignment" --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0 --output-dir a3/krish/results/adaptations/rag_fewshot
python3 a3/krish/scripts/adaptation_standard_conditioning.py --dataset publisher_full --relations "Addressing,Alignment" --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0 --output-dir a3/krish/results/adaptations/standard_conditioning
```

### Building On (`relations="Building On"`)

```bash
python3 a3/krish/scripts/roberta_similarity.py --dataset publisher_full --relations "Building On" --top-k 3 --grade-filter --out-preds a3/krish/results/building_on/core/roberta_similarity_building_on_k3.jsonl --out-metrics a3/krish/results/building_on/core/roberta_similarity_building_on_k3.json
python3 a3/krish/scripts/train_roberta_ft.py --dataset publisher_full --relations "Building On" --epochs 2 --output-dir a3/krish/results/building_on/core/roberta_ft_building_on_e2_light
python3 a3/krish/scripts/adaptation_hierarchical.py --relations "Building On" --output-dir a3/krish/results/building_on/adaptations/hierarchical
python3 a3/krish/scripts/adaptation_rag_fewshot.py --dataset publisher_full --relations "Building On" --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0 --output-dir a3/krish/results/building_on/adaptations/rag_fewshot
python3 a3/krish/scripts/adaptation_standard_conditioning.py --dataset publisher_full --relations "Building On" --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0 --output-dir a3/krish/results/building_on/adaptations/standard_conditioning
```

### Descriptive Analyses

```bash
python3 a3/krish/scripts/descriptive_analyses.py --data-path mathfish_train.jsonl
```

## Environment Notes

- LLM adaptations require API keys in `.env` (`OPENAI_API_KEY`, optionally other providers).
- Large source dataset `mathfish_train.jsonl` is intentionally gitignored.
