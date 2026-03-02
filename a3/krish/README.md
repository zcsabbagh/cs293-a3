# A3 - Krish Workstream

This folder contains Krish's A3 implementation artifacts: model training, adaptations, annotation-prep utilities, and descriptive analyses.

## Folder Layout

- `scripts/`: runnable A3 scripts for Krish's section
- `results/core/`: baseline + fine-tuned RoBERTa results
- `results/adaptations/`: three adaptation experiments
- `results/descriptive/`: scale analyses over full MathFish
- `results/annotation_prep/`: adjudication + active learning outputs
- `figures/`: paper-ready figures for Section 4

## Completed Deliverables (Krish Scope)

### Direction Status

- Section 3 (Performance Enhancement): done
  - Adaptation 1 (hierarchical): implemented + evaluated
  - Adaptation 2 (RAG few-shot): implemented + evaluated
  - Adaptation 3 (standard-description conditioning): implemented + evaluated
- Section 4 (Descriptive Analyses): done
  - Analysis 1 coverage heatmap: generated
  - Analysis 2 prerequisite graph: generated
  - Analysis 3 publisher comparison: generated

### 2) Performance Analysis

- RoBERTa similarity baseline (publisher ground truth)
  - `results/core/roberta_similarity_publisher_k3.json`
- RoBERTa fine-tuning pipeline + run
  - `results/core/roberta_ft_publisher_e2_light/metrics.json`
  - `results/core/roberta_ft_publisher_e2_light/error_examples_test.jsonl`

Key standard-level F1 (publisher-label split):

- Baseline: `0.0675`
- Fine-tuned: `0.0741`

Note: the fine-tuned run evaluates `198` test items after label-frequency filtering; baseline evaluates `216`.

Compact table summary:
- `results/RESULTS_SUMMARY.md`

### 3) Performance Enhancement (3 Adaptations)

- Adaptation 1: Hierarchical prediction (`domain -> cluster -> standard`)
  - `results/adaptations/hierarchical/metrics.json`
- Adaptation 2: Retrieval-augmented few-shot prompting
  - `results/adaptations/rag_fewshot/metrics.json`
- Adaptation 3: Standard description conditioning (codes-only vs codes+descriptions)
  - `results/adaptations/standard_conditioning/metrics.json`

Adaptation summary (standard-level F1):

- Hierarchical: `0.0427` (test, publisher split)
- RAG few-shot: `0.4924` (test, publisher split, n=216)
- Standard conditioning:
  - Codes only: `0.3443` (test, publisher split, n=216)
  - Codes + descriptions: `0.4383` (test, publisher split, n=216)

### Building On Runs (Requested Follow-up)

I also ran the same Section 2/3 pipeline using only `Building On` labels.

- Outputs:
  - `results/building_on/core/`
  - `results/building_on/adaptations/`
  - `results/building_on/RESULTS_SUMMARY.md`
- Key Building On standard F1:
  - Baseline: `0.0167`
  - Fine-tuned: `0.0403`
  - Hierarchical: `0.0123`
  - RAG few-shot: `0.0809`
  - Conditioning (codes -> codes+desc): `0.0568` -> `0.0658`

### 4) Descriptive Analyses (3 at Scale)

Generated with:
- `scripts/descriptive_analyses.py`

Outputs:

- Analysis 1 (Standards coverage map):
  - `figures/analysis1_coverage_heatmap.png`
- Analysis 2 (Prerequisite chains graph):
  - `figures/analysis2_prereq_graph.png`
- Analysis 3 (Publisher comparison):
  - `figures/analysis3_publisher_comparison.png`
  - `figures/analysis3_publisher_comparison_domains.png`
- Full summary:
  - `results/descriptive/summary.json`
  - `results/descriptive/summary.md`

Highlights from the latest full run:
- Problems scored: `13,065`
- Predicted zero-count standards: `51`
- Predicted prerequisite edge types: `554`
- Avg predicted standards/problem:
  - IM: `4.7929`
  - Fishtank: `4.6977`

### Annotation Expansion/Adjudication Prep

- Active-learning candidate pool:
  - `results/annotation_prep/active_learning_candidates.jsonl` (150 items)
- Consensus draft:
  - `results/annotation_prep/shared_consensus_draft.jsonl`
- Adjudication report:
  - `results/annotation_prep/shared_adjudication_report.md`

## Quick Re-run Commands

```bash
python3 a3/krish/scripts/roberta_similarity.py --dataset publisher_full --top-k 3 --grade-filter
python3 a3/krish/scripts/train_roberta_ft.py --dataset publisher_full --output-dir a3/krish/results/core/roberta_ft_with_model --epochs 2 --save-model
python3 a3/krish/scripts/select_active_learning.py --run-dir a3/krish/results/core/roberta_ft_with_model --top-n 150
python3 a3/krish/scripts/build_consensus.py
python3 a3/krish/scripts/adaptation_hierarchical.py
python3 a3/krish/scripts/adaptation_rag_fewshot.py --dataset publisher_full --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0
python3 a3/krish/scripts/adaptation_standard_conditioning.py --dataset publisher_full --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0
python3 a3/krish/scripts/descriptive_analyses.py --data-path mathfish_train.jsonl
```

Building On reruns:

```bash
python3 a3/krish/scripts/roberta_similarity.py --dataset publisher_full --relations "Building On" --top-k 3 --grade-filter --out-preds a3/krish/results/building_on/core/roberta_similarity_building_on_k3.jsonl --out-metrics a3/krish/results/building_on/core/roberta_similarity_building_on_k3.json
python3 a3/krish/scripts/train_roberta_ft.py --dataset publisher_full --relations "Building On" --epochs 2 --output-dir a3/krish/results/building_on/core/roberta_ft_building_on_e2_light
python3 a3/krish/scripts/adaptation_hierarchical.py --relations "Building On" --output-dir a3/krish/results/building_on/adaptations/hierarchical
python3 a3/krish/scripts/adaptation_rag_fewshot.py --dataset publisher_full --relations "Building On" --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0 --output-dir a3/krish/results/building_on/adaptations/rag_fewshot
python3 a3/krish/scripts/adaptation_standard_conditioning.py --dataset publisher_full --relations "Building On" --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0 --output-dir a3/krish/results/building_on/adaptations/standard_conditioning
```
