# A3 Results Summary (Krish)

## Section 2: Performance Analysis

Publisher-label evaluation (same metric family as A2 table):

| Model | Standard F1 | Cluster F1 | Domain F1 | Exact Match (Standard) |
|---|---:|---:|---:|---:|
| RoBERTa similarity baseline | 0.0675 | 0.1419 | 0.2554 | 0.0000 |
| RoBERTa fine-tuned (`e2_light`) | 0.0741 | 0.1811 | 0.3176 | 0.0000 |

Note: fine-tuned evaluation uses `198` test items after label-frequency filtering; baseline uses `216`.

Sources:

- `core/roberta_similarity_publisher_k3.json`
- `core/roberta_ft_publisher_e2_light/metrics.json`

## Section 3: Adaptations

### Adaptation 1 - Hierarchical Prediction (domain -> cluster -> standard)

Test metrics:

- Domain F1: `0.1706`
- Cluster F1: `0.0859`
- Standard F1: `0.0427`

Source: `adaptations/hierarchical/metrics.json`

### Adaptation 2 - Retrieval-Augmented Few-Shot Prompting

Test metrics (publisher split, `n=216`):

- Standard F1: `0.4924`
- Cluster F1: `0.5905`
- Domain F1: `0.6940`

Source: `adaptations/rag_fewshot/metrics.json`

### Adaptation 3 - Standard Description Conditioning

Test metrics (publisher split, `n=216`):

- Codes-only prompt standard F1: `0.3443`
- Codes+descriptions prompt standard F1: `0.4383`

Source: `adaptations/standard_conditioning/metrics.json`

## Section 4: Descriptive Analyses

Scored all `13,065` MathFish problems.

- Coverage heatmap: `../figures/analysis1_coverage_heatmap.png`
- Prerequisite graph: `../figures/analysis2_prereq_graph.png`
- Publisher comparison: `../figures/analysis3_publisher_comparison.png`
- Summary data: `descriptive/summary.json`

Notable aggregate stats:

- Predicted zero-count standards: `51`
- Predicted prerequisite edge types: `554`
- Publisher avg predicted standards/problem:
  - IM: `4.7929`
  - Fishtank: `4.6977`

## Building On Follow-up

For runs using only `relations = "Building On"`, see:

- `building_on/RESULTS_SUMMARY.md`
