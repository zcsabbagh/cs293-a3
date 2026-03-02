# A3 Final Results (Detailed)

This file contains the consolidated A3 metrics for both target relations:

- `Addressing,Alignment` (standard alignment objective)
- `Building On` (prerequisite objective)

All metrics are from files in `a3/krish/results/`.

## Evaluation Setup

- Dataset: `publisher_full` from `mathfish_train.jsonl`
- Split for most runs: `train/val/test = 0.8/0.1/0.1`
- Level metrics reported at: `standard`, `cluster`, `domain`
- Metrics: precision, recall, F1, exact match

## Addressing/Alignment Results

Relations: `Addressing,Alignment`

Test sizes:
- Baseline / Hierarchical / RAG / Conditioning: `n=216`
- Fine-tuned RoBERTa: `n=198` (post label-frequency filtering)

| Model | n | Std P | Std R | Std F1 | Std Exact | Cluster P | Cluster R | Cluster F1 | Cluster Exact | Domain P | Domain R | Domain F1 | Domain Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa similarity baseline | 216 | 0.0494 | 0.1067 | 0.0675 | 0.0000 | 0.1039 | 0.2234 | 0.1419 | 0.0046 | 0.1877 | 0.3992 | 0.2554 | 0.0278 |
| RoBERTa fine-tuned (`e2_light`) | 198 | 0.0539 | 0.1185 | 0.0741 | 0.0000 | 0.1336 | 0.2811 | 0.1811 | 0.0101 | 0.2372 | 0.4805 | 0.3176 | 0.0455 |
| Adaptation 1: Hierarchical | 216 | 0.0331 | 0.0600 | 0.0427 | 0.0000 | 0.0972 | 0.0769 | 0.0859 | 0.0602 | 0.1852 | 0.1581 | 0.1706 | 0.1250 |
| Adaptation 2: RAG few-shot | 216 | 0.4192 | 0.5967 | 0.4924 | 0.3009 | 0.5324 | 0.6630 | 0.5905 | 0.4491 | 0.6572 | 0.7352 | 0.6940 | 0.5880 |
| Adaptation 3a: Conditioning (codes only) | 216 | 0.2493 | 0.5567 | 0.3443 | 0.1574 | 0.3886 | 0.6520 | 0.4870 | 0.3565 | 0.5697 | 0.7431 | 0.6449 | 0.5185 |
| Adaptation 3b: Conditioning (codes+descriptions) | 216 | 0.3421 | 0.6100 | 0.4383 | 0.2037 | 0.4672 | 0.7033 | 0.5614 | 0.3935 | 0.6138 | 0.8103 | 0.6985 | 0.5741 |

Additional adaptation details:
- Hierarchical best val decode: `k_domain=1, k_cluster=1, k_standard=3`
- Fine-tuned RoBERTa best decode: `threshold=0.2` on Building On, `topk=3` on Addressing/Alignment (from run-specific metrics JSON)

## Building On Results

Relations: `Building On`

Test sizes:
- Baseline / Hierarchical / RAG / Conditioning: `n=75`
- Fine-tuned RoBERTa: `n=69` (post label-frequency filtering)

| Model | n | Std P | Std R | Std F1 | Std Exact | Cluster P | Cluster R | Cluster F1 | Cluster Exact | Domain P | Domain R | Domain F1 | Domain Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa similarity baseline | 75 | 0.0133 | 0.0222 | 0.0167 | 0.0000 | 0.0300 | 0.0556 | 0.0390 | 0.0000 | 0.0601 | 0.1146 | 0.0789 | 0.0133 |
| RoBERTa fine-tuned (`e2_light`) | 69 | 0.0221 | 0.2314 | 0.0403 | 0.0000 | 0.0329 | 0.2680 | 0.0586 | 0.0000 | 0.0532 | 0.2989 | 0.0903 | 0.0000 |
| Adaptation 1: Hierarchical | 75 | 0.0105 | 0.0148 | 0.0123 | 0.0000 | 0.0267 | 0.0185 | 0.0219 | 0.0133 | 0.0400 | 0.0312 | 0.0351 | 0.0267 |
| Adaptation 2: RAG few-shot | 75 | 0.0803 | 0.0815 | 0.0809 | 0.0267 | 0.1296 | 0.1296 | 0.1296 | 0.0667 | 0.1818 | 0.1875 | 0.1846 | 0.0800 |
| Adaptation 3a: Conditioning (codes only) | 75 | 0.0495 | 0.0667 | 0.0568 | 0.0133 | 0.1022 | 0.1296 | 0.1143 | 0.0533 | 0.1636 | 0.1875 | 0.1748 | 0.0933 |
| Adaptation 3b: Conditioning (codes+descriptions) | 75 | 0.0592 | 0.0741 | 0.0658 | 0.0133 | 0.1037 | 0.1296 | 0.1152 | 0.0667 | 0.1754 | 0.2083 | 0.1905 | 0.0800 |

## Section 4 Descriptive Analyses (Full MathFish)

Source: `a3/krish/results/descriptive/summary.json`

- Problems scored: `13,065`
- Predicted zero-count standards: `51`
- Predicted prerequisite edge types: `554`

### Analysis 1: Standards Coverage

Top predicted standards (count):
- `N-RN.A.1` (2078)
- `5.NF.B.3` (1902)
- `7.EE.B.3` (1888)
- `6.NS.A.1` (1830)
- `A-REI.D.11` (1497)
- `S-CP.A.4` (1473)
- `8.SP.A.4` (1322)
- `6.SP.B.5` (1030)
- `4.NF.B.4c` (950)
- `F-IF.C.8b` (912)

Figure:
- `a3/krish/figures/analysis1_coverage_heatmap.png`

### Analysis 2: Prerequisite Chains

Top predicted prerequisite edges (`source -> target`, count):
- `8.EE.A.2 -> N-RN.A.1` (2078)
- `8.NS.A.1 -> N-RN.A.1` (2078)
- `3.OA.A.2 -> 5.NF.B.3` (1902)
- `3.OA.A.1 -> 5.NF.B.3` (1902)
- `6.RP.A.3c -> 7.EE.B.3` (1888)
- `6.EE.B.7 -> 7.EE.B.3` (1888)
- `5.NF.B.7 -> 6.NS.A.1` (1830)
- `5.NF.B.6 -> 6.NS.A.1` (1830)
- `8.EE.C.8 -> A-REI.D.11` (1497)
- `A-REI.D.10 -> A-REI.D.11` (1497)

Figure:
- `a3/krish/figures/analysis2_prereq_graph.png`

### Analysis 3: Publisher Comparison

Publisher-level stats:
- Illustrative Mathematics:
  - Problems: `11,712`
  - Avg predicted standards/problem: `4.7929`
  - Multi-standard rate: `97.2678%`
- Fishtank Learning:
  - Problems: `1,353`
  - Avg predicted standards/problem: `4.6977`
  - Multi-standard rate: `95.7871%`

Top predicted domains overall:
- `5.NF`, `7.EE`, `6.NS`, `4.NF`, `6.RP`, `N-RN.A`, `S-CP.A`, `1.OA`, `3.OA`, `4.MD`

Figures:
- `a3/krish/figures/analysis3_publisher_comparison.png`
- `a3/krish/figures/analysis3_publisher_comparison_domains.png`

## Raw Metric File Index

Addressing/Alignment:
- `a3/krish/results/core/roberta_similarity_publisher_k3.json`
- `a3/krish/results/core/roberta_ft_publisher_e2_light/metrics.json`
- `a3/krish/results/adaptations/hierarchical/metrics.json`
- `a3/krish/results/adaptations/rag_fewshot/metrics.json`
- `a3/krish/results/adaptations/standard_conditioning/metrics.json`
- `a3/krish/results/descriptive/summary.json`

Building On:
- `a3/krish/results/building_on/core/roberta_similarity_building_on_k3.json`
- `a3/krish/results/building_on/core/roberta_ft_building_on_e2_light/metrics.json`
- `a3/krish/results/building_on/adaptations/hierarchical/metrics.json`
- `a3/krish/results/building_on/adaptations/rag_fewshot/metrics.json`
- `a3/krish/results/building_on/adaptations/standard_conditioning/metrics.json`
