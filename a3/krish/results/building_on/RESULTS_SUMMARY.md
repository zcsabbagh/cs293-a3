# A3 Results Summary (Krish) - Building On

All runs in this folder use `relations = "Building On"` (publisher labels), not `Addressing/Alignment`.

## Section 2: Performance Analysis

| Model | Standard F1 | Cluster F1 | Domain F1 | Exact Match (Standard) |
|---|---:|---:|---:|---:|
| RoBERTa similarity baseline | 0.0167 | 0.0390 | 0.0789 | 0.0000 |
| RoBERTa fine-tuned (`e2_light`) | 0.0403 | 0.0586 | 0.0903 | 0.0000 |

Note: fine-tuned evaluation uses `69` test items after label-frequency filtering; baseline uses `75`.

Sources:

- `core/roberta_similarity_building_on_k3.json`
- `core/roberta_ft_building_on_e2_light/metrics.json`

## Section 3: Adaptations

### Adaptation 1 - Hierarchical Prediction (domain -> cluster -> standard)

- Standard F1: `0.0123`
- Cluster F1: `0.0219`
- Domain F1: `0.0351`
- Source: `adaptations/hierarchical/metrics.json`

### Adaptation 2 - Retrieval-Augmented Few-Shot Prompting

- Standard F1: `0.0809`
- Cluster F1: `0.1296`
- Domain F1: `0.1846`
- Source: `adaptations/rag_fewshot/metrics.json`

### Adaptation 3 - Standard Description Conditioning

- Codes-only standard F1: `0.0568`
- Codes+descriptions standard F1: `0.0658`
- Source: `adaptations/standard_conditioning/metrics.json`
