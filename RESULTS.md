# Results

## TF-IDF Baseline (Publisher Labels, 40 problems)

Top-k results using `benchmark.py` (ground truth = Addressing/Alignment):

| k | Level | Precision | Recall | F1 | Exact |
|---|-------|-----------|--------|----|-------|
| 1 | Standard | 0.125 | 0.114 | 0.119 | 0.100 |
| 1 | Cluster | 0.150 | 0.150 | 0.150 | 0.100 |
| 1 | Domain | 0.175 | 0.184 | 0.179 | 0.150 |
| 3 | Standard | 0.075 | 0.205 | 0.110 | 0.000 |
| 3 | Cluster | 0.112 | 0.325 | 0.167 | 0.000 |
| 3 | Domain | 0.153 | 0.447 | 0.228 | 0.000 |
| 5 | Standard | 0.060 | 0.273 | 0.098 | 0.000 |
| 5 | Cluster | 0.087 | 0.400 | 0.143 | 0.000 |
| 5 | Domain | 0.112 | 0.500 | 0.184 | 0.000 |

## LLM Benchmarks (Publisher Labels)

Models used: Gemini `gemini-3.1-pro-preview`, OpenAI `gpt-5.2`, Anthropic `claude-sonnet-4-6`.

### OpenAI (gpt-5.2)

| Level | Precision | Recall | F1 | Exact |
|-------|-----------|--------|----|-------|
| Standard | 0.446 | 0.568 | 0.500 | 0.350 |
| Cluster | 0.520 | 0.650 | 0.578 | 0.400 |
| Domain | 0.587 | 0.711 | 0.643 | 0.500 |

### Anthropic (claude-sonnet-4-6)

| Level | Precision | Recall | F1 | Exact |
|-------|-----------|--------|----|-------|
| Standard | 0.400 | 0.591 | 0.477 | 0.375 |
| Cluster | 0.481 | 0.650 | 0.553 | 0.475 |
| Domain | 0.521 | 0.658 | 0.581 | 0.525 |

### Gemini (gemini-3.1-pro-preview)

Failed with quota error (HTTP 429). See `results/llm_results.json` for full error text.

Re-run if quota is available:

```bash
python3 llm_benchmark.py all --output-dir preds --results results/llm_results.json
```

## IRR (Krippendorff’s Alpha)

Pending: requires at least two annotators with overlapping annotations.

```bash
python3 irr.py --output results/irr.json
```
