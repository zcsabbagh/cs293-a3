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

### Notes

- Precision/recall tradeoff follows expectations: k=1 is more precise, k=5 is more recall-heavy.
- Exact match is near-zero for TF-IDF, suggesting lexical overlap alone rarely recovers full standard sets.
- The gap between domain- and standard-level scores shows most lexical matches land in the right neighborhood but miss fine-grained standards.
- For reporting, k=3 is a reasonable middle ground (balanced recall with modest precision), while k=1 can be framed as a high-precision baseline.

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

| Level | Precision | Recall | F1 | Exact |
|-------|-----------|--------|----|-------|
| Standard | 0.682 | 0.341 | 0.455 | 0.400 |
| Cluster | 0.714 | 0.375 | 0.492 | 0.400 |
| Domain | 0.714 | 0.395 | 0.508 | 0.450 |

### Notes

- GPT-5.2 leads on recall and F1 at all levels, with the strongest domain-level F1.
- Gemini is notably more precise but less recall-heavy, suggesting tighter predictions.
- Claude Sonnet 4.6 is consistently competitive, especially at cluster/domain granularity.
- Exact match improves substantially vs TF-IDF, especially at higher granularity.
- LLMs show a clear lift over TF-IDF on standard-level F1, indicating benefits beyond lexical overlap.
- Precision/recall profiles differ by model; this can motivate a discussion about cost vs coverage (recall) tradeoffs.
- If you need a single “best” model for downstream analysis, GPT-5.2 is the safest choice on aggregate F1.

Re-run if needed:

```bash
python3 llm_benchmark.py all --output-dir preds --results results/llm_results.json
```

## IRR (Krippendorff’s Alpha)

Computed with annotators: `krish`, `sera`, `teacher_buddy`, `zane`.

| Level | Alpha | Items | Annotators |
|-------|-------|-------|------------|
| Standard | 0.263 | 1340 | 4 |
| Cluster | 0.330 | 900 | 4 |
| Domain | 0.413 | 600 | 4 |
| Grade | 0.492 | 240 | 4 |
| Standard (no grade, 7.NS.A.3=8.NS.A.3) | 0.284 | 1180 | 4 |

### Notes

- Agreement is highest at domain level and lowest at exact standard level, which is typical for fine-grained tagging.
- Alpha values < 0.667 indicate only tentative agreement; flag this in the writeup and discuss ambiguity.
- Grade-level agreement is higher than domain/standard, reinforcing that annotators align more on coarse structure than exact codes.
- Standard (no grade) collapses grade-level differences (e.g., 7.NS.A.3 vs 8.NS.A.3). Agreement rises slightly vs exact standard, but remains low.
- Low alpha at the standard level suggests either ambiguous items, differing interpretations of “Addressing,” or standards that are close in meaning.
- Consider adding a short error analysis: list a few standards with frequent disagreement and explain why they are confusable.
- This supports a narrative that the task is inherently fine-grained and that model evaluation should be reported at multiple levels.

```bash
python3 irr.py --output results/irr.json
```
