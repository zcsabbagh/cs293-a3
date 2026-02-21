# Krish - Benchmarking & IRR

## Your Responsibilities

1. **Benchmark 4 models** against two ground truths (our annotations + publisher labels)
2. **Calculate IRR** across all annotators on the 20 shared problems
3. **Find a 4th reviewer** (our teacher buddy) to annotate the shared problems

## Data You Need

**Download the full dataset** (too large for GitHub):
- `mathfish_train.jsonl` (13,065 problems, ~125MB): [Google Drive](https://drive.google.com/file/d/1_QdLBzRw35UdPFm8wxIH-LYjRVi2MlBS/view)
- Place it in the repo root (`cs293-a3/mathfish_train.jsonl`)
- You need this for running the TF-IDF baseline (needs all standard descriptions) and for any broader analysis

Everything else is already in the repo.

---

## Part 1: Find a 4th Reviewer

We need **4 annotators total** for our validation set (required by the assignment). Find a teacher buddy or 4th person willing to annotate the 20 shared problems. They should:

- Clone the repo and follow `INSTRUCTIONS.md` to run the annotation tool
- Use `--name teacher_buddy` when running the server
- Annotate **without any AI assistance** -- this is critical for a valid human baseline
- Send you their `annotations/teacher_buddy_annotations.jsonl` when done

**IMPORTANT: All human annotation must be done with absolutely no AI assistance. No ChatGPT, no Claude, no Copilot. The whole point is to create a human ground truth.**

---

## Part 2: Calculate IRR

Once all 4 annotators finish the 20 shared problems, calculate inter-rater reliability.

### Metric: Krippendorff's Alpha

We use Krippendorff's alpha because it handles:
- Multiple raters (4 annotators)
- Multi-label data (each problem can have multiple standards)
- Nominal/categorical data

### Formula

```
alpha = 1 - (D_o / D_e)
```

Where:
- `D_o` = observed disagreement (how much annotators actually disagree)
- `D_e` = expected disagreement (how much they'd disagree by chance)

For multi-label standards tagging, treat each (problem, standard) pair as a binary decision (tagged or not). Then:

- `D_o = (number of disagreeing pairs) / (total pairs)`
- `D_e` = expected disagreement based on marginal frequencies

**Interpretation:**
- `alpha = 1.0` = perfect agreement
- `alpha > 0.8` = reliable
- `alpha = 0.667-0.8` = tentative conclusions
- `alpha < 0.667` = unreliable, revisit codebook

### Compute at 3 Granularity Levels

| Level | What you compare | Example | Expected agreement |
|-------|-----------------|---------|-------------------|
| **Domain** | Did annotators pick the same domain? | `NBT` vs `OA` | Highest |
| **Cluster** | Same cluster? | `4.NBT.A` vs `4.NBT.B` | Medium |
| **Standard** | Exact same standard? | `4.NBT.A.1` vs `4.NBT.A.2` | Lowest |

### Python Library

```bash
pip install krippendorff
```

```python
import krippendorff
import numpy as np

# Build a reliability matrix: rows = annotators, columns = (problem, standard) pairs
# Values: 1 = tagged, 0 = not tagged, np.nan = missing
# See: https://github.com/pln-fing-udelar/fast-krippendorff

alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
```

### Annotation Files

All annotator files are JSONL at `annotations/<name>_annotations.jsonl`. Each line:

```json
{
  "problem_id": "im_practice_004579",
  "annotator": "zane",
  "standards": ["4.NBT.A.1", "4.NBT.A.2"],
  "notes": "",
  "skipped": false
}
```

The 20 shared problem IDs are listed in `annotations/assignments.json` under `"shared_ids"`.

---

## Part 3: Benchmark 4 Models

### Two Ground Truths

You'll evaluate each model against **two separate baselines**:

#### Ground Truth 1: Publisher Labels (already in the data)

Every problem in `annotations/problems.json` already has a `"standards"` field -- these are the labels assigned by the original curriculum publishers (Illustrative Mathematics / Fishtank Learning). Each entry is a `[relation_type, standard_code]` pair, e.g. `["Addressing", "4.NBT.A.1"]`.

For evaluation, filter to only `"Addressing"` and `"Alignment"` relations (these are equivalent per the MathFish paper -- both mean "directly aligns"). Ignore `"Building On"` and `"Building Towards"`.

This is your **pre-existing ground truth** -- no waiting on annotators needed.

#### Ground Truth 2: Human Consensus (from our annotations)

Once all 4 annotators finish the 20 shared problems, build consensus:
- For each problem, collect all standards tagged by any annotator
- Use **majority vote**: a standard is in the consensus if 3+ of 4 annotators tagged it
- If there's a 2-2 split, flag for discussion

This lets you compare: **do the models agree more with publishers or with us?** And do **we** agree with the publishers? (This is interesting analysis for the report.)

### Model 1: Lexical Baseline (TF-IDF Cosine Similarity)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load all 385 standard descriptions from standards.jsonl
# 2. TF-IDF vectorize: standard descriptions + problem text
# 3. For each problem, rank standards by cosine similarity to problem text
# 4. Predict top-k standards (tune k = 1, 3, 5)
```

This is NOT an LLM -- it's a simple text matching heuristic. The assignment specifically asks for at least one non-LLM approach.

#### Results

All benchmark results live in `RESULTS.md`.

#### Scripts

Run TF-IDF baseline (publisher labels):

```bash
python3 benchmark.py tfidf --k 1 --output preds/tfidf_k1.jsonl
python3 benchmark.py eval --preds preds/tfidf_k1.jsonl

python3 benchmark.py tfidf --k 3 --output preds/tfidf_k3.jsonl
python3 benchmark.py eval --preds preds/tfidf_k3.jsonl

python3 benchmark.py tfidf --k 5 --output preds/tfidf_k5.jsonl
python3 benchmark.py eval --preds preds/tfidf_k5.jsonl
```

Run LLM baselines (publisher labels):

```bash
python3 llm_benchmark.py all --output-dir preds --results results/llm_results.json
```

Run IRR (Krippendorff’s alpha):

```bash
python3 irr.py --output results/irr.json
```

### Model 2: Gemini (gemini-3.1-pro-preview)

Zero-shot prompt:

```
You are a K-12 math curriculum expert. Given this math problem, identify
which Common Core standard(s) it directly addresses (the "Addressing" relation).

Use this standards hierarchy to narrow your answer:
[insert relevant grade-level hierarchy from standards.jsonl]

Problem: {problem_text}

Return ONLY a JSON array of standard codes, e.g. ["4.NBT.A.1", "4.OA.A.3"]
```

### Model 3: OpenAI (gpt-5.2)

Same prompt as above, different model. Compare cost vs accuracy tradeoff.

### Model 4: Claude (claude-sonnet-4-6)

Same prompt as above. Now you have 3 LLMs + 1 lexical baseline = 4 models total (assignment asks for 3, we're doing 4).

### Evaluation Metrics

For each model, compute against **both** ground truths:

| Metric | Description |
|--------|-------------|
| **Precision** | Of standards the model predicted, how many were correct? |
| **Recall** | Of the ground truth standards, how many did the model find? |
| **F1 (standard-level)** | Harmonic mean of precision and recall at exact standard level |
| **F1 (cluster-level)** | Same but only comparing up to cluster (e.g., `4.NBT.A`) |
| **F1 (domain-level)** | Same but only comparing domains (e.g., `4.NBT`) |
| **Exact match** | Did the model get the exact set of standards right? (strict) |

Also analyze:
- Does performance vary by **grade level**? (K-5 vs 6-8 vs HS)
- Do models systematically confuse **nearby standards** (right domain, wrong cluster)?
- Which model is best at which granularity level?
- **Do models agree more with publisher labels or human consensus?**

---

## Where to Find Everything

| Resource | Location |
|----------|----------|
| Repo | `https://github.com/zcsabbagh/cs293-a3` |
| Full dataset | [Google Drive](https://drive.google.com/file/d/1_QdLBzRw35UdPFm8wxIH-LYjRVi2MlBS/view) -- download as `mathfish_train.jsonl` |
| Standards hierarchy | `standards.jsonl` (737 entries, 385 actual standards with descriptions) |
| Assigned problems | `annotations/problems.json` (40 problems with text + publisher labels) |
| Assignment config | `annotations/assignments.json` (who gets which problem IDs) |
| Annotation tool | `python3 annotate.py --name <name>` |
| Setup instructions | `INSTRUCTIONS.md` |
| Dataset documentation | `CLAUDE.md` (full schema, stats, field descriptions) |
| MathFish paper | [arXiv:2408.04226](https://arxiv.org/abs/2408.04226) |

---

## Timeline Suggestion

1. Find the 4th reviewer ASAP -- they need time to annotate
2. Do your own 25 annotations (use `--name krish` after re-running setup with your name)
3. **While waiting on annotators**: run models against publisher labels (Ground Truth 1) -- you can start this immediately
4. Collect all annotation files, build consensus (Ground Truth 2)
5. Compute IRR
6. Re-run model eval against human consensus
7. Write up results tables and error analysis
