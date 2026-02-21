# Sera - Report Writing

## Your Responsibility

Write the **4-6 page PDF report** (the main deliverable for Assignment 3). The report becomes the methods and results sections of our final paper.

---

## Report Structure

Follow the sections from `ASSIGNMENT.md`:

### Section 1: Validation Dataset (~1 page)

**What to write:**
- We annotated **25 problems per person** (20 shared + 5 unique) from the MathFish train split
- 4 annotators total (us 3 + teacher buddy)
- **Codebook**: Common Core State Standards hierarchy from Achieve the Core (`standards.jsonl`). Annotators tag each problem with standard codes (e.g., `4.NBT.A.1`) and relation types (Addressing, Building On, Building Towards)
- **Annotator training**: We built a browser-based annotation tool with hierarchical navigation (grade -> domain -> cluster -> standard) so annotators didn't need to memorize 385 standards
- **Selection rationale**: Filtered to problems that are labeled (for sanity-checking against publisher labels), non-image, non-duplicate, 20-2000 chars. Stratified sample via random shuffle (seed=42) to cover grade diversity
- **Why this size**: Assignment guideline says 20-25 per person for analytic models. 20 shared problems give us IRR, 5 unique per person adds breadth

**Where to find the data:**
- Problem text and metadata: `annotations/problems.json`
- Who got what: `annotations/assignments.json`
- Individual annotations: `annotations/<name>_annotations.jsonl`
- Filtering criteria: `setup_annotations.py` lines 19-41

### Section 2: IRR (~0.5-1 page)

**What to write:**
- Metric: **Krippendorff's alpha** (handles 4 raters, multi-label, nominal data)
- Formula: `alpha = 1 - (D_o / D_e)` where D_o = observed disagreement, D_e = expected disagreement by chance
- Computed at 3 levels: domain, cluster, standard (expect decreasing agreement)
- Justify why Krippendorff's alpha over Cohen's kappa (kappa is pairwise only, alpha handles n raters)
- Report the numbers Krish computes + interpretation (>0.8 = reliable, 0.667-0.8 = tentative)
- If agreement is low at standard level but high at domain level, discuss what that means (the task is inherently hard at fine granularity -- the MathFish paper found the same thing)
- Compare our annotations to publisher labels as additional validation

**Data from Krish:** He'll provide the alpha values at each level + a confusion analysis.

### Section 3: Off-the-shelf Models (~1.5-2 pages)

**What to write:**
- We tested **4 models** (assignment asks for 3, we did 4):

| Model | Type | Why included |
|-------|------|-------------|
| TF-IDF cosine similarity | Lexical baseline | Non-LLM approach (required). Matches problem text to standard descriptions by word overlap |
| Gemini | LLM (Google) | One of the big-name LLMs |
| GPT-4o | LLM (OpenAI) | Another big-name LLM, different architecture |
| Claude | LLM (Anthropic) | Third LLM for comparison |

- Report **precision, recall, F1** at standard / cluster / domain levels
- Report **exact match accuracy**
- Include a results table and ideally a figure (bar chart of F1 by model by granularity level)
- Error analysis: where do models fail? Right domain but wrong cluster? Confusing nearby standards? Worse at certain grade levels?
- Reference the MathFish paper's findings: they found GPT-4 predicted standards that were "conceptually close but subtly wrong" -- do our results confirm this?

**Data from Krish:** He'll provide per-model predictions and evaluation metrics.

### Section 4: Track Choice (~0.5 page)

**What to write:**
- Based on results, we choose **Track 2: Build a Model**
- Justify: if off-the-shelf models don't perform well at fine-grained tagging (likely -- the paper found this), there's room to improve
- Brief preview of our plan (3 proposed improvements):
  1. **Hierarchical prediction**: Force model to predict domain -> cluster -> standard sequentially
  2. **Retrieval-augmented few-shot**: Retrieve similar labeled problems as in-context examples
  3. **Standard description conditioning**: Include full standard descriptions, not just codes

---

## Where to Find Everything

| Resource | Location |
|----------|----------|
| Repo | `https://github.com/zcsabbagh/cs293-a3` |
| Assignment spec | `ASSIGNMENT.md` |
| Dataset documentation | `CLAUDE.md` (comprehensive dataset schema, stats, field descriptions) |
| MathFish paper | [arXiv:2408.04226](https://arxiv.org/abs/2408.04226) |
| Paper authors | Li Lucy, Tal August, Rose E. Wang, Luca Soldaini, Courtney Allison, Kyle Lo (Allen AI) |
| Standards file | `standards.jsonl` -- 737 entries, 385 actual standards with descriptions |
| Annotation tool code | `annotate.py` + `static/index.html` |
| Setup script | `setup_annotations.py` |
| Sample problems | `mathfish_sample.jsonl` (5 labeled examples) |
| Full dataset | [HuggingFace](https://huggingface.co/datasets/allenai/mathfish) (train split, 13,065 records) |
| Original paper repo | [github.com/allenai/mathfish](https://github.com/allenai/mathfish) |

## Figures to Include

1. **Table**: IRR values at domain/cluster/standard levels
2. **Table**: Model performance (P/R/F1) at each granularity level
3. **Figure**: Bar chart comparing model F1 scores
4. **Example**: 1-2 annotated problems showing what correct vs model-predicted standards look like
5. **Optional figure**: Confusion matrix or heatmap of model errors by domain

## Style Notes

- 4-6 pages, PDF
- Use figures and text examples where appropriate (assignment says this explicitly)
- This should read like methods + results sections of a research paper
- Cite the MathFish paper (Lucy et al., EMNLP 2024 Findings)
- Attach all annotated datasets with the submission
