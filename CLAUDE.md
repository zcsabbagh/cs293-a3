# CS293 Assignment 3 - MathFish Standards Alignment

## Project Overview

This project uses the **MathFish** dataset (Lucy et al., EMNLP 2024 Findings, Allen Institute for AI) to evaluate and improve models that assess whether K-12 math problems align with Common Core educational standards. The assignment involves creating a validation set, measuring inter-rater reliability, evaluating off-the-shelf models, and either building a tool or improving a model.

**Paper:** "Evaluating Language Model Math Reasoning via Grounding in Educational Curricula" (arXiv:2408.04226)
**Authors:** Li Lucy, Tal August, Rose E. Wang, Luca Soldaini, Courtney Allison, Kyle Lo
**Repository:** https://github.com/allenai/mathfish
**HuggingFace:** https://huggingface.co/datasets/allenai/mathfish

## Dataset: `mathfish_train.jsonl`

The train split of the MathFish dataset. **13,065 records**, one JSON object per line.

### Sources

| Source | Count | Description |
|--------|-------|-------------|
| Illustrative Mathematics (IM) | 11,712 | Open educational resource, K-12 math curriculum |
| Fishtank Learning (FL) | 1,353 | Open educational resource, grades 3-11 |

### Schema (top-level keys)

| Key | Type | Description |
|-----|------|-------------|
| `id` | string | Unique problem ID (e.g. `im_practice_004579`, `fl_target_task_000123`) |
| `text` | string | Plain-text version of the math problem/activity |
| `metadata` | object | Source-specific metadata (see below) |
| `elements` | object | Map of placeholder tokens (e.g. `###IMAGE0###`) to filenames for embedded images/tables |
| `standards` | array | List of `[relation_type, standard_code]` pairs (e.g. `["Addressing", "A-CED.A.3"]`) |
| `acquisition_date` | string | Date scraped (YYYY-MM-DD format) |
| `source` | string | `"Illustrative Mathematics"` or `"Fishtank Learning"` |
| `has_image` | boolean | Whether the problem references images |
| `is_teacher_facing` | boolean | Whether the content is teacher-facing (includes pedagogy notes) vs student-facing |
| `num_problems` | integer | Number of sub-problems in the activity |
| `is_duplicate` | boolean | Whether flagged as a duplicate |

### Metadata fields

Varies by source. Common fields include:

- `problem_activity_type`: Type of content (see distribution below)
- `url`: Original source URL
- `html`: HTML filename
- `grade / subject`: Grade level or subject (IM only; FL uses `None`)
- `title`: Activity title (lesson activities and tasks)
- `unit_number`, `lesson_number`: Curriculum location (lessons only)
- `problem_activity_html`: Raw HTML of the problem

### Problem Activity Types

| Type | Count | Source |
|------|-------|--------|
| `lesson activity` | 5,978 | IM |
| `practice` | 4,403 | IM |
| `task` | 908 | IM |
| `target_task` | 680 | FL |
| `anchor_problems` | 444 | FL |
| `center` | 406 | IM |
| `anchor_task` | 229 | FL |
| `modeling prompt` | 17 | IM |

### Standards Relation Types

Standards use Common Core State Standards (CCSS) codes (e.g. `4.NBT.A.1`, `A-CED.A.3`). Each standard entry is a 2-element array: `[relation_type, standard_code]`.

| Relation Type | Count | Meaning |
|--------------|-------|---------|
| `Addressing` | 7,278 | Problem directly addresses this standard (equivalent to `Alignment`) |
| `Building On` | 2,843 | Problem builds on prerequisite skills from this standard |
| `Alignment` | 1,498 | Problem aligns with this standard (equivalent to `Addressing`) |
| `Building Towards` | 1,008 | Problem builds towards future skills in this standard |

**Note:** `Addressing` == `Alignment` per the paper. The primary evaluation target.

6,698 records have at least one standard label; 6,367 have empty standards arrays.

### Grade Levels

Covers K-12. IM uses `grade-N` / `kindergarten` / `algebra-1` / `geometry` / `algebra-2` slugs. FL uses `3rd-grade` through `8th-grade` slugs (908 FL entries have no grade field).

### Key Statistics

- **5,674** problems contain images; **7,391** do not
- **6,401** teacher-facing; **6,664** student-facing
- **1,922** flagged as duplicates (14.7%)
- **6,413** have non-empty `elements` (embedded images/tables)

## Common Core Standards Hierarchy

Standards follow a tree structure defined by Achieve the Core (ATC):

```
Grade (e.g. 4)
  -> Domain (e.g. 4.NBT - Number & Operations in Base Ten)
    -> Cluster (e.g. 4.NBT.A - Generalize place value understanding)
      -> Standard (e.g. 4.NBT.A.1 - Recognize that a digit represents 10x the value...)
```

385 total standards across K-12. The companion dataset `allenai/achieve-the-core` on HuggingFace contains standard descriptions and metadata.

## Tasks Defined by the Paper

1. **Verification**: Given a problem and a standard description, determine if the problem aligns with that standard (binary yes/no).
2. **Tagging**: Given a problem and a set of candidate standards, select all standards the problem aligns with.

Key finding: LMs (including GPT-4) struggle with fine-grained verification and tagging, often predicting standards that are conceptually close but subtly wrong.

## File Structure

```
cs293-a3/
  ASSIGNMENT.md          # Assignment specification
  CLAUDE.md              # This file
  mathfish_train.jsonl   # MathFish train split (13,065 problems)
```

## Useful Commands

```bash
# Count records
wc -l mathfish_train.jsonl

# Pretty-print first record
head -1 mathfish_train.jsonl | python3 -m json.tool

# Count records with standards
python3 -c "
import json
with open('mathfish_train.jsonl') as f:
    data = [json.loads(l) for l in f]
print(f'Total: {len(data)}')
print(f'With standards: {sum(1 for d in data if d[\"standards\"])}')
"
```
