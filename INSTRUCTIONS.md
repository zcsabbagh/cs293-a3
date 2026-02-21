# MathFish Annotation Tool - Setup Instructions

## Overview

This tool lets you annotate K-12 math problems with Common Core standards. Each annotator has **20 shared problems** (for inter-rater reliability) and **5 unique problems**.

## Prerequisites

- Python 3 (no extra packages needed)
- A modern web browser

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/zanedurante/cs293-a3.git
cd cs293-a3
```

### 2. Download the dataset

Download `mathfish_train.jsonl` from [HuggingFace](https://huggingface.co/datasets/allenai/mathfish) and place it in the repo root.

> Only needed if you want to regenerate assignments. The pre-built assignment files in `annotations/` are already included.

### 3. Run the annotation server

```bash
python3 annotate.py --name <your_name>
```

Replace `<your_name>` with your assigned annotator name. Valid names:

- `zane`
- `alice`
- `bob`
- `teacher_buddy`

This opens a browser tab at `http://localhost:8000` with the annotation interface.

### 4. Annotate

For each problem:

1. **Read the math problem** displayed at the top
2. **Pick a grade level** (K through HS)
3. **Pick a domain** within that grade
4. **Select standards** that the problem aligns with
5. Click **Save & Next** to record your annotation and move on

You can also:
- **Search** for standards by keyword or code using the search bar
- **Skip** problems you're unsure about
- **Add from another domain** if a problem spans multiple areas
- **Navigate** between problems using the progress dots at the bottom

### 5. Your annotations are saved automatically

Annotations save to `annotations/<your_name>_annotations.jsonl` each time you click Save. You can stop and resume at any time — your progress is preserved.

## Troubleshooting

**Port already in use:**
```bash
python3 annotate.py --name <your_name> --port 8001
```

**"Run setup_annotations.py first" error:**
The `annotations/assignments.json` and `annotations/problems.json` files should already be in the repo. If they're missing, re-clone or ask the team lead.

**Regenerating assignments (team lead only):**
```bash
python3 setup_annotations.py --annotators zane alice bob teacher_buddy --overlap 20 --unique 5
```
This requires `mathfish_train.jsonl` in the repo root.

## Submitting Annotations

When you're done annotating, send your `annotations/<your_name>_annotations.jsonl` file to the team lead (e.g., via Slack or email). These files are gitignored to avoid merge conflicts.
