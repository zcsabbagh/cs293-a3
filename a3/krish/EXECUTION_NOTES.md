# A3 Execution Notes (Option A: Reconnect to Prerequisites)

## Framing Paragraph Draft

Our A1 hypothesis focused on prerequisite knowledge: students often miss target problems not because the current skill is unknown, but because foundational knowledge is missing. In A2, we evaluated automated CCSS alignment as an intermediate step, since identifying what a problem directly addresses is necessary before inferring what it builds on. In A3, we connect these explicitly by modeling both direct alignment ("Addressing") and prerequisite dependencies ("Building On"), so standards tagging becomes the mechanism for prerequisite prediction rather than the endpoint. This restores continuity with Koedinger's KLI framing and our A1 motivation while preserving A2's empirical results as methodological groundwork.

## Immediate Team Plan (Current)

1. Adjudicate shared 20 items:
- Run `python3 a3/krish/scripts/build_consensus.py`
- Use `a3/krish/results/annotation_prep/shared_adjudication_report.md` in team meeting
- Finalize consensus labels for disputed items

2. Establish RoBERTa baseline (no FT):
- Run `python3 a3/krish/scripts/roberta_similarity.py --dataset publisher_full --top-k 3 --grade-filter`
- Record metrics from `a3/krish/results/core/roberta_similarity_publisher_k3.json`

3. Fine-tune RoBERTa:
- Run `python3 a3/krish/scripts/train_roberta_ft.py --dataset publisher_full --output-dir a3/krish/results/core/roberta_ft_publisher --save-model`
- Use `a3/krish/results/core/roberta_ft_publisher/metrics.json` as primary A3 model table

4. Expand annotation set with active learning:
- Run `python3 a3/krish/scripts/select_active_learning.py --run-dir a3/krish/results/core/roberta_ft_publisher --top-n 150`
- Split candidates across annotators and label 30-50 each

5. Re-train after expansion:
- Re-run fine-tuning command
- Compare old/new `metrics.json`
- Use `error_examples_test.jsonl` for written error analysis examples

6. Run A3 Section 3 adaptations:
- `python3 a3/krish/scripts/adaptation_hierarchical.py`
- `python3 a3/krish/scripts/adaptation_rag_fewshot.py --dataset publisher_full --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0`
- `python3 a3/krish/scripts/adaptation_standard_conditioning.py --dataset publisher_full --provider openai --model gpt-5.2 --train-ratio 0.8 --val-ratio 0.1 --max-test 0`

7. Run A3 Section 4 descriptive analyses:
- `python3 a3/krish/scripts/descriptive_analyses.py --data-path mathfish_train.jsonl`
- Use generated figures under `a3/krish/figures/`

## Deliverable Mapping

- Validation dataset + adjudication process:
  - `a3/krish/results/annotation_prep/shared_consensus_draft.jsonl`
  - `a3/krish/results/annotation_prep/shared_adjudication_report.md`
- Model performance table:
  - `a3/krish/results/core/roberta_similarity_publisher_k3.json`
  - `a3/krish/results/core/roberta_ft_publisher_e2_light/metrics.json`
- Error analysis:
  - `a3/krish/results/core/roberta_ft_publisher_e2_light/error_examples_test.jsonl`
- Annotation expansion details:
  - `a3/krish/results/annotation_prep/active_learning_candidates.jsonl`
- Adaptation metrics:
  - `a3/krish/results/adaptations/hierarchical/metrics.json`
  - `a3/krish/results/adaptations/rag_fewshot/metrics.json`
  - `a3/krish/results/adaptations/standard_conditioning/metrics.json`
- Descriptive analyses:
  - `a3/krish/results/descriptive/summary.json`
  - `a3/krish/figures/analysis1_coverage_heatmap.png`
  - `a3/krish/figures/analysis2_prereq_graph.png`
  - `a3/krish/figures/analysis3_publisher_comparison.png`
