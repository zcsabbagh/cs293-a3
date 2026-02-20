# CS293 Assignment 3

This assignment is the heart of your work in this class. If you end up in Track 1, this one will feel heavier and the next assignment lighter. If you're in Track 2, it's the other way around -- we push some deliverables to the next assignment, but this breather is a bit deceptive, so keep pace. Divide work among your group members!

**Objective:** Create a reliable validation set and use it to test existing models; analyze performance and make progress towards measuring and/or supporting practice.

**Deliverable:** A 4-6 page write-up (PDF) that will easily become the methods and results sections of your final paper. Include the following sections, using figures and text examples where appropriate. Attach all annotated datasets.

## 1. Validation Dataset

Manually annotate a dataset that represents "ground truth". You should have 4 annotators - yourselves and your teacher buddy. Write up your codebook and your process for training annotators, as well as a brief rationale explaining the quality of your validation set. In a perfect world, a validation set would cover every possible scenario and really thoroughly test a model's performance... but in our constrained world, how did you choose the composition of this set to still feel confident about its utility?

The size and nature of this dataset depends on your project, so check with us whether your annotation scope is sufficient! As a rough guideline:

- If you're targeting an **analytic model** (classification, scoring, etc), it might depend on the length of the unit of analysis. Taking PERSUADE as an example, target 20-25 examples per person. For NCTE, since utterances are short, maybe 50-80 per person.
- If you're targeting a **generative model**, it'll depend on the length of generation, but taking MathFish as an example, target writing 10-15 math items per person.

## 2. IRR

Do you and your teacher buddy agree on the annotations? How do you know? Which metric do you choose and why? To calculate IRR metrics, you don't necessarily need to all annotate the same examples (you don't need total overlap). You can choose a smaller subset to evaluate whether you're on the same page, then divide and conquer new examples if this would be more helpful for your project. Do you agree with the IRR metric? If you're using a generative model, most standard IRR metrics don't naturally fit, so how will you choose to evaluate alignment?

## 3. Off-the-shelf Models

Try 3 existing off-the-shelf models. We'd prefer that not all 3 are big-name LLMs, so try to search for existing methods from prior research or use a reasonable lexical heuristic! If truly nothing exists, not even close, come talk to us and we'll likely let you evaluate smaller LLMs. Evaluate these models on your validation data, and report performance. What do you make of these statistics? Are there certain contexts in which the model performs better than others?

## 4. Track Choice

At this point, based on whether these existing models perform well or badly, you can make a choice about whether you'd like to improve the model or leverage the model to design a tool. Let us know your choice and why.

---

## Track 2: Build a Model

### 1. Proposal

The ready-made models aren't cutting it, and you can do better. What's the plan? Describe your proposed modeling pipeline with at least 3 potential adjustments to further improve performance. Address the following considerations:

1. **Annotation:** If your novel approach involves novel data, how will you best leverage and respect (!!) annotator expertise? How will you ensure high quality annotations?
2. **Construct validity:** How will you justify your model to the education world (which values validated instruments)? Can you demonstrate alignment with another instrument (a survey, an observational score, etc) that is well established? How is this usually evaluated in education? Cite sources.
3. **Limitations:** What will likely slip through the cracks with this approach? In industry we call this a pre-mortem. What do you anticipate later in the error analysis and why?

### 2. Annotate

Make significant progress on obtaining new data. Submit your annotated dataset.
