# Research Task Analysis

This note compares the current handcrafted tasks in [src/tasks.py](./src/tasks.py)
with the dialogue setup used in RLFF-ESC.

## What the paper uses

RLFF-ESC evaluates on two real ESC datasets:

- ESConv: 1,300 crowd-sourced dialogues with problem type, emotion type, and
  strategy annotations.
- ExTES: 11,177 ChatGPT-generated dialogues verified by human annotators.

Paper references:

- https://arxiv.org/pdf/2508.12935
- https://huggingface.co/papers/2508.12935
- https://huggingface.co/datasets/thu-coai/esconv

The paper's stage-1 simulation operates on dialogue contexts `c_{1:t-1}` drawn
from those datasets. For each context, the policy samples a candidate system
response, the user simulator rolls the dialogue forward, and the critic judges
the resulting future trajectory.

## How the current handcrafted tasks differ

The three tasks in `src/tasks.py` are useful benchmark personas, but they
diverge from the paper's data regime in several important ways:

1. They are handcrafted scenarios, not real dataset prefixes.
2. They expose only three narrow situations, while ESConv and ExTES contain
   much broader variation in emotions, problem types, and support styles.
3. They encode progress through hidden scalar states and scripted lines, while
   the paper's training pipeline works from real dialogue history.
4. They compress the support goal into a few thresholds (`trust`, `distress`,
   `stage`) instead of using authentic dialogue outcomes and dataset contexts.
5. They only loosely reflect strategy diversity. ESConv has eight strategies;
   ExTES expands this further. The handcrafted tasks do not reproduce that
   distribution.

## What to use instead for training

For paper-aligned training, the seed "task" should be:

- a real dialogue prefix from ESConv or ExTES,
- plus metadata such as `problem_type`, `emotion_type`, and `situation`,
- optionally the ground-truth supporter reply for later supervised warm-start or
  offline inspection.

That is now supported in the training pipeline via:

- `--examples-source esconv_hf`
- `--examples-source extes_jsonl`
- `--examples-source jsonl`

The benchmark-style tasks can still exist for ablations, but they should no
longer be treated as the primary training distribution.
