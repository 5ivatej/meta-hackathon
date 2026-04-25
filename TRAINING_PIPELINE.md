# RLFF-ESC Training Pipeline

This repo now contains a paper-faithful training scaffold in `training/` that
separates the benchmark environment from the training stack.

## Goal

Implement the three-stage RLFF-ESC pipeline:

1. Multi-agent simulation to build a dialogue reward dataset `Dr`.
2. Train a future-oriented reward model on `Dr`.
3. Fine-tune the policy model with GRPO using the learned reward plus a format reward.

## Repo layout

```text
training/
|-- datasets.py            # seed example loading + prompt dataset preparation
|-- llm.py                 # OpenAI-compatible client helpers
|-- memory.py              # persistent episode memory for long-horizon rollouts
|-- prompts.py             # policy / user-sim / critic / summarizer prompts
|-- simulate_dialogues.py  # stage 1: build Dr with 3-role simulation
|-- reward_model.py        # stage 2: future-oriented reward model training
`-- grpo_policy.py         # stage 3: GRPO policy optimization
```

## Recommended execution order

### 1. Build simulation data

Start with a small model set on Colab and generate candidate rollouts:

```bash
python -m training.simulate_dialogues \
  --output-dir artifacts/sim_data \
  --policy-model Qwen/Qwen2.5-3B-Instruct \
  --user-model Qwen/Qwen2.5-3B-Instruct \
  --critic-model Qwen/Qwen2.5-7B-Instruct \
  --examples-source tasks \
  --episodes-per-seed 4 \
  --num-candidates 4 \
  --rollout-steps 6 \
  --max-turns 16
```

Outputs:

- `candidate_rewards.jsonl`: `(prompt_context, response, reward)` records for RM training
- `trajectories.jsonl`: chosen trajectory traces

### 2. Train the reward model

```bash
python -m training.reward_model \
  --input-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/reward_model
```

### 3. Run GRPO

```bash
accelerate launch -m training.grpo_policy \
  --prompt-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --reward-model-dir artifacts/reward_model \
  --output-dir artifacts/grpo_policy
```

## Practical first run

With your budget, the first milestone should be:

- generate `5k-20k` candidate reward examples
- train a small reward model (`0.5B-1.5B`)
- run a short GRPO job on a `3B` policy with LoRA

Do not start with a 7B GRPO run. Use the current benchmark server to evaluate
checkpoints after every stage.
