# Therapy Assistant Training Pipeline

This repo centers on one methodology: train a long-horizon personal therapy
assistant with future-oriented rewards.

The implemented pipeline has three stages:

1. Multi-agent simulation builds a candidate-response dataset `Dr`
2. A future-oriented reward model is trained on `Dr`
3. GRPO fine-tunes the policy model with future reward plus think-format reward

## Core modules

```text
training/
|-- datasets.py            # ESConv / ExTES / JSONL seed loading
|-- llm.py                 # OpenAI-compatible client helpers
|-- memory.py              # durable episode memory used in rollouts
|-- prompts.py             # policy / critic prompt templates grounded in env state
|-- simulator.py           # environment-backed rollout wrapper
|-- critic.py              # separated critic + future reward computation
|-- simulate_dialogues.py  # stage 1
|-- reward_model.py        # stage 2
`-- grpo_policy.py         # stage 3
```

## Recommended first run

### 1. Generate simulation data

Use real dialogue prefixes whenever possible. `esconv_hf` is the default
recommended source.

```bash
python -m training.simulate_dialogues \
  --output-dir artifacts/sim_data \
  --examples-source esconv_hf \
  --dataset-name thu-coai/esconv \
  --dataset-split train \
  --max-seed-examples 200 \
  --policy-model Qwen/Qwen2.5-3B-Instruct \
  --critic-model Qwen/Qwen2.5-7B-Instruct \
  --episodes-per-seed 2 \
  --num-candidates 4 \
  --rollout-steps 6 \
  --max-turns 16
```

Outputs:

- `artifacts/sim_data/candidate_rewards.jsonl`
- `artifacts/sim_data/trajectories.jsonl`
- env-aligned trajectories with session continuity, budget usage, and hidden-state summaries

### 2. Train the future-oriented reward model

```bash
python -m training.reward_model \
  --input-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/reward_model
```

Outputs:

- trained reward model under `artifacts/reward_model/`
- `reward_model_audit.jsonl`
- `reward_model_audit_summary.json`

### 3. Run GRPO

```bash
accelerate launch -m training.grpo_policy \
  --prompt-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --reward-model-dir artifacts/reward_model \
  --output-dir artifacts/grpo_policy
```

## Dataset choices

Supported seed sources:

- `esconv_hf`
- `extes_hf`
- `extes_jsonl`
- `jsonl`
- `tasks` for ablations only

For a competition submission, prefer `esconv_hf` or `extes_hf` over handcrafted tasks.

## Practical advice

- Start with a `0.5B-1.5B` reward model.
- Start GRPO on a `3B` policy, not `7B`.
- Save rollout traces and audit files after every run.
- Refresh the `results/` folder only with current methodology outputs.
