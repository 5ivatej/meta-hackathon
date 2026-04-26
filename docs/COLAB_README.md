# Colab Runbook

This is the shortest path to running the current methodology in Google Colab:

1. build simulation data
2. train the future-oriented reward model
3. run GRPO on the therapy-assistant policy

The direct Python path is the recommended Colab path. Docker Compose is
optional for local reproducibility.

If you want an executable notebook instead of a prose runbook, use
[Therapy_Assistant_OpenEnv_Colab.ipynb](Therapy_Assistant_OpenEnv_Colab.ipynb).

## 1. Install

```bash
!pip install -r requirements-training.txt
```

## 2. Optional: start the OpenEnv server

If you want to inspect the environment interactively:

```bash
!python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## 3. Generate simulation data

Recommended first pass:

```bash
!python -m training.simulate_dialogues \
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

## 4. Train the reward model

```bash
!python -m training.reward_model \
  --input-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/reward_model
```

Outputs:

- `artifacts/reward_model/`
- `artifacts/reward_model/reward_model_audit.jsonl`
- `artifacts/reward_model/reward_model_audit_summary.json`

## 5. Run GRPO

```bash
!accelerate launch -m training.grpo_policy \
  --prompt-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --reward-model-dir artifacts/reward_model \
  --output-dir artifacts/grpo_policy
```

## 6. ExTES variant

If you want to train on a public ExTES-style source instead of ESConv:

```bash
!python -m training.simulate_dialogues \
  --output-dir artifacts/sim_data_extes \
  --examples-source extes_hf \
  --dataset-name ailover/ExTES \
  --max-seed-examples 200 \
  --policy-model Qwen/Qwen2.5-3B-Instruct \
  --critic-model Qwen/Qwen2.5-7B-Instruct
```

## 7. Optional local Docker Compose

The local Docker Compose flow mirrors the same three stages:

```bash
docker compose --profile simulate up --build simulator-data
docker compose --profile reward up --build reward-model-trainer
docker compose --profile grpo up --build grpo-trainer
```

## 8. Practical guidance

- Start with `200` seed examples before scaling.
- Keep the reward model small and cheap.
- Save the audit files from the reward model after every run.
- Refresh `results/` only with artifacts from the current methodology, not older benchmark outputs.
