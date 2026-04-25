---
title: Therapy Assistant OpenEnv
emoji: "🧠"
sdk: docker
pinned: false
tags:
  - openenv
  - emotional-support
  - personal-therapy-assistant
  - long-horizon
---

# Therapy Assistant OpenEnv

OpenEnv training environment for a long-horizon personal therapy assistant.

This project is built for the OpenEnv Hackathon 2026. The core claim is:

> train a therapy-style assistant to optimize future user outcomes, not just
> locally empathetic one-turn replies.

## Why this matters

Emotional support is a hard personalized task for LLMs because:

- the user's real issue is often hidden at the start
- strong responses only pay off after several turns or sessions
- premature advice damages trust
- safety handling must stay calm, timely, and continuous

This environment is designed to make those failures trainable.

## Competition fit

The strongest fit is:

- Theme #2: long-horizon planning and instruction following
- Theme #3.2: personalized tasks

The environment trains a model to:

- infer hidden emotional state
- recover from early mistakes
- track continuity across sessions
- decide when to explore, reflect, plan, or escalate for safety

## Environment design

The OpenEnv environment lives in:

- [src/env.py](src/env.py)
- [src/tasks.py](src/tasks.py)
- [src/seeker.py](src/seeker.py)
- [src/grader.py](src/grader.py)

Key properties:

- multi-session therapy arcs
- hidden user state: trust, distress, openness, reveal progress
- delayed consequences and hard recovery
- continuity, budget, and time constraints
- hard-gated success conditions

### Built-in therapy arcs

| Task ID | Difficulty | What the agent must do |
| --- | --- | --- |
| `work_stress_venting` | easy | Build alliance, surface burnout, and move toward one realistic recovery step |
| `guarded_relationship` | medium | Earn trust before the real issue is disclosed, then help the user name and plan carefully |
| `crisis_fragile_trust` | hard | Preserve fragile trust, handle risk safely, and carry support forward across sessions |

### Action / observation

Action is free-text:

```python
class Action(BaseModel):
    message: str
```

Observation is partially observable:

```python
class Observation(BaseModel):
    seeker_utterance: str
    turn: int
    remaining_turns: int
    stage_hint: str
    task_id: str
    scenario_brief: str
```

The environment tracks much more hidden state internally, including continuity,
working goals, memory summary, budget usage, and session index.

## Reward design

The reward is designed around future trajectory quality, not only local style.

At a high level it combines:

- immediate conversational quality
- future-oriented trajectory value
- anti-gaming penalties
- long-horizon continuity terms

The project’s training stack then learns from that idea using:

- multi-agent simulation
- a learned future-oriented reward model
- GRPO policy optimization

## Current methodology

The active training stack is in [training/](training/).

### Stage 1: multi-agent simulation

- the policy proposes candidate therapist responses
- `ESCEnv` itself rolls the dialogue forward with hidden long-horizon state
- a separated critic scores future trajectory quality using the env summary

Entry point:

```bash
python -m training.simulate_dialogues ...
```

### Stage 2: reward model

- train a scalar future-oriented reward regressor
- keep the backbone frozen by default
- write audit files for overestimates / underestimates

Entry point:

```bash
python -m training.reward_model ...
```

### Stage 3: GRPO

- optimize the policy using the learned reward model
- add `<think>...</think><response>...</response>` format reward

Entry point:

```bash
accelerate launch -m training.grpo_policy ...
```

## Datasets

Supported training seeds:

- `esconv_hf`
- `extes_hf`
- `extes_jsonl`
- `jsonl`
- `tasks` for ablations only

For submission-quality training, prefer real dialogue prefixes from ESConv or ExTES over handcrafted task seeds.

See:

- [TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md)
- [EXTES_PREPROCESSING_SCHEMA.md](docs/EXTES_PREPROCESSING_SCHEMA.md)
- [RESEARCH_TASK_ANALYSIS.md](docs/RESEARCH_TASK_ANALYSIS.md)

## Quickstart

### 1. Install

```bash
pip install -r requirements-training.txt
```

### 2. Run the OpenEnv server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Generate simulation data

```bash
python -m training.simulate_dialogues \
  --output-dir artifacts/sim_data \
  --examples-source esconv_hf \
  --dataset-name thu-coai/esconv \
  --policy-model Qwen/Qwen2.5-3B-Instruct \
  --critic-model Qwen/Qwen2.5-7B-Instruct
```

### 4. Train the reward model

```bash
python -m training.reward_model \
  --input-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/reward_model
```

### 5. Run GRPO

```bash
accelerate launch -m training.grpo_policy \
  --prompt-jsonl artifacts/sim_data/candidate_rewards.jsonl \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --reward-model-dir artifacts/reward_model \
  --output-dir artifacts/grpo_policy
```

For Colab, see [COLAB_README.md](docs/COLAB_README.md) and [Therapy_Assistant_OpenEnv_Colab.ipynb](docs/Therapy_Assistant_OpenEnv_Colab.ipynb).

## What judges should see

The strongest submission package is:

- one sharp problem statement
- one clean explanation of hidden state and delayed reward
- one before/after demo transcript
- one reward curve
- one eval table
- one short demo video

The right story is not “we built a benchmark.”

The right story is:

> we built a trainable long-horizon therapy assistant environment with
> future-oriented rewards and measurable post-training improvement.

## Deployment

The repo includes:

- OpenEnv-compatible server in [server/app.py](server/app.py)
- Docker packaging via [Dockerfile](Dockerfile)
- local orchestration via [docker-compose.yml](docker-compose.yml)

## Citation

The reward idea is inspired by:

```bibtex
@article{yang2025rlffesc,
  title   = {Towards Open-Ended Emotional Support Conversations in LLMs via
             Reinforcement Learning with Future-Oriented Rewards},
  author  = {Yang, Ting and Chen, Li and Wang, Huimin},
  journal = {arXiv preprint arXiv:2508.12935},
  year    = {2025}
}
```
