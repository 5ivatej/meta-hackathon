# Technical Detail

This document explains the current project as a trainable personal therapy
assistant environment, not as a benchmark suite.

## 1. Research Goal

The target capability is long-horizon emotional support under partial
observability.

The system should learn to:

- infer hidden user state from dialogue
- build trust before advice
- recover from early conversational mistakes
- preserve continuity across sessions
- handle risk calmly and safely

This is the project’s main claim for the competition.

## 2. Environment

The OpenEnv-compatible environment is implemented in:

- [src/env.py](../src/env.py)
- [src/tasks.py](../src/tasks.py)
- [src/seeker.py](../src/seeker.py)
- [src/grader.py](../src/grader.py)

Key properties:

- multi-session therapy arcs
- hidden trust, distress, and openness
- delayed consequences
- rupture and recovery
- continuity tracking
- cost and time budgets

The environment exposes standard OpenEnv/Gym-style operations:

- `reset()`
- `step()`
- `state()`

and the HTTP server mirrors those at:

- `/reset`
- `/step`
- `/state`
- `/tasks`

## 3. Tasks

The built-in tasks are therapy-style arcs, not evaluation baselines:

- `work_stress_venting`
- `guarded_relationship`
- `crisis_fragile_trust`

Each arc defines:

- the surface concern
- the hidden underlying issue
- reveal threshold
- trust fragility
- session count
- per-session limits
- working goals
- follow-up openers
- success conditions

These tasks are useful for ablations and demos. For training, the preferred
data sources are dataset-derived dialogue prefixes such as ESConv and ExTES.

## 4. Reward Logic

The reward combines four ideas:

- immediate local quality
- future-oriented trajectory quality
- anti-gaming penalties
- long-horizon continuity and session-transition terms

The core design principle is:

> reward responses that improve the future of the conversation, not just the
> current turn.

That makes the task much closer to therapy-style support than to ordinary
response scoring.

## 5. Training Pipeline

The active methodology is implemented in `training/`.

### Stage 1: simulation

Implemented in:

- [training/simulate_dialogues.py](../training/simulate_dialogues.py)
- [training/simulator.py](../training/simulator.py)
- [training/critic.py](../training/critic.py)

Roles:

- policy model
- environment-backed simulator
- critic

Process:

1. Take a real or synthetic dialogue prefix.
2. Sample candidate therapist responses.
3. Roll the dialogue forward through `ESCEnv` and its hidden state.
4. Let the critic judge future trajectory quality from the rollout summary.
5. Save `(context, response, reward)` style records.

### Stage 2: reward model

Implemented in:

- [training/reward_model.py](../training/reward_model.py)

Design:

- scalar future-oriented reward regressor
- frozen backbone by default
- trainable regression head
- audit outputs for overestimates, underestimates, and error summaries

### Stage 3: GRPO

Implemented in:

- [training/grpo_policy.py](../training/grpo_policy.py)

Design:

- optimize the policy model with the learned reward
- add think-format reward
- train with LoRA-friendly settings for practical Colab runs

## 6. Data Sources

Supported sources:

- ESConv via `esconv_hf`
- ExTES via `extes_hf`
- custom JSONL via `jsonl` / `extes_jsonl`
- built-in tasks via `tasks`

Relevant docs:

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- [EXTES_PREPROCESSING_SCHEMA.md](EXTES_PREPROCESSING_SCHEMA.md)
- [RESEARCH_TASK_ANALYSIS.md](RESEARCH_TASK_ANALYSIS.md)

## 7. Deployment and Demo Surface

The environment is deployable as:

- a Hugging Face Space using the Docker image
- a local OpenEnv server
- an interactive browser sandbox from [server/app.py](../server/app.py)

The demo surface is intentionally simple:

- inspect tasks
- start an episode
- send therapist messages
- watch hidden-state effects emerge through dialogue progression and reward

## 8. Competition Story

The strongest competition framing is:

- long-horizon personalized support
- future-oriented rewards
- hidden-state reasoning
- actual post-training improvement

The repo should be read as a training environment for a personal therapy
assistant, not as a generic evaluation package.
