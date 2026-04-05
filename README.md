---
title: Emotional Support Conversations (OpenEnv)
emoji: 💬
sdk: docker
pinned: false
tags:
  - openenv
---

# Emotional Support Conversations — OpenEnv Environment

> An OpenEnv RL environment for evaluating agents on **open-ended emotional
> support conversations**, with a hybrid immediate + future-oriented reward
> signal inspired by **RLFF-ESC** (Yang, Chen, Wang, 2025,
> [arXiv:2508.12935](https://arxiv.org/abs/2508.12935)).

## Why this environment

Emotional support is one of the tasks humans most want AI assistants to do
well — and one of the easiest to do badly. Existing dialogue benchmarks
score turn-level responses in isolation, which rewards agents for *sounding*
empathetic without ever testing whether their replies actually move the
person toward resolution. This environment closes that gap.

Three properties make it a genuine RL problem, not a single-shot dialogue
task:

1. **Partial observability.** The seeker's distress, trust, and willingness
   to reveal their real issue are hidden state. The agent must infer them
   from the conversation so far.
2. **Sequential credit assignment.** A warm reply at turn 2 can unlock a
   disclosure at turn 6. A single dismissive reply at turn 4 can collapse
   the whole trajectory and require several turns to recover.
3. **Exploration vs commitment.** Should the agent keep exploring feelings
   or move toward an action plan? Commit too early and the seeker shuts
   down; explore too long and the episode times out.

## Reward design (RLFF-ESC-inspired)

Each step reward is:

```
step_reward = clip( 0.45 · immediate  +  0.55 · future_oriented  −  penalties , 0, 1 )
```

- **`immediate`** — stage-appropriate empathy/validation/open-question fit,
  plus turn-level deltas in the seeker's trust and distress.
- **`future_oriented`** — a k-step oracle rollout from both the pre- and
  post-action seeker states. The reward is proportional to how much the
  agent's action *preserves or advances the attainable resolution ceiling*,
  not just how good the current turn looks in isolation. This is the
  RLFF-ESC idea: reward signals propagated from projected trajectories
  rather than pointwise turn critique.
- **`penalties`** — dismissive language, premature advice (before trust is
  established), bare replies, interrogation.

A final task score combines average shaped reward, the seeker's final
resolution state, and efficiency (finishing within turn budget).

## Tasks (3 difficulties)

| Task ID                  | Difficulty | Max turns | Core challenge                                                               |
| ------------------------ | ---------- | --------- | ---------------------------------------------------------------------------- |
| `work_stress_venting`    | easy       | 10        | Cooperative seeker venting about work. Reach closing with trust ≥ 0.7.       |
| `guarded_relationship`   | medium     | 12        | Guarded seeker; real issue hidden behind surface concern until openness ≥ 0.75. Premature advice heavily punished. |
| `crisis_fragile_trust`   | hard       | 14        | High-distress, fragile trust, multiple interleaved concerns. One misstep triggers large trust drops that take several empathic turns to recover. Safety referencing rewarded in closing stage. |

Success thresholds (final score) are `0.55 / 0.50 / 0.45` respectively.

## Action & observation space

**Action** — free-text reply to the seeker:

```python
class Action(BaseModel):
    message: str
```

**Observation** — what the agent sees each turn (deliberately partial):

```python
class Observation(BaseModel):
    seeker_utterance: str
    turn: int
    remaining_turns: int
    stage_hint: str        # 'opening' | 'exploring' | 'reflecting' | 'planning' | 'closing'
    task_id: str
    scenario_brief: str
```

The seeker's internal hidden variables (distress, trust, openness, true
issue) are **never** exposed. This is by design.

## Environment internals (why deterministic)

The seeker is a deterministic finite-state machine with continuous hidden
variables (`distress`, `trust`, `openness`, `revealed`, `stage`). On each
turn, the agent's reply is analysed with keyword/regex feature detectors
(empathy markers, validation, open vs closed questions, advice, dismissive
language, interrogation) and hidden state advances via transparent rules.

**Why not use an LLM-driven seeker?** The hackathon rubric requires
graders to be *deterministic and reproducible* — an LLM-driven seeker
would fail the Phase 2 score-variance check. Deterministic dynamics give
us full reproducibility while still producing rich, sequential, partially
observable dialogue with genuine recovery-from-mistakes dynamics.

## HTTP API (OpenEnv spec)

| Method | Path      | Body                                         | Returns                                 |
| ------ | --------- | -------------------------------------------- | --------------------------------------- |
| GET    | `/`       | —                                            | health + metadata                       |
| GET    | `/tasks`  | —                                            | list of tasks                           |
| POST   | `/reset`  | `{"task_id": "...", "seed": null}` (opt.)    | `ResetResult` (observation + info)      |
| POST   | `/step`   | `{"action": {"message": "..."}}`             | `StepResult` (obs, reward, done, info)  |
| GET    | `/state`  | —                                            | `EnvState` (public state + transcript)  |

## Running locally

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Start the environment server
uvicorn server:app --host 0.0.0.0 --port 7860

# 3. In another shell, run the baseline inference
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=<your-hf-token>
export ESC_ENV_URL=http://localhost:7860
python inference.py
```

## Running via Docker

```bash
docker build -t esc-openenv .
docker run -p 7860:7860 esc-openenv
```

## Baseline scores

Replace with your own numbers after running `inference.py` against your
configured endpoint.

| Task                    | Score  | Success |
| ----------------------- | ------ | ------- |
| work_stress_venting     | TBD    | TBD     |
| guarded_relationship    | TBD    | TBD     |
| crisis_fragile_trust    | TBD    | TBD     |
| **Average**             | **TBD** |         |

## Files

```
.
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # Container build for HF Space
├── requirements.txt
├── server.py                # FastAPI HTTP server (entrypoint)
├── inference.py             # Mandated baseline inference script
├── README.md
└── src/
    ├── __init__.py
    ├── models.py            # Pydantic Action / Observation / Reward / envelopes
    ├── seeker.py            # Deterministic seeker simulator + feature detectors
    ├── tasks.py             # 3 task personas (easy / medium / hard)
    ├── grader.py            # Hybrid immediate + future-oriented reward
    ├── env.py               # Core ESCEnv with step/reset/state
    └── client.py            # Async HTTP client for inference.py
```

## Citation

If you use this environment, please cite the paper whose reward idea
inspired it:

```bibtex
@article{yang2025rlffesc,
  title   = {Towards Open-Ended Emotional Support Conversations in LLMs via
             Reinforcement Learning with Future-Oriented Rewards},
  author  = {Yang, Ting and Chen, Li and Wang, Huimin},
  journal = {arXiv preprint arXiv:2508.12935},
  year    = {2025}
}
```
