---
title: Emotional Support Conversations (OpenEnv)
emoji: "💬"
sdk: docker
pinned: false
tags:
  - openenv
---

# Emotional Support Conversations - OpenEnv Environment

> An OpenEnv RL environment for evaluating agents on open-ended emotional
> support conversations, with a hybrid immediate + future-oriented reward
> signal inspired by RLFF-ESC (Yang, Chen, Wang, 2025,
> [arXiv:2508.12935](https://arxiv.org/abs/2508.12935)).

## Why this environment

Emotional support is one of the tasks humans most want AI assistants to do
well, and one of the easiest to do badly. Existing dialogue benchmarks often
score turn-level responses in isolation, which rewards agents for sounding
empathetic without testing whether their replies actually move the person
toward resolution. This environment closes that gap.

Three properties make it a genuine RL problem, not a single-shot dialogue
task:

1. Partial observability. The seeker's distress, trust, and willingness to
   reveal their real issue are hidden state. The agent must infer them from
   the conversation so far.
2. Sequential credit assignment. A warm reply at turn 2 can unlock a
   disclosure at turn 6. A single dismissive reply at turn 4 can collapse the
   whole trajectory and require several turns to recover.
3. Exploration vs commitment. Should the agent keep exploring feelings or move
   toward an action plan? Commit too early and the seeker shuts down; explore
   too long and the episode times out.

## Reward design (RLFF-ESC-inspired)

Each step reward is:

```text
step_reward = clip(0.45 * immediate + 0.55 * future_oriented - penalties, 0, 1)
```

- `immediate`: stage-appropriate empathy/validation/open-question fit, plus
  turn-level deltas in the seeker's trust and distress.
- `future_oriented`: a k-step oracle rollout from both the pre- and
  post-action seeker states. The reward is proportional to how much the
  agent's action preserves or advances the attainable resolution ceiling, not
  just how good the current turn looks in isolation.
- `penalties`: dismissive language, premature advice, bare replies,
  interrogation, and repeated template-like responses.

A final task score combines average shaped reward, the seeker's final
resolution state, efficiency, and a completion bonus. Success is hard-gated:
timing out with a generic but non-harmful conversation can still earn partial
score, but it does not count as a solved episode.

## Tasks (3 difficulties)

| Task ID | Difficulty | Max turns | Core challenge |
| --- | --- | ---: | --- |
| `work_stress_venting` | easy | 10 | Cooperative seeker venting about work. Must reach closing with trust >= 0.70 and distress <= 0.40. |
| `guarded_relationship` | medium | 12 | Guarded seeker; real issue is hidden behind the surface concern until openness >= 0.75. Must reveal the true issue and finish in closing with trust >= 0.72 and distress <= 0.45. |
| `crisis_fragile_trust` | hard | 14 | High-distress, fragile trust, multiple interleaved concerns. Must reveal the crisis concern, reference external safety support, and finish in closing with trust >= 0.75 and distress <= 0.40. |

Success thresholds (final score) are `0.60 / 0.62 / 0.65` respectively, and
they are only evaluated after the task-specific completion conditions are met.

## Action and observation space

Action is a free-text reply to the seeker:

```python
class Action(BaseModel):
    message: str
```

Observation is deliberately partial:

```python
class Observation(BaseModel):
    seeker_utterance: str
    turn: int
    remaining_turns: int
    stage_hint: str
    task_id: str
    scenario_brief: str
```

The seeker's internal hidden variables are never exposed.

## Environment internals

The seeker is a deterministic finite-state machine with continuous hidden
variables (`distress`, `trust`, `openness`, `revealed`, `stage`). On each
turn, the agent's reply is analyzed with keyword and regex feature detectors,
then hidden state advances via transparent rules.

Why not use an LLM-driven seeker? The hackathon rubric requires graders to be
deterministic and reproducible. An LLM-driven seeker would risk score variance
between runs. Deterministic dynamics give full reproducibility while still
producing rich, sequential, partially observable dialogue with genuine
recovery-from-mistakes dynamics.

## HTTP API (OpenEnv spec)

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/` | none | health + metadata |
| `GET` | `/tasks` | none | list of tasks |
| `POST` | `/reset` | `{"task_id": "...", "seed": null}` | `ResetResult` |
| `POST` | `/step` | `{"action": {"message": "..."}}` | `StepResult` |
| `GET` | `/state` | none | `EnvState` |

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

## Skills / agents extension

The environment itself stays deterministic and reproducible. To align with the
hackathon's optional skills/agents framing, this repo also includes a
policy-side agentic controller that routes between five reusable skills:
`empathize`, `validate`, `explore`, `plan`, and `safety_escalate`.

This keeps the benchmark honest:

- the environment and grader remain unchanged
- the agentic story lives in the policy, not in a hidden stochastic seeker
- judges can inspect turn-by-turn routing traces in the benchmark outputs

## Benchmarking

### Deterministic local benchmark ladder

Run the built-in rubric ladder and write reusable Markdown/JSON artifacts:

```bash
py -3 benchmark.py
```

Outputs:

- `results/local_benchmarks.md`
- `results/local_benchmarks.json`

### Deterministic skill-routed benchmark

Run the explicit agentic baseline comparison and write route-aware artifacts:

```bash
py -3 benchmark_agentic.py
```

Outputs:

- `results/agentic_benchmarks.md`
- `results/agentic_benchmarks.json`

### LLM benchmark with Markdown output

When you have a real model endpoint and token, run:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=<your-hf-token>
export ESC_ENV_URL=http://localhost:7860
py -3 benchmark_llm.py
```

Outputs:

- `results/llm_benchmark.md`
- `results/llm_benchmark.json`

### Skill-routed LLM benchmark

Use the same environment endpoint, but add the policy-side router and skill
traces around the model:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=<your-hf-token>
export ESC_ENV_URL=http://localhost:7860
py -3 benchmark_agentic_llm.py
```

Outputs:

- `results/agentic_llm_benchmark.md`
- `results/agentic_llm_benchmark.json`

## Baseline scores

Deterministic local numbers below were generated with `py -3 benchmark.py`.
Add real model rows after running `benchmark_llm.py` and
`benchmark_agentic_llm.py`.

### Deterministic baselines

| Baseline | Avg score | Success rate | Notes |
| --- | ---: | ---: | --- |
| `generic_template` | 0.393 | 0.00 | Safe-sounding repeated empathy; no task completion |
| `validation_only` | 0.539 | 0.00 | Better partial reward, still fails hard-gated completion |
| `stage_aware_heuristic` | 0.821 | 1.00 | Task-aware staged policy; completes all 3 tasks |

### Skill-routed agentic baselines

| Baseline | Avg score | Success rate | Notes |
| --- | ---: | ---: | --- |
| `skill_routed_deterministic` | 0.821 | 1.00 | Explicit router over `empathize` / `validate` / `explore` / `plan` / `safety_escalate`; matches the strong staged baseline while exposing route traces |

### Real LLM baseline

| Model | Avg score | Success rate | Report |
| --- | ---: | ---: | --- |
| `Qwen/Qwen2.5-72B-Instruct` (or chosen final model) | TBD | TBD | Run `py -3 benchmark_llm.py` |

### Skill-routed real LLM baseline

| Model | Avg score | Success rate | Report |
| --- | ---: | ---: | --- |
| `Qwen/Qwen2.5-72B-Instruct` (or chosen final model) + router | TBD | TBD | Run `py -3 benchmark_agentic_llm.py` |

## Benchmark narrative

- The generic repeated-empathy template now fails completely under the hardened rubric.
- The deterministic skill-routed policy matches the strong staged heuristic at `0.821` average score and `1.00` success rate, while making turn-level skill choices visible.
- The hard task is only successful when the trajectory reaches a safety-aware finish, so sounding supportive without escalation is not enough.

## Files

```text
.
|-- openenv.yaml             # OpenEnv metadata
|-- Dockerfile               # Container build for HF Space
|-- benchmark.py             # Deterministic local benchmark ladder
|-- benchmark_agentic.py     # Deterministic skill-routed benchmark
|-- benchmark_agentic_llm.py # Skill-routed LLM benchmark
|-- benchmark_llm.py         # LLM benchmark that writes Markdown/JSON
|-- requirements.txt
|-- server.py                # FastAPI HTTP server (entrypoint)
|-- inference.py             # Mandated baseline inference script
|-- SUBMISSION_NEXT_STEPS.md # Manual checklist before final submission
|-- README.md
`-- src/
    |-- __init__.py
    |-- agentic.py           # Skill router + reusable policy-side skills
    |-- baselines.py         # Deterministic baseline policies
    |-- models.py            # Pydantic Action / Observation / Reward / envelopes
    |-- seeker.py            # Deterministic seeker simulator + feature detectors
    |-- tasks.py             # 3 task personas (easy / medium / hard)
    |-- grader.py            # Hybrid immediate + future-oriented reward
    |-- env.py               # Core ESCEnv with step/reset/state
    `-- client.py            # Async HTTP client for inference.py
```

## Citation

If you use this environment, please cite the paper whose reward idea inspired
it:

```bibtex
@article{yang2025rlffesc,
  title   = {Towards Open-Ended Emotional Support Conversations in LLMs via
             Reinforcement Learning with Future-Oriented Rewards},
  author  = {Yang, Ting and Chen, Li and Wang, Huimin},
  journal = {arXiv preprint arXiv:2508.12935},
  year    = {2025}
}
```
