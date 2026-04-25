# Round 2 Alignment Audit

This document is distilled from the Round 2 brief you shared and a local review of the current repo.

## 1. Round 2 Criteria, Guidelines, and Framework

### Themes

| Theme | What judges want |
| --- | --- |
| Multi-Agent Interactions | Cooperation, competition, negotiation, coalition behavior, theory-of-mind reasoning, strategic behavior under partial observability. |
| Long-Horizon Planning & Instruction Following | Deep multi-step reasoning, sparse or delayed rewards, long-running sessions, recovery from early mistakes, memory beyond short context windows. |
| World Modeling: Professional Tasks | Real tool/API/dynamic-system interaction, persistent state, causal reasoning, multi-step workflow execution. |
| World Modeling: Personalized Tasks | Realistic personal assistant tasks, conflicts, delegations, difficult replies, planning around human constraints. |
| Self-Improvement | Self-play, adaptive curricula, challenge generation, recursive capability improvement. |
| Wild Card | A creative environment that still meaningfully helps train LLMs. |

### Judging Criteria

| Criterion | Weight | What it means |
| --- | ---: | --- |
| Environment Innovation | 40% | Novel, challenging, genuinely useful for training. |
| Storytelling & Presentation | 30% | Clear pitch, engaging demo, easy to understand. |
| Showing Improvement in Rewards | 20% | Real evidence that training improved behavior. |
| Reward & Training Pipeline | 10% | Coherent reward logic and a real trainable setup. |

### Minimum Submission Requirements

1. Use OpenEnv, ideally the latest release.
2. Provide a minimal training script using Unsloth or Hugging Face TRL, ideally in Colab.
3. Show actual training evidence: reward/loss plots or before/after behavior.
4. Publish a short writeup or short video and link it from the README.
5. Host the environment on Hugging Face Spaces.
6. Include a README that explains the problem, the environment, and the results.

### Engineering / Framework Checklist

From the judging notes, the expected implementation shape is:

1. Use OpenEnv `Environment` or `MCPEnvironment` patterns properly.
2. Respect client/server separation.
3. Expose standard Gym-style `reset`, `step`, and `state`.
4. Include a valid `openenv.yaml`.
5. Avoid reserved MCP tool names for custom tools.
6. Keep grading deterministic and reproducible.

## 2. What the Current Round 1 Repo Is

Current repo summary:

- A deterministic OpenEnv-style environment for online therapist / therapy-style support conversations.
- A hidden-state seeker simulator with `trust`, `distress`, `openness`, `revealed`, and stage progression.
- Dense shaped reward plus future-oriented reward lookahead.
- Three tasks with escalating difficulty.
- A FastAPI server with `/reset`, `/step`, `/state`, `/tasks`.
- Local benchmark scripts and benchmark artifacts.
- A policy-side agentic baseline with skills such as `empathize`, `validate`, `explore`, `plan`, and `safety_escalate`.

Key files reviewed:

- `README.md`
- `openenv.yaml`
- `src/env.py`
- `src/grader.py`
- `src/seeker.py`
- `src/tasks.py`
- `server/app.py`
- `benchmark.py`
- `benchmark_agentic.py`
- `results/*.md`

## 3. Best Round 2 Theme Fit

### Strongest Fit: Theme 3.2 Personalized Tasks

This repo aligns best with **World Modeling -> Personalized Tasks**.

Why:

- The task is directly about handling emotionally difficult personal interactions in a therapist-like setting.
- The environment is partially observable and requires maintaining hidden beliefs about the user.
- The agent must balance empathy, disclosure, safety escalation, and therapeutic pacing.
- The task is realistic enough to pitch as a therapy-support niche rather than a toy benchmark.

Scope boundary:

- This fit does **not** depend on adding external tools.
- You can still strengthen the submission substantially by adding cross-session memory, longer horizons, and persistence without adding tool use.

### Strong Secondary Fit: Theme 2 Long-Horizon Planning

There is a credible Theme 2 path because:

- mistakes have delayed effects
- the reward is sequential
- the agent must manage stage transitions over multiple turns
- the hidden state already provides a foundation for longer-horizon state tracking

But it is still weak for Round 2 Theme 2 because:

- episodes are only 10 to 14 turns
- there is no cross-session memory
- there is no beyond-context persistence problem
- there is no truly long-running workflow

These should now be treated as **current gaps to build**, not fixed product boundaries.

If you add:

- cross-session memory
- longer episode structures
- persistent state across sessions
- delayed rewards across longer therapeutic trajectories

then the project becomes a real **Theme 2 + Theme 3.2 hybrid** even without tool calling.

### Poor Fit for the Other Themes

| Theme | Alignment | Why |
| --- | ---: | --- |
| Multi-Agent Interactions | 20/100 | Only one active agent; the seeker is a simulator, not a strategic second agent. |
| Long-Horizon Planning | 35/100 today, higher after extension | Sequential and partially observable already, with a clear upgrade path to persistent multi-session planning. |
| World Modeling: Professional Tasks | 15/100 | No real external tools, APIs, or enterprise workflow loop. |
| World Modeling: Personalized Tasks | 82/100 | Strong direct fit. |
| Self-Improvement | 10/100 | No self-play, curriculum generation, or adaptive challenge creation. |

## 4. Recommended Round 2 Problem Statement

If you keep this codebase, the cleanest Round 2 framing is:

**Problem statement:** Train an LLM agent for online therapist-style conversations under partial observability, where success depends on inferring hidden emotional state, building trust, choosing the right moment to validate vs. explore vs. suggest a next step, and escalating to real-world safety support when needed.

This is much stronger than pitching it as a generic chat benchmark.

One important framing note:

- For the submission, it is better to say **therapist-style assistant**, **online therapy simulator**, or **therapy-support agent** than to imply the model is a licensed therapist.
- That keeps the story strong while avoiding an unsafe or overclaimed product framing.

## 5. Alignment Against Round 2 Minimum Requirements

| Requirement | Status | Evidence in repo | Gap |
| --- | --- | --- | --- |
| OpenEnv environment | Partial | `openenv.yaml`, OpenEnv-style API, `openenv-core` dependency | I did not find usage of OpenEnv base classes; latest release is not verified locally. |
| Minimal training script with Unsloth or HF TRL | Missing | No training notebook/script found | Must add a real training script. |
| Real training evidence | Missing | Only benchmark reports exist in `results/` | Need reward/loss curves or before/after trained policy evidence. |
| Hugging Face Space hosting | Partial | Dockerized app and Space-style README front matter | Hosting URL is not present or verified. |
| Short writeup/video linked from README | Missing | No blog/video links found | Must add links. |
| README with problem, environment, results | Partial | README explains the environment well | Needs Round 2 framing, training results, Space URL, and external links. |

## 6. Alignment Against the Engineering Framework

| Framework expectation | Status | Notes |
| --- | --- | --- |
| Deterministic grading | Met | Strong point of the repo. |
| Client/server separation | Mostly met | `src/client.py` and `server/app.py` are separated. |
| `reset` / `step` / `state` API | Met | Present in `src/env.py` and server routes. |
| Valid `openenv.yaml` | Met | Present and reasonably complete. |
| OpenEnv base class usage | Not evident | Current env is a custom `ESCEnv`, not an obvious OpenEnv subclass. |
| No reserved MCP tool misuse | Met / not applicable | No custom MCP tool layer found. |

## 7. How Much Round 1 Aligns With Round 2

There are two different answers:

### A. Theme Alignment

If you pitch this as **Theme 3.2 Personalized Tasks**, the core environment already aligns fairly well.

**Theme-fit score: 72/100**

Why it scores reasonably well:

- realistic therapy-support niche
- partial observability
- nontrivial reward shaping
- meaningful failure modes
- visible skill decomposition

Why it is not higher:

- no real tool use
- no longer-term memory across sessions
- limited horizon
- still a scripted simulator rather than a richer personal world

These are acceptable constraints for Theme 3.2, but they cap how ambitious the environment feels.

### A2. Planned Extended Alignment

If you implement the extensions you described:

- cross-session memory
- longer horizons
- beyond-context persistence
- longer-running therapeutic workflows

then the project shifts from a narrow Theme 3.2 submission to a stronger **Theme 2 + Theme 3.2** submission.

Projected alignment after those changes:

| Theme | Projected alignment | Why |
| --- | ---: | --- |
| Long-Horizon Planning | 75/100 | Persistent memory and long-running therapeutic trajectories directly address the theme. |
| Personalized Tasks | 85/100 | The therapy niche remains strong and becomes more realistic. |

### B. Submission Readiness Against Round 2 Judging

If judges reviewed the repo in its current state as a Round 2 submission, the score would be much lower because the mandatory training proof is missing.

#### Estimated judging-readiness score

| Criterion | Weight | Current estimate | Reason |
| --- | ---: | ---: | --- |
| Environment Innovation | 40 | 25 | Good personalized benchmark, but not obviously frontier-pushing yet. |
| Storytelling & Presentation | 30 | 19 | README is clear, but not yet framed as a Round 2 story with final submission assets. |
| Showing Improvement in Rewards | 20 | 3 | Benchmarks exist, but no real training improvement evidence. |
| Reward & Training Pipeline | 10 | 5 | Reward logic is strong; training pipeline is missing. |
| Total | 100 | 52 | Respectable environment, incomplete Round 2 submission. |

**Estimated current Round 2 readiness: 52/100**

## 8. Current State Gaps

The current repo should explicitly present these as **current-state gaps**, not permanent boundaries:

- no real tool use
- no longer-term memory across sessions
- limited horizon
- still a scripted simulator rather than a richer personal world

Why this is still defensible:

- it keeps grading deterministic and reproducible
- it makes training easier to run and compare
- it keeps the benchmark focused on therapist-style conversational decision-making

What this means strategically:

- do not pitch it as a professional workflow environment
- do not pitch it as a long-horizon memory benchmark until those features are built
- do not promise a rich personal world simulator that does not exist
- do pitch it as a focused personalized conversation environment with hidden state and safety-sensitive timing
- and once the planned extensions exist, re-pitch it as a Theme 2 + Theme 3.2 hybrid

## 9. What Is Already Strong

- The environment is not a toy gridworld.
- Hidden state makes the task genuinely partially observable.
- The reward logic is thoughtful and denser than a binary success metric.
- The repo already has benchmark scripts and reusable result artifacts.
- The online therapist niche is stronger and more memorable than a generic chatbot demo.
- The deterministic design is judge-friendly and reproducible.

## 10. Main Gaps Blocking a Strong Round 2 Submission

1. No actual TRL or Unsloth training script.
2. No training curves or before/after trained-policy evidence.
3. No verified Hugging Face Space URL in the README.
4. No short blog/video link in the README.
5. No explicit Round 2 problem statement and story.
6. No proof that the implementation uses the latest OpenEnv release or official base classes.
7. The current task is only moderately sequential, not truly long-running.

## 11. Recommendation

Do **not** throw away the repo.

The practical move is:

1. Keep the current environment core and reward design, and make the niche explicitly **online therapist / therapy-style support**.
2. Skip tool-calling features.
3. Expand into cross-session memory, longer horizons, and persistent therapeutic workflows.
4. Reframe the submission as a **Theme 2 + Theme 3.2** story once those extensions exist.
5. Add a minimal real training pipeline with TRL or Unsloth.
6. Produce one clear before/after comparison with plots.
7. Update the README to tell the Round 2 story around therapist-style interaction, hidden state, memory, and safety-aware escalation.

## 12. Short Verdict

**Round 1 codebase fit today:** good if framed as Personalized Tasks with an online therapist niche.  
**Round 1 codebase fit after your planned extensions:** strong Theme 2 + Theme 3.2 hybrid, even without tool calling.  
**Round 1 codebase fit to Round 2 submission requirements:** incomplete.  
**Best interpretation:** keep the therapist niche, skip tool calling, add memory and longer-horizon structure, then add training proof and package it properly.
