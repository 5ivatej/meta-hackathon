# Competition Alignment

## Executive View

Ignoring results for now, the current project is conceptually well aligned with
the competition.

The strongest fit is:

- `Theme #2: Long-Horizon Planning & Instruction Following`
- `Theme #3.2: Personalized Tasks`

The current concept is stronger than a generic emotional-support bot because it
has the exact properties the brief is asking for:

- long trajectories
- sparse and delayed success
- hidden state
- recovery from early mistakes
- state continuity across sessions
- a trainable reward and post-training loop

The project should be positioned as:

> an OpenEnv training environment for long-horizon personalized support, where
> the model must infer hidden user state, preserve rapport across sessions, and
> optimize future emotional outcomes rather than locally pleasant replies.

That is a competition-grade story. The main remaining work is not the idea. It
is making the methodology maximally defensible and maximally legible.

## Alignment With The Brief

### Problem statement

The brief asks for:

- problem statement
- environment design
- agent capabilities
- tasks
- reward model / evaluation logic
- post-training strategy

The current repo has all six in substance:

- problem: long-horizon emotional support is under-modeled by one-turn chat objectives
- environment: [src/env.py](../src/env.py)
- agent capabilities: state inference, pacing, continuity, safety follow-through
- tasks: [src/tasks.py](../src/tasks.py)
- reward logic: [src/grader.py](../src/grader.py), [training/critic.py](../training/critic.py), [training/reward_model.py](../training/reward_model.py)
- post-training strategy: GRPO in [training/grpo_policy.py](../training/grpo_policy.py)

So at the concept level, the project clears the required shape of a strong
submission.

### Minimum submission requirements

Ignoring the results artifacts on purpose, the current codebase aligns well:

- OpenEnv-compatible environment: yes
- Gym-style `reset()`, `step()`, `state()`: yes
- client/server separation: yes
- `openenv.yaml`: yes
- TRL-based training script: yes
- Colab runnable: yes, via [COLAB_README.md](COLAB_README.md) and [Therapy_Assistant_OpenEnv_Colab.ipynb](Therapy_Assistant_OpenEnv_Colab.ipynb)
- HF Space story: yes, via Docker + `server/app.py`

The engineering baseline is strong enough.

## Where The Current Concept Is Strong

### 1. Environment innovation

This is the strongest part of the repo.

Why it scores well:

- hidden user state with trust, distress, openness, reveal progress
- multi-session trajectories instead of flat chats
- continuity and rupture mechanics
- task-specific completion gates
- future-oriented reward instead of pure next-turn quality

This is much closer to a real training environment than to a toy chatbot demo.
It teaches something models are currently weak at: staying useful across a
fragile, partially observed interaction.

### 2. The task is ambitious enough

The brief explicitly rewards ambitious problems. This one qualifies.

It is:

- underexplored as an RL/OpenEnv environment
- socially legible to judges
- technically rich enough to look like a research project
- easy to demo qualitatively

That combination is rare and valuable.

### 3. The current training flow is now coherent

The current methodology is materially better than the earlier benchmark-shaped
version.

Current flow:

1. dataset prefixes from ESConv / ExTES / JSONL
2. candidate policy responses
3. environment-backed rollouts through `ESCEnv`
4. critic scoring of the future trajectory
5. scalar reward-model distillation
6. GRPO on the policy model

That is a clean training story for judges:

- environment produces structured long-horizon behavior
- critic labels the future trajectory
- reward model makes RL feasible
- GRPO closes the loop

### 4. The project has a natural demo narrative

This matters because `30%` of the judging is presentation.

The repo supports a simple story judges can understand quickly:

- the user looks superficially fine
- the model can either rush, drift, or over-advise
- the environment reveals whether trust was earned, whether the real issue surfaced, and whether continuity held

That is a good submission shape.

## Where The Current Concept Still Needs Tightening

These are the highest-value conceptual improvements before focusing on results.

### 1. The environment and training world are aligned, but not fully unified

This is much better than before, but still not perfect.

The training rollout is now grounded in `ESCEnv`, which is correct. But the
initial state still comes from mapping ESConv / ExTES examples into the
environment’s task family via heuristic task assignment in
[training/datasets.py](../training/datasets.py).

That means the current project is:

- not pure offline dataset tuning
- not pure environment-native curriculum
- but a hybrid

That is acceptable, but the winning version should describe this more crisply:

- dataset prefixes are used to diversify openings and emotional contexts
- the authoritative task dynamics come from `ESCEnv`

Suggested improvement:

- formalize the mapping from dataset examples to env tasks
- document the mapping logic as an explicit curriculum layer instead of leaving it as heuristic glue

### 2. The reward story is still split across two layers

Right now there are two reward notions:

- environment-side shaped reward in [src/grader.py](../src/grader.py)
- critic-derived future reward in [training/critic.py](../training/critic.py)

This is workable, but conceptually messy if left unexplained.

Suggested improvement:

- make one statement the repo repeats everywhere:
  - `ESCEnv` provides the task dynamics and shaped trajectory signals
  - the critic defines the future-outcome label used to train the learned reward model
- explicitly describe the environment reward as training-time structure and evaluation context, not the final learned judge

Without that framing, judges may feel there are too many reward authorities.

### 3. The critic is still too thinly specified

The critic currently returns:

- `score`
- `goal_achieved`
- `rationale`

That is compact, but for a competition-winning reward story it would be better
if the critic emitted a more compositional rubric.

Suggested improvement:

- keep the scalar score
- add rubric subfields such as:
  - emotional attunement
  - trust-building / disclosure progress
  - continuity / memory use
  - safety handling
  - forward progress toward stabilization or planning

Why this matters:

- it makes the reward easier to explain
- it makes reward hacking easier to detect
- it gives you stronger training diagnostics later

This is probably the single best conceptual upgrade still available.

### 4. The reward model is better now, but still somewhat generic

The shift from binary classification to scalar regression was the right move.
That said, the current reward model still learns from:

- `prompt`
- `response`
- scalar target

That is better than the old thresholded classifier, but it still compresses a
rich long-horizon judgment into one scalar.

Suggested improvement:

- keep scalar regression
- consider also storing critic sub-scores in the training data
- optionally train the reward model to predict both:
  - total future score
  - auxiliary rubric heads

That would make the pipeline feel more like a serious reward-learning system,
not just a lightweight scorer.

### 5. The current tasks are strong archetypes, but narrow

The built-in tasks are good demos:

- `work_stress_venting`
- `guarded_relationship`
- `crisis_fragile_trust`

But from a competition perspective, three hand-authored arcs are not enough to
carry the whole environment claim by themselves.

This is not a results issue. It is a concept issue.

Suggested improvement:

- position the built-in tasks as canonical environment archetypes
- add a clearer story that ESConv / ExTES prefixes expand the situation space
- optionally add a few more archetypes:
  - shame / self-worth
  - grief / loss
  - academic or career paralysis
  - family obligation conflict

This would strengthen the claim that the environment teaches a general skill,
not only three stories.

### 6. Budget and time should stay as control constraints, not emotional objectives

This is important for conceptual clarity.

The current env uses `cost_budget` and `time_budget`. These are useful for:

- controlling horizon
- stopping runaways
- comparing efficiency

They should not become central therapy objectives in the learned reward.

Suggested improvement:

- state explicitly that budget/time are rollout-control constraints
- keep them weak or absent in the critic’s main emotional score
- use them primarily for termination, curriculum control, and secondary metrics

That preserves the therapy framing and avoids teaching the wrong thing.

## Best-Case Positioning For Winning

If the goal is to maximize winning odds, the project should be framed in this
exact order:

### 1. What problem are we solving?

LLMs fail at long-horizon support because they optimize the next reply rather
than the user’s future trajectory.

### 2. What makes the environment hard?

- hidden state
- delayed success
- fragile trust
- session continuity
- safety-sensitive disclosure

### 3. What is trainable here?

The model can improve on:

- emotional pacing
- when to explore vs reflect vs plan
- when to avoid advice
- when to surface safety support
- how to carry context across sessions

### 4. Why is the reward credible?

Because it is not just local style scoring. It evaluates whether the candidate
response makes the future conversation go better under the environment’s
long-horizon state dynamics.

### 5. Why is this a strong OpenEnv submission?

Because it is not only a simulator. It is a trainable world with:

- state
- reward
- trajectories
- post-training improvement path

That is the competition’s core ask.

## Recommended Modifications Before Final Submission

These are the highest-value changes, ordered by impact on winning odds at the
concept/code-flow level.

### Tier 1

- Make the critic rubric compositional, not only scalar.
- Clarify in docs that dataset prefixes diversify openings, while `ESCEnv` defines the authoritative task dynamics.
- Unify the reward story across `src/grader.py`, `training/critic.py`, and `training/reward_model.py`.

### Tier 2

- Expand the built-in therapy archetypes beyond the current three.
- Add a documented curriculum story: easier trust-building arcs first, harder crisis / fragile-trust arcs later.
- Make the README’s environment section more explicit about partial observability and recovery from early mistakes.

### Tier 3

- Tighten the policy prompt so it references env state clearly without overconstraining surface style.
- Add critic-side schema logging for sub-scores and failure modes.
- Add a short design note explaining why budget/time are control constraints rather than core therapeutic objectives.

## Final Assessment

Ignoring the results package, the current project is already a serious
competition candidate.

Its strongest assets are:

- strong theme fit
- ambitious and legible problem choice
- a real long-horizon environment
- an RL-compatible training loop
- a clear path from environment to reward model to policy optimization

The best remaining work is not to change the core idea. The best remaining work
is to make the methodology more explicit, more compositional, and more clearly
owned by one story:

> train a therapy assistant to optimize future user outcomes inside a
> long-horizon, partially observed OpenEnv world.
