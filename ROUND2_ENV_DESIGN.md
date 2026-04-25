# Round 2 Environment Design

This document translates the Round 2 direction into concrete environment changes for this codebase.

## 1. Design Goal

Extend the current therapist-style environment from a short single-session benchmark into a **multi-session, long-horizon therapy environment** with:

- persistent hidden state across sessions
- memory beyond a single context window
- delayed consequences from early mistakes
- long-horizon reward shaping
- durable pause/resume execution
- budget-aware control and safety guardrails
- no external tool calling

This should support a stronger **Theme 2 + Theme 3.2** submission.

## 2. Core Product Idea

The agent is not solving one conversation. It is managing a **therapy arc** over multiple sessions.

Each episode should become a structured care trajectory such as:

1. intake / initial trust building
2. disclosure and problem formulation
3. deeper exploration and pattern discovery
4. coping-plan formation
5. follow-up, relapse handling, or repair after mistakes

The important shift is:

- current env: `one short conversation`
- target env: `a sequence of linked therapy sessions`

The infrastructure shift is also important:

- current runner: short-lived, process-local memory
- target runner: persistent memory, resumable execution, restart-safe episodes

## 3. New Episode Model

### Current Model

- 1 episode
- 1 task
- 10 to 14 turns
- success mostly judged within that local window

### Target Model

- 1 episode = `N sessions`
- each session has its own turn budget
- hidden state persists across sessions
- memory summaries survive session boundaries
- final success depends on cumulative progress, not only the last exchange

### Recommended Structure

Use:

- `3 to 5 sessions` per episode
- `6 to 10 turns` per session

This gives enough horizon without making the implementation unmanageable.

Also add:

- per-session time budget
- per-episode cost/token budget

## 4. Session State Model

Add explicit session-level state on top of the current hidden seeker variables.

### Keep Existing Hidden Variables

- `trust`
- `distress`
- `openness`
- `revealed`
- `stage`

### Add New Persistent Variables

- `session_index`
- `sessions_total`
- `alliance_strength`
- `stability`
- `adherence`
- `rupture_count`
- `safety_risk_level`
- `working_goals`
- `active_coping_plan`
- `memory_summary`
- `agent_memory_state`
- `recent_breakthrough`
- `unfinished_threads`
- `episode_budget_spent`
- `episode_time_spent`
- `resume_checkpoint_id`

### Meaning of New Variables

| Variable | Meaning |
| --- | --- |
| `alliance_strength` | Longer-term therapeutic relationship quality across sessions. |
| `stability` | Whether the user is trending toward regulation or deterioration over time. |
| `adherence` | Whether the user follows through on agreed coping actions between sessions. |
| `rupture_count` | Number of major trust ruptures caused by poor agent behavior. |
| `safety_risk_level` | Ongoing safety severity, not just one-turn crisis detection. |
| `working_goals` | The current therapy targets identified so far. |
| `active_coping_plan` | The plan the agent and seeker have developed together. |
| `memory_summary` | Compact cross-session memory available to the policy. |
| `agent_memory_state` | Durable policy-side memory snapshot persisted outside process memory. |
| `recent_breakthrough` | Important progress moment to preserve across sessions. |
| `unfinished_threads` | Open issues that should be revisited later. |
| `episode_budget_spent` | Accumulated cost/token spend across the long-running episode. |
| `episode_time_spent` | Accumulated time spent across sessions. |
| `resume_checkpoint_id` | Runner checkpoint used for pause/resume and restart recovery. |

## 5. Observation Design

The observation should remain partially observable, but it now needs explicit long-horizon structure.

### Recommended Observation Fields

- `seeker_utterance`
- `turn`
- `remaining_turns`
- `session_index`
- `sessions_total`
- `stage_hint`
- `task_id`
- `scenario_brief`
- `memory_summary`
- `last_session_outcome`
- `current_goal_hint`

### Important Constraint

Do **not** expose raw hidden state such as exact trust/distress numbers.

The observation should give the agent:

- enough persistent structure to act coherently across sessions
- not enough internal state to trivialize the task

## 6. Memory Design

This is the main Theme 2 feature.

### Policy-Facing Memory

At the start of each session, expose a compact summary such as:

- what the seeker disclosed so far
- what seemed to help
- what caused friction
- current therapy goals
- whether there was a prior coping commitment
- whether safety concerns are active
- what budget/time constraints are still available if relevant

### Environment-Facing Memory

Internally store:

- prior session events
- unresolved disclosures
- progress markers
- previous mistakes by the agent
- promised follow-up topics
- summary checkpoints for restart recovery
- cumulative budget/time usage

### Memory Rules

1. Good summaries should help future performance.
2. Dropping an important thread should be penalized.
3. False continuity should be penalized.
4. Remembering the wrong thing should be worse than admitting uncertainty.
5. Memory must survive pauses and worker restarts.

## 7. Prompting Architecture Change

The current LLM path should move away from short sliding-window prompting.

### Current Pattern To Replace

Today, `inference.py` builds the prompt from the recent exchange only, using the last 8 history items.

That is not enough for a Theme 2 claim.

### Target Pattern

Build prompts from:

- persistent memory summary
- unresolved commitments
- current-session recent turns
- safety status
- budget status

### Recommended Prompt Layout

1. therapy arc summary
2. current goals and unresolved threads
3. safety / risk status
4. recent local turns
5. current seeker message
6. deterministic draft reply

This preserves local fluency while making the policy depend on durable memory instead of raw transcript length.
## 8. Task Redesign

Replace the current single-session tasks with multi-session therapy trajectories.

### Recommended Task Ladder

#### 1. Burnout Recovery Arc

- starts as workplace stress
- evolves into burnout, avoidance, identity strain
- requires trust-building, goal-setting, and follow-up

#### 2. Guarded Relationship Arc

- starts vague and defensive
- true issue emerges slowly across sessions
- requires continuity, non-pushy exploration, and later repair planning

#### 3. Crisis Stabilization Arc

- starts dysregulated
- requires immediate safety-sensitive pacing
- later sessions test whether the agent remembers risk context and follows up responsibly

### Why this works

It preserves the current personas but makes them feel more realistic over time.

## 9. Stage Model

The current stage model is still useful, but should become hierarchical.

### Within-Session Stages

- `opening`
- `exploring`
- `reflecting`
- `planning`
- `closing`

### Across-Session Arc Stages

- `intake`
- `formulation`
- `stabilization`
- `skills_building`
- `follow_up`

This lets the environment judge both local turn quality and global therapy progress.

## 10. Reward Design Changes

The reward should now combine:

1. turn quality
2. session quality
3. cross-session consistency
4. long-horizon progress
5. safety-aware behavior

### Recommended Step Reward Components

Keep:

- empathy / validation / timing quality
- trust delta
- distress delta
- anti-repetition penalty

Add:

- memory continuity reward
- follow-up correctness reward
- rupture repair reward
- plan adherence support reward
- delayed stability reward
- budget discipline reward
- restart continuity reward

### Recommended Session-End Reward

At session end, score:

- whether the right threads were carried forward
- whether the session advanced the therapy arc
- whether the agent handled safety correctly
- whether the agent strengthened or damaged alliance
- whether the memory summary is adequate for the next session
- whether time/cost remained within acceptable bounds

### Recommended Episode-End Reward

At final episode end, score:

- alliance strength
- symptom/stability improvement
- durable goal progress
- safety handling
- consistency across sessions
- recovery from earlier mistakes
- budget efficiency
- robustness across pause/resume boundaries

### High-Level Formula

```text
total_reward =
  turn_reward
  + session_transition_bonus
  + memory_continuity_bonus
  + long_horizon_progress_bonus
  + budget_efficiency_bonus
  - rupture_penalties
  - false_memory_penalties
  - drift_penalties
  - repetition_penalties
  - runaway_budget_penalties
```

## 11. Durable Runner Design

Long-horizon episodes need a real runner, not only an in-memory loop.

### Required Properties

- pause episode
- resume episode
- survive worker restart
- persist agent memory separately from environment state
- checkpoint after each turn or session

### Recommended Runner State

- `episode_id`
- `task_id`
- `session_index`
- `turn`
- `env_snapshot`
- `agent_memory_snapshot`
- `rolling_summary`
- `cumulative_reward`
- `budget_spent`
- `time_spent`
- `status`

### Persistence Strategy

Mirror the environment-state persistence pattern already used by the server, but do it for policy memory and runner state as well.

Possible storage options:

- signed cookie for small state
- file/blob store for larger state
- lightweight database record for resumable episodes

## 12. Failure Modes To Explicitly Model

These are important because they make the long horizon matter.

### Failure Types

- forgetting an important prior disclosure
- giving advice before trust is built
- failing to revisit a prior commitment
- acting as if a safety concern disappeared when it did not
- generic empathy with no cumulative progress
- pushing too hard after a rupture instead of repairing
- losing continuity after a runner restart
- burning budget with repetitive low-value turns
- drifting away from active therapy goals

### Desired Training Signal

The environment should make these failures expensive across later sessions, not only immediately.

## 13. Success Criteria

A successful episode should require more than sounding good.

### Suggested Success Gates

1. seeker reaches acceptable final stability
2. alliance strength exceeds threshold
3. one or more working goals are advanced
4. key disclosures are handled consistently across sessions
5. safety obligations are completed when relevant
6. continuity survives pause/resume boundaries
7. budget usage stays within threshold

This prevents shallow local optimization.

## 14. API / Model Changes

Likely model additions:

### `src/models.py`

Add fields for:

- session metadata
- memory summary
- last session outcome
- cross-session final metrics
- budget/time metrics
- runner checkpoint metadata

### `src/env.py`

Add support for:

- session transitions
- persistent state storage inside the episode
- memory summary generation
- session-end and episode-end scoring
- pause/resume snapshots

### `src/seeker.py`

Add:

- long-horizon seeker state
- between-session progression rules
- relapse / adherence / stability dynamics
- budget-sensitive deterioration or frustration dynamics if sessions drift

### `src/tasks.py`

Replace or extend current tasks with:

- multi-session arc configs
- session templates
- long-horizon success thresholds

### `src/grader.py`

Extend grading with:

- memory continuity scoring
- session transition scoring
- delayed outcome scoring
- rupture repair scoring
- budget efficiency scoring
- restart continuity scoring

## 15. File-by-File Implementation Plan

### Phase 1: Data Model

Files:

- `src/models.py`
- `src/tasks.py`

Changes:

- add session-aware observation/result fields
- add multi-session task config

### Phase 2: Persistent State

Files:

- `src/seeker.py`
- `src/env.py`
- `src/agentic.py`
- `server/app.py`

Changes:

- add cross-session seeker state
- add per-session reset logic inside one episode
- add memory summary generation
- persist `AgentMemory`
- add runner/checkpoint persistence

### Phase 3: Reward Logic

Files:

- `src/grader.py`
- `src/env.py`

Changes:

- add memory continuity reward
- add long-horizon progress reward
- add session transition scoring
- add budget and drift penalties
- add resume continuity scoring

### Phase 4: Benchmarks

Files:

- `benchmark.py`
- `benchmark_agentic.py`
- `benchmark_llm.py`
- `benchmark_agentic_llm.py`
- `inference.py`

Changes:

- report session-level and episode-level outcomes
- add memory consistency and follow-up metrics
- replace short-window prompt construction
- report budget and restart-resilience metrics

### Phase 5: README / Submission Story

Files:

- `README.md`
- `openenv.yaml`

Changes:

- explain multi-session design
- explain persistent memory
- show before/after training behavior

## 14. Minimal Viable Version

If time is tight, do this minimal version:

1. convert each task into `3 sessions`
2. persist `trust`, `distress`, `openness`, `revealed`
3. add `memory_summary`
4. add one continuity reward
5. add one penalty for forgetting a key disclosure
6. add session-end evaluation
7. replace `last 8 turns` prompting with rolling summary + recent turns
8. persist `AgentMemory`
9. add one budget penalty

That is enough to make a credible Theme 2 claim.

## 15. Recommended Implementation Order

1. define new task schema in `src/tasks.py`
2. extend seeker state in `src/seeker.py`
3. update observation/result models in `src/models.py`
4. replace short-window prompting in `inference.py` and related LLM runners
5. persist `AgentMemory` and add runner durability
6. refactor `src/env.py` for multi-session episodes
7. extend `src/grader.py`
8. update benchmarks
9. update README and training pipeline

## 16. Short Engineering Verdict

You do **not** need tool calling to make this strong.

The strongest move is:

- keep the therapist niche
- add persistent memory
- replace short prompt windows with rolling summaries
- make execution durable across pauses and restarts
- make the benchmark multi-session
- make early mistakes matter later
- add budget and safety guardrails for long loops
- train against that longer-horizon structure
