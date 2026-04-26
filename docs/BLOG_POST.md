# Building a Long-Horizon Distress AI Coach with Future-Oriented Rewards

## Why We Built This

Most emotional-support assistants are optimized for the wrong target.

They are usually trained to produce a good-looking answer for the current turn:

- sound empathetic
- avoid obviously unsafe language
- give a polished reply

That is not enough for distress support.

Real emotional support is a long-horizon problem. A reply that looks good in one screenshot can still be the wrong move if it:

- pushes advice too early
- reduces trust
- misses the real issue
- prevents deeper disclosure
- escalates too aggressively
- wastes limited conversational budget

We built this project around a different objective:

> optimize for what happens next, not just for how nice the current answer sounds.

This `round2` branch is our attempt to turn that idea into a trainable system.

## What We Built

We built an OpenEnv-compatible environment and training stack for a personal distress-support / therapy-style assistant.

At a high level, the project includes:

- a long-horizon emotional-support environment
- hidden user state and delayed consequences
- multi-session therapy-like task arcs
- simulation-driven reward data generation
- a learned future-oriented reward model
- GRPO-based policy optimization
- a Hugging Face deployment path for inference

The goal is not just to benchmark whether a model can produce a kind response.

The goal is to train a model to manage an unfolding emotional process:

- infer the hidden issue
- build trust
- pace the conversation correctly
- preserve continuity
- decide when to explore vs reflect vs guide vs escalate for safety

## The Core Product Idea

The product we are building is a distress AI coach that behaves more like a careful long-horizon support system than a one-shot empathy generator.

In practice, that means the assistant should learn to:

- recognize when the user is only venting versus asking for direction
- avoid over-solving too early
- maintain fragile trust across turns
- adapt to what has and has not been disclosed
- recover after weak turns when recovery is still possible
- move toward better future emotional outcomes

That last point is the center of the branch.

We are not only teaching the model to sound supportive now.
We are trying to teach it to make the future conversation better.

## Why Long-Horizon Support is Hard

Distress support is one of the clearest examples of partial observability.

The model does not directly know:

- the user’s true trust level
- how overwhelmed they are
- whether they are close to disclosure
- whether they are shutting down
- whether a suggestion will feel grounding or invalidating

A user may begin by talking about work, but the real issue may be:

- burnout
- shame
- conflict at home
- fear of failure
- relationship instability
- crisis-adjacent emotional fragility

If the assistant is rewarded only for local style, it can learn a shallow strategy:

- validate
- gently advise
- repeat supportive language

That often looks good in isolation and still fails over time.

The long-horizon framing forces the assistant to care about trajectory quality:

- does trust rise or fall?
- does the user open up?
- does the model create a productive next turn?
- does it avoid damaging the relationship?
- does it steer the arc toward a realistic next step?

## Environment Design

The environment lives in the OpenEnv-compatible stack implemented in:

- `src/env.py`
- `src/tasks.py`
- `src/seeker.py`
- `src/grader.py`
- `server/app.py`

The environment exposes a simple action/observation loop, but internally tracks much richer hidden state.

### Observable Surface

The model sees a public observation like:

- current seeker utterance
- turn index
- remaining turns
- stage hint
- task ID
- scenario brief

### Hidden State

Internally, the environment tracks long-horizon state such as:

- trust
- distress
- openness
- reveal progress
- session continuity
- working memory summary
- budget usage
- task stage progression

This hidden-state structure is what makes the problem meaningful. The model must infer latent dynamics from the interaction itself.

## Task Design

The branch includes built-in therapy-style arcs with distinct failure modes.

### `work_stress_venting`

This tests whether the assistant can:

- avoid empty generic comfort
- build alliance
- identify burnout or overload patterns
- move toward one realistic recovery step

### `guarded_relationship`

This tests whether the assistant can:

- avoid pushing too hard too early
- earn trust before the real issue is stated
- notice guarded disclosure patterns
- help the user name the issue carefully

### `crisis_fragile_trust`

This is the hardest setting and tests whether the assistant can:

- preserve fragile trust
- handle risk without becoming robotic
- stay grounding and emotionally precise
- introduce real-world safety support when necessary

These are not just topic labels. They are long-horizon control problems.

## Reward Philosophy

The project is inspired by the idea that emotional-support training should care about future trajectory quality rather than local stylistic preference alone.

So the reward structure combines:

- immediate conversational quality
- future-oriented trajectory value
- anti-gaming penalties
- long-horizon continuity terms

In plain English:

- a response should not only sound supportive
- it should lead to a better next state
- it should preserve the relationship
- it should keep the conversation emotionally safe

## Training Pipeline

The active training pipeline is organized into three stages inside `training/`.

### Stage 1: Simulation Data Generation

Entry point:

- `python -m training.simulate_dialogues`

This stage generates candidate responses and evaluates them through environment-backed future rollouts.

What happens here:

1. a policy model proposes candidate assistant replies
2. the environment rolls the conversation forward
3. a critic scores the future trajectory
4. the system records candidate-response examples with scalar future-oriented rewards

Artifacts produced:

- `candidate_rewards.jsonl`
- `trajectories.jsonl`

This stage is the bridge between the environment and learnable reward supervision.

### Stage 2: Reward Model Training

Entry point:

- `python -m training.reward_model`

This stage trains a scalar reward regressor over the simulated candidate data.

The reward model learns to estimate:

> given this prompt context and candidate response, how good is the likely future trajectory?

Artifacts produced:

- trained reward model directory
- `reward_model_audit.jsonl`
- `reward_model_audit_summary.json`
- reward metadata and tokenizer/model files

### Stage 3: GRPO Policy Optimization

Entry point:

- `accelerate launch -m training.grpo_policy`

This stage uses:

- the learned reward model
- a think/response formatting reward
- GRPO optimization

The result is a policy adapter trained to improve future-oriented support behavior under the environment’s structure.

Artifacts produced:

- adapter weights
- adapter config
- tokenizer metadata
- GRPO checkpoints

## Datasets and Seed Sources

The branch supports multiple ways of seeding conversations:

- `tasks`
- `jsonl`
- `esconv_hf`
- `extes_hf`
- `extes_jsonl`

For realism, the intended path is to use public emotional-support dialogue sources such as:

- ESConv
- ExTES

For rapid iteration and debugging, the built-in `tasks` source was extremely useful because it let us prove the pipeline end-to-end with small runs.

## What We Actually Ran in This Environment

In the current environment, we completed a full reduced-scale proof run of the pipeline.

### Simulation Run

We generated simulation data under a fast configuration using:

- the built-in `tasks` seeds
- `Qwen/Qwen2.5-7B-Instruct` as the remote simulation policy/critic model through the Hugging Face router
- a small rollout configuration for speed

That produced:

- `artifacts/sim_data_fast/candidate_rewards.jsonl`
- `artifacts/sim_data_fast/trajectories.jsonl`

The simulation output contained 24 candidate-reward examples in the reduced test run.

### Reward Model Run

For the reward model, we trained on:

- `meta-llama/Llama-3.2-1B-Instruct`

This was used as a sequence-classification backbone with a newly initialized scalar head.

That produced:

- `artifacts/reward_model_fast/model.safetensors`
- `artifacts/reward_model_fast/config.json`
- `artifacts/reward_model_fast/reward_model_audit.jsonl`
- `artifacts/reward_model_fast/reward_model_audit_summary.json`
- associated tokenizer and metadata files

### GRPO Run

For the policy optimization step, we successfully ran GRPO with:

- base model: `meta-llama/Llama-3.2-1B-Instruct`
- learned reward model from stage 2
- LoRA-based parameter-efficient fine-tuning

The successful GRPO run produced:

- `adapter_config.json`
- `adapter_model.safetensors`
- checkpoint artifacts
- training state and metadata

This gave us a deployable adapter suitable for Hugging Face model hosting and Space-based inference.

## The Engineering Reality: What Broke and What We Fixed

A big part of building real training systems is dealing with the gap between the clean intended pipeline and the actual runtime environment.

This branch hit several practical issues, and fixing them was part of the work.

### 1. Remote model support mismatch

The Hugging Face router did not support some of the initial Qwen variants we first tried for simulation.

In practice:

- `Qwen/Qwen2.5-3B-Instruct` was not available through the enabled providers
- `Qwen/Qwen2.5-7B-Instruct` was available
- `meta-llama/Llama-3.1-8B-Instruct` was also available

This meant the simulation stage had to be adapted to use a provider-supported chat model.

### 2. Missing module entrypoints

Two training files were missing their executable module guards:

- `training/reward_model.py`
- `training/grpo_policy.py`

Without:

```python
if __name__ == "__main__":
    main()
```

the commands appeared to succeed while actually doing nothing.

That bug had to be fixed before the pipeline would run correctly in Colab.

### 3. `transformers` version incompatibility

The default environment drifted onto `transformers 5.0.0`, while the project code expected `4.x` behavior.

This caused:

- `TrainingArguments` API mismatches
- incompatible argument names

We stabilized the environment by moving back to a `4.57.1`-style stack.

### 4. Reward-model dataset collation issue

The reward-model trainer originally kept the raw `text` column during tokenization. That caused the collator to attempt to tensorize strings.

The fix was to remove the raw text column after tokenization during dataset mapping.

### 5. GRPO config mismatch

The installed `trl` version did not accept the same GRPO configuration fields the code originally used.

In particular, `max_prompt_length` had to be removed from the `GRPOConfig(...)` construction for compatibility in the current environment.

### 6. Gated base model deployment

The deployed Space initially failed because:

- `meta-llama/Llama-3.2-1B-Instruct` is gated
- the Space was trying to load it without authenticated access

The deployment fix required:

- accepting the Meta license on the owning Hugging Face account
- adding `HF_TOKEN` as a Space secret
- loading the base model with token-authenticated `from_pretrained(...)`

### 7. Adapter repo cleanup

When pushing the trained adapter to Hugging Face, the first push failed because unnecessary checkpoint and tokenizer files exceeded normal repo file-size limits.

The final clean deployment path was:

- create a fresh model repo clone
- copy only the minimal adapter files
- push only:
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `README.md`

## What We Deployed

We deployed the trained adapter to a Hugging Face model repo and connected it to a Hugging Face Space.

### Model Repo

The model repo contains the minimal GRPO adapter artifacts needed for inference:

- LoRA adapter weights
- adapter config
- model README

### Space

The Space is a simple Gradio chat interface that:

1. loads the base model
2. attaches the trained adapter
3. applies a system prompt tuned for calm, supportive interaction
4. returns generated support responses to user input

The Space is designed as a direct product-facing demo of the branch’s output.

## Why This Matters

The strongest idea in this branch is not that we made a nicer chatbot.

It is that we built a trainable long-horizon emotional-support environment where future outcomes matter.

That is a more serious target than short-horizon “empathetic response quality.”

This matters because many real human-facing AI tasks are long-horizon by nature:

- emotional support
- coaching
- education
- adherence
- conflict resolution
- behavior change

In all of them, the best current-turn answer is not always the answer that leads to the best future state.

If we keep optimizing only for local response quality, we will keep producing systems that look good in demos and fail over time.

This branch is our attempt to train for the thing that actually matters:

- better future trajectories
- better continuity
- better timing
- better trust preservation
- better recovery from mistakes

## Current State of the Project

As of this branch and current environment, we have:

- a long-horizon OpenEnv-compatible distress-support environment
- hidden-state emotional dynamics
- multiple task arcs with delayed consequences
- simulation-based candidate-reward generation
- a trained reward model
- a GRPO-trained adapter
- a Hugging Face model repo for the adapter
- a Hugging Face Space deployment path for interactive inference

This is already more than a benchmark or a static prompt demo.

It is a full environment-to-training-to-deployment pipeline.

## Where We Can Take It Next

The current branch is a working proof and product direction, not the endpoint.

The obvious next steps are:

- scale simulation runs beyond the small debug configuration
- train on larger seed sets from ESConv and ExTES
- improve reward-model quality with more data and stronger audits
- tune formatting reward so structured reasoning is actually reinforced
- compare pre-training and post-training transcripts on the hardest arcs
- add evaluation dashboards and before/after qualitative demos
- improve the deployed Space UX and safety messaging

The long-term objective is straightforward:

build a distress-support assistant that is not just superficially empathetic, but strategically supportive over time.

## Final Takeaway

The main lesson from this branch is simple:

distress support is not a one-turn generation problem.

It is a long-horizon control problem under hidden human state.

That means the model must learn more than style. It must learn:

- pacing
- trust management
- disclosure timing
- continuity
- future outcome awareness

This project is our attempt to operationalize that idea in a trainable stack.

Not just a chatbot.
Not just a benchmark.
Not just a demo.

A long-horizon distress AI coach trained to care about what happens next.
