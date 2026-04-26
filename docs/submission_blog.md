# Training a Therapy Assistant to Optimize the Next Few Sessions, Not Just the Next Reply

Most LLM assistants are rewarded for sounding good right now.

That is not enough for emotionally sensitive support.

In real support conversations, the best response is often not the one that looks smartest or most helpful in a single turn. It is the one that makes the next part of the conversation safer, more honest, and more productive. If the assistant gives advice too early, the user may shut down. If it misses a fragile moment, trust can collapse. If it forgets what happened last session, the whole interaction starts to feel fake.

That is the problem we chose to work on for the OpenEnv Hackathon.

We built an OpenEnv training environment for a long-horizon therapy-style assistant: a system that must infer hidden emotional state, build trust over time, handle delayed consequences, and learn to optimize future user outcomes rather than locally polished replies.

This project is strongest in two competition themes:

- Theme #2: Long-Horizon Planning and Instruction Following
- Theme #3.2: Personalized Tasks

## The Core Idea

The central claim of this project is simple:

> a support assistant should be trained on the future trajectory of the conversation, not only on one-turn response quality.

That changes the task in an important way.

Instead of asking, "Was this reply empathetic?", we ask:

- Did this reply increase trust?
- Did it reduce distress?
- Did it help the user reveal the real issue?
- Did it preserve continuity across sessions?
- Did it move the conversation toward a safe and useful next step?

Those questions are inherently long-horizon. They require state, memory, and delayed reward. That is exactly why OpenEnv is the right framework for the problem.

## Why This Problem Matters

LLMs are already capable of producing supportive-sounding language. What they are still weak at is pacing.

They often:

- rush into advice before trust is earned
- respond to the surface issue while missing the real issue underneath
- fail to recover after a weak early turn
- lose continuity across sessions
- treat safety as a single phrase instead of an ongoing responsibility

Emotional support is therefore a strong benchmark for agentic intelligence under partial observability. The assistant does not get direct access to the user’s internal state. It has to infer that state from behavior, adapt over time, and choose responses that improve the future, not just the present moment.

## What We Built

We built a trainable OpenEnv environment for multi-session emotional support conversations.

At the interaction level, the agent only sees the user’s messages plus a compact task context. But internally, the environment tracks hidden variables that determine whether the conversation is actually improving.

These include:

- trust
- distress
- openness
- reveal progress
- conversation stage
- continuity across sessions
- working goals and unfinished threads
- safety-sensitive progress
- budget and time constraints for controlling the rollout

This makes the environment partially observable by design. The model cannot directly read "trust = 0.62" or "the user is ready to disclose now." It has to infer those facts from the dialogue and respond accordingly.

That is important, because real support conversations work exactly this way.

## The Environment Design

The environment is not a static prompt set. It is a stateful world with transitions, reward, and termination logic.

Each episode is a therapy-style arc with multiple sessions. The assistant must navigate the conversation through stages such as opening, exploring, reflecting, planning, and closing. Progress is not guaranteed. A poorly timed response can slow progress, trigger resistance, or damage trust.

We currently include three built-in archetypes:

- `work_stress_venting`: the user first presents stress, but the deeper issue is burnout and fear about what comes next
- `guarded_relationship`: the user is emotionally guarded and only reveals the real issue once enough trust has been built
- `crisis_fragile_trust`: the user is overwhelmed, trust is extremely fragile, and safety handling becomes essential

These tasks were chosen to demonstrate three distinct failure modes:

- the assistant must avoid premature fixing
- the assistant must earn disclosure rather than force it
- the assistant must remain calm and supportive under risk-sensitive conditions

Each task also has hard success gates. Finishing a conversation is not enough. The model must reach the right stage, maintain enough trust, lower distress enough, surface the real issue when required, and include safety reference when the scenario demands it.

That makes success much harder to game.

## Why Hidden State Matters

A good support system cannot be evaluated purely from surface text.

Two replies may both sound caring, but one might actually increase the chance of future disclosure while the other quietly pushes the user away. Our environment captures that distinction through hidden state transitions.

For example:

- empathy can increase openness
- validation can reduce distress
- advice at the wrong stage can hurt trust
- continuity can improve later-session performance
- repetitive or generic responses can create drift

This is one of the most important properties of the project. We are not training the model to imitate a tone. We are training it to act on a latent emotional process that unfolds over time.

## Our Reward Philosophy

The reward design is the heart of the project.

We do not only reward the assistant for producing a nice-looking answer. We reward it for making the future conversation better.

The environment-side reward combines four ideas:

- immediate conversational quality
- future-oriented trajectory value
- anti-gaming penalties
- long-horizon continuity terms

Immediate quality captures whether the response fits the current stage well, improves trust, lowers distress, and helps the conversation progress naturally.

Future-oriented reward captures something more important: whether the new state creates a better next few turns than the old state would have. In other words, does this response improve the conversation’s future ceiling?

We also penalize behaviors that look superficially active but are strategically poor, such as:

- dismissiveness
- premature advice
- bare or low-effort replies
- interrogative overload
- repetition

Finally, we add long-horizon terms for continuity, goal-following, session transitions, and controlled use of the conversation budget. These keep the training signal aligned with a multi-session assistant rather than a one-turn chatbot.

## From Environment to Training Pipeline

We wanted the project to be more than a simulator. It needed a credible path from environment design to post-training improvement.

So we built a three-stage training pipeline.

### Stage 1: Environment-Backed Simulation

We begin with seed dialogue prefixes from sources such as ESConv and ExTES, plus built-in task arcs for ablations and demos.

For each seed:

- a policy model proposes candidate therapist responses
- the environment rolls the dialogue forward under its hidden state dynamics
- a separate critic evaluates the future trajectory quality
- the system saves training records of context, response, and future-oriented reward

This step matters because it transforms emotional support from a static supervised task into a trajectory-based learning problem.

### Stage 2: Learned Future-Oriented Reward Model

Next, we distill those rollout judgments into a learned scalar reward model.

This makes reinforcement learning practical. Instead of asking a large critic to judge every policy update from scratch, we train a smaller model to approximate future-oriented reward from the simulated data.

We also keep audit outputs so we can inspect where the reward model overestimates or underestimates candidate responses.

That is important for a competition setting because it makes the reward story more transparent and easier to defend.

### Stage 3: GRPO Policy Optimization

Finally, we fine-tune the policy with GRPO using the learned reward model.

This closes the loop:

- the environment defines the world and its dynamics
- simulation generates trajectory-level supervision
- the reward model distills that supervision
- GRPO optimizes the assistant policy against it

That is the main research contribution of the project. We are not only building a benchmark. We are building a trainable world for long-horizon emotional support.

## What Makes This a Strong OpenEnv Submission

This project aligns well with what the competition is actually asking for.

First, it is an environment, not just a dataset wrapper. The assistant interacts with a stateful system that has hidden variables, transitions, rewards, multi-session dynamics, and hard completion conditions.

Second, it is ambitious but legible. Judges do not need a niche domain background to understand why emotional support is difficult. But beneath that familiar surface, the task is technically rich: partial observability, delayed reward, safety sensitivity, recovery from mistakes, and continuity over time.

Third, the reward is meaningful. It is grounded in the future trajectory of the conversation rather than pure style scoring.

Fourth, the training pipeline is real. There is a clear path from environment interaction to reward learning to policy improvement.

That combination is exactly what makes an OpenEnv project compelling.

## What We Expect the Agent to Learn

If training works as intended, the model should improve in ways that matter behaviorally, not just cosmetically.

We expect improvement on:

- knowing when to listen instead of solving
- building enough trust for the real issue to surface
- choosing when to explore, reflect, plan, or escalate for safety
- recovering from an imperfect earlier turn instead of compounding it
- carrying memory and rapport across sessions
- producing replies that improve the user’s future trajectory

The strongest evidence will not be a single impressive reply. It will be a pattern:

- higher rewards
- better task completion rates
- stronger continuity
- safer handling in fragile cases
- qualitatively better multi-turn transcripts before vs after training

## Why This Matters Beyond the Competition

We think this project points to a broader lesson for LLM training.

Many important real-world tasks are not one-turn tasks. They involve hidden state, delayed consequences, and the need to shape future behavior rather than optimize present style. Emotional support makes those challenges visible very quickly, but the same structure appears in tutoring, coaching, negotiation, healthcare triage, and personal assistant workflows.

So even though this project is framed as a therapy-style assistant environment, the deeper contribution is a reusable pattern:

> build environments where the right action is the one that improves the future state of the interaction.

That is a much more realistic target for agent training.

## Closing

This project is our attempt to move emotional support training from "sound supportive now" to "make the next few sessions go better."

By combining hidden-state conversation dynamics, future-oriented reward, multi-session trajectories, and an explicit post-training loop, we built an OpenEnv environment where long-horizon personalized support becomes trainable.

That is the story of this submission:

we are not training a model to write nicer replies.

We are training it to create better outcomes over time.
