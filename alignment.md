# Competition Alignment

## Verdict

With the project in hand, you can be competitive, but not in its current framing.

The competition is optimizing for:

- `40%` environment innovation
- `30%` storytelling and presentation
- `20%` proof of improvement
- `10%` reward and training pipeline

The repo is currently strongest on environment design and reward logic, moderate on clean engineering, and weakest on competition-facing story and training evidence. If it is submitted as an emotional-support benchmark inspired by RLFF-ESC, it is unlikely to win. If it is submitted as a long-horizon personalized interaction training environment with hidden mental state, delayed reward, future-oriented simulation, and actual post-training gains, it has a real chance.

## Best Angle

The strongest theme fit is:

- `Theme #2: Long-Horizon Planning & Instruction Following`
- secondarily `Theme #3.2: Personalized Tasks`

The strongest narrative is:

- LLMs are bad at sustained emotional support because they optimize local empathy, not long-term user outcomes.
- This environment forces the model to infer hidden state, recover from early mistakes, track rapport over many turns, and decide when to shift from exploration to action.
- The training system uses future-oriented simulation and reward modeling to improve long-horizon conversational behavior.

That framing is much stronger than positioning this as a benchmark with baselines.

## Competition Mapping

What already helps:

- Clear problem with real importance.
- Hidden state plus delayed consequences.
- Non-trivial reward design.
- OpenEnv server/client separation.
- Natural Hugging Face Space deployment story.

What currently hurts:

- The README still reads like a benchmark/environment writeup, not a competition-winning training story.
- The handcrafted tasks are too small and benchmark-like to carry the training claim alone.
- There is still no visible end-to-end evidence of a trained model improving.
- The project currently has too many parallel stories:
  - deterministic benchmark
  - skill router
  - paper-faithful RLFF-ESC training scaffold
  - emotional-support environment

Judges need one story.

## What Will Actually Win

If the goal is to maximize the chance of winning, focus on six deliverables:

1. One sharp claim

   "We built an OpenEnv training environment that improves long-horizon emotional-support behavior under partial observability using future-oriented rewards."

2. Ablation-ready training evidence

   Show:

   - base model vs SFT vs reward-model plus GRPO
   - average reward
   - success rate
   - reveal rate
   - unsafe-response rate
   - average turns to successful resolution

3. One strong demo

   Same scenario, same opening:

   - before training: generic empathy loop, misses reveal, bad timing
   - after training: steady rapport, timely exploration, correct safety escalation, closes well

4. A clean reward story

   Judges do not need every paper detail. They need:

   - immediate quality
   - future trajectory quality
   - anti-gaming penalties
   - why the reward matches real success

5. A training notebook

   Colab-runnable, low-friction, producing at least:

   - reward curve
   - eval table
   - one saved checkpoint
   - one transcript comparison

6. A compact presentation layer

   README plus 90-second demo plus 3 plots. That is where `30%` of the score lives.

## What To Deprioritize

These are not where this project will win:

- perfect benchmark completeness
- too many task variants
- too much emphasis on deterministic baselines
- overexplaining RLFF-ESC fidelity
- broad multi-theme positioning

Pick one story and drive it hard.

## Current Readiness

Against the competition checklist:

- `OpenEnv latest release`: likely fine
- `training script (TRL/Unsloth)`: present
- `Colab runnable`: close, but not yet packaged as a clear notebook path
- `actual training evidence`: missing
- `loss/reward curves`: missing
- `HF Space`: environment side mostly there
- `README with results and demo links`: not competition-ready yet

The biggest gap is no longer environment design. It is proof.

## Judge Positioning

Do not position the project as:

- "We created a benchmark for emotional support."

Position it as:

- "We created a training environment for long-horizon personalized support, where success depends on inferring hidden emotional state, pacing disclosure, and optimizing future user outcomes rather than locally pleasant responses."

That is more aligned with the competition brief.

## Priority Order

To maximize winning odds:

1. Run one real training cycle and collect curves.
2. Build one strong evaluation suite with before/after comparisons.
3. Rewrite the README around the competition criteria, not around the benchmark.
4. Keep only the benchmark pieces that support the training story.
5. Publish one HF Space demo and one short video.

## Recommendation

The winning version of this project is not the most faithful RLFF-ESC reproduction.

It is:

- a clear long-horizon personalized-task environment
- with future-oriented rewards
- with actual training evidence
- explained so a non-specialist judge can understand it in under 3 minutes
