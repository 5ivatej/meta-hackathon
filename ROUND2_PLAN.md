# Round 2 Execution Plan

This plan assumes you keep the current environment, skip tool calling, and extend it into a longer-horizon therapist-style environment for Round 2 instead of rebuilding from scratch.

## 1. Recommended Direction

### Chosen Theme

**Primary target: Theme 3.2 Personalized Tasks**  
**Extended target: Theme 2 + Theme 3.2 hybrid**

### Submission Pitch

Build a trainable OpenEnv environment for **online therapist-style conversations** where the agent must:

- infer hidden emotional state
- build trust over time
- decide when to explore vs. validate vs. suggest a next step
- escalate to real-world safety support when needed

Recommended wording for the submission:

- `online therapist-style assistant`
- `therapy-support agent`
- `therapy conversation simulator`

Avoid claiming the system is a licensed therapist.

### Deliberate Scope Limits

This plan does **not** require adding:

- real external tool use

This plan **does** assume you want to add:

- longer-term memory across sessions
- long-horizon task structure
- beyond-context persistence
- a more realistic personal therapeutic workflow

Those are the changes that make this a stronger Theme 2 submission.

### Why this is the right move

- It fits the existing repo much better than multi-agent or long-horizon.
- The environment is already partially observable and reward-shaped.
- It avoids diluting the submission with half-built tool-use features.
- It builds directly on the hidden-state conversation model you already have.
- It gives you a credible path to a Theme 2 + Theme 3.2 hybrid.

## 2. Goal State

By submission time, the repo should clearly show:

1. a strong Round 2 problem statement
2. a valid OpenEnv environment hosted on HF Spaces
3. a minimal TRL or Unsloth training script/notebook
4. real training evidence with plots
5. a README that tells a clean before/after story
6. a short blog/video/slides link
7. multi-session memory or persistent state that makes the benchmark genuinely longer horizon

## 3. Workstreams

Design reference:

- `ROUND2_ENV_DESIGN.md` should be treated as the concrete implementation blueprint for the long-horizon extension.

## A. Reframe the Submission

Objective:
Make the project read like a Round 2 **Theme 2 + Personalized Tasks** submission, not a Round 1 benchmark repo.

Files to update:

- `README.md`
- `openenv.yaml`
- optionally `SUBMISSION_NEXT_STEPS.md`

Changes:

1. Rewrite the top of `README.md` around the Round 2 problem statement.
2. Add a section called `Why This Matters for Online Therapist Agents`.
3. Explain the hidden state clearly: `trust`, `distress`, `openness`, `revealed`, `stage`.
4. Add a section called `Persistent Memory and Long-Horizon Design`.
5. Add a section called `What the Agent Learns`.
6. Add a section called `Before vs After Training`.
7. Add links to the HF Space, Colab notebook, and blog/video.

Definition of done:

- A judge can understand the problem, environment, and result in under 5 minutes.

## B. Tighten OpenEnv Compliance

Objective:
Reduce risk around the framework checklist.

Files to inspect/update:

- `src/env.py`
- `server/app.py`
- `pyproject.toml`
- `openenv.yaml`

Tasks:

1. Confirm the installed `openenv-core` version and whether the env should subclass an official OpenEnv base class.
2. If required by current OpenEnv docs, refactor `ESCEnv` to inherit from the official class.
3. Keep `reset`, `step`, and `state` unchanged from the caller perspective.
4. Verify the manifest fields in `openenv.yaml` are aligned with the latest expected schema.

Definition of done:

- You can confidently say the repo uses OpenEnv correctly and is not just OpenEnv-shaped.

## C. Add a Minimal Real Training Pipeline

Objective:
Satisfy the most important missing requirement.

New files to add:

- `train_trl.py` or `train_grpo.py`
- `notebooks/round2_training_colab.ipynb` or a `.md`/Colab link if you prefer remote
- `scripts/plot_training.py` or equivalent plot generation utility

Recommended scope:

- Keep it minimal.
- Use HF TRL unless Unsloth gives you faster execution for the same story.
- Train a small model or a lightweight policy adapter against the environment.
- Do not block on tool calling.
- Make sure the training setup can handle the longer-horizon memory version once you add it.

What the training script should do:

1. connect to the environment, not a static dataset
2. run episodes against the benchmark
3. collect rewards and losses
4. save checkpoints or adapters
5. write metrics to a file you can plot later

Definition of done:

- You can run one command or one notebook and produce real training logs.

## D. Extend the Environment to Long-Horizon Therapy

Objective:
Add the capabilities you described without introducing tool use.

Likely design changes:

1. Turn one episode into a multi-session arc instead of a 10-14 turn single session.
2. Persist seeker state across sessions.
3. Add memory summaries or compact state that survives context-window boundaries.
4. Make early-session mistakes affect later-session trust, disclosure, and progress.
5. Delay some rewards so the agent is rewarded for durable progress, not just immediate tone.

Likely files to change:

- `src/env.py`
- `src/tasks.py`
- `src/seeker.py`
- `src/grader.py`
- `src/models.py`

Design source:

- see `ROUND2_ENV_DESIGN.md`

Definition of done:

- The benchmark is credibly long-horizon even without tool calling.

## E. Generate Training Evidence

Objective:
Show actual improvement, not just evaluation.

Artifacts to produce:

- `results/training_metrics.json`
- `results/reward_curve.png`
- `results/loss_curve.png`
- `results/before_after.md`

Minimum acceptable evidence:

1. untrained baseline score
2. trained model score
3. reward curve over training
4. one short qualitative transcript comparison

Best simple story:

- before training: agent stays generic and fails completion on medium/hard tasks
- after training: agent reaches stage transitions more reliably, carries useful memory across sessions, and uses safety escalation correctly on the hard task

Definition of done:

- The README contains at least one plot and one before/after example.

## F. Improve Storytelling

Objective:
Convert technical work into something judges remember.

Assets to create:

- short Hugging Face blog post or README-style article
- or a `<2 min` YouTube demo
- optional 3-5 slide deck

Suggested narrative:

1. LLMs often sound empathetic without behaving like a good online therapist.
2. Real therapist-style support is long-horizon: trust, disclosure, and progress unfold across sessions.
3. This environment makes therapist-style support partially observable, memory-dependent, and safety-sensitive.
4. The reward teaches timing, pacing, disclosure handling, memory use, and escalation, not just nice wording.
5. Training improves completion, cross-session consistency, and safety-aware behavior.
6. The system intentionally avoids tool calling so the benchmark stays focused on conversational therapeutic reasoning.

Definition of done:

- All external materials are linked from `README.md`.

## G. Host and Demo

Objective:
Make the project runnable for judges.

Tasks:

1. Deploy the environment to HF Spaces.
2. Verify the Space exposes the expected OpenEnv interface.
3. Keep the browser UI if it helps the demo, but make the API story primary.
4. Put the live URL at the top of the README.

Definition of done:

- A judge can click one link and run or inspect the environment immediately.

## 4. Recommended File Plan

### Must Add

- `ROUND2_PLAN.md`
- training script: `train_trl.py` or similar
- training notebook or Colab link
- training plots in `results/`
- submission narrative artifact link in `README.md`

### Must Edit

- `README.md`
- `openenv.yaml`
- possibly `src/env.py` if official OpenEnv inheritance is needed

### Nice to Add

- `results/before_after.md`
- one screenshot from the Space UI
- one ablation note comparing plain vs skill-routed policy

## 5. Priority Order

Do this in order:

1. Lock the Round 2 framing around Theme 2 + Personalized Tasks.
2. Finalize the long-horizon extension design in `ROUND2_ENV_DESIGN.md`.
3. Verify OpenEnv compliance against the latest release.
4. Implement the long-horizon environment changes.
5. Add the minimal training script.
6. Run one real training job and save metrics.
7. Generate plots and before/after comparisons.
8. Rewrite the README around the new story.
9. Publish the Space and external explainer.

## 6. Practical Timeline

### Phase 1: Today

Deliverables:

- final theme selection
- long-horizon design decision
- environment blueprint in `ROUND2_ENV_DESIGN.md`
- Round 2 README outline
- confirmed environment framing
- confirmed OpenEnv compliance gaps

### Phase 2: Next Build Session

Deliverables:

- long-horizon env changes
- minimal training script
- one successful local training run
- raw metrics saved to `results/`

### Phase 3: Packaging Session

Deliverables:

- plots
- updated README
- HF Space deployment
- blog/video/slides links

## 7. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Training is too slow or unstable | High | Use a smaller model, fewer tasks, and one clear training story rather than chasing scale. |
| OpenEnv compliance turns out incomplete | High | Fix framework usage before polishing presentation. |
| No visible reward improvement | High | Run a simpler baseline and compare against trained behavior on medium/hard tasks only. |
| Story feels too similar to Round 1 | Medium | Reframe around therapist-style capability learning, cross-session memory, and safety-sensitive planning. |
| Therapist framing sounds overclaimed or unsafe | Medium | Use therapist-style / therapy-support wording and emphasize escalation to real-world help. |
| The task still feels too short-horizon after changes | Medium | Make memory persistence visible in both env design and evaluation outputs. |
| Judges want tool use | Medium | Be explicit that this submission targets long-horizon personalized conversation, not professional tool workflows. |

## 8. What Not To Do

Avoid these traps:

1. Do not re-pitch this as multi-agent unless you actually redesign the environment.
2. Do not imply the environment has real tool use when it does not.
3. Do not claim persistent memory until it is actually implemented.
4. Do not spend most of your time polishing the UI before training evidence exists.
5. Do not submit with only benchmark numbers and no actual training proof.
6. Do not overcomplicate the first training pipeline; minimal and real beats ambitious and broken.

## 9. Submission Checklist

- Theme clearly stated as Personalized Tasks
- Theme 2 long-horizon angle clearly stated if implemented
- Niche clearly stated as online therapist-style support
- No-tool-calling scope clearly stated
- Problem statement visible near the top of the README
- HF Space URL linked
- OpenEnv manifest valid
- Training script included
- Colab or notebook included
- Reward/loss plots committed
- Before/after behavior comparison included
- Blog/video/slides linked from README
- Results easy to scan in under 5 minutes

## 10. Immediate Next Step

The next concrete engineering task should be:

**Design and implement the long-horizon memory extension first, then add the minimal training pipeline.**

That is the highest-leverage change if your real goal is Theme 2. The current repo already has enough hidden-state logic to evolve into a longer-horizon therapist environment, but the long-horizon behavior needs to exist before the training evidence will tell the right story.
