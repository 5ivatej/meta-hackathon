# Round 2 Execution Plan

This plan assumes you keep the current environment, keep the scope narrow, and reposition it for Round 2 instead of rebuilding from scratch.

## 1. Recommended Direction

### Chosen Theme

**Theme 3.2: World Modeling -> Personalized Tasks**

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
- longer-term memory across sessions
- long-horizon task structure
- a richer open-ended personal world simulator

Those are valid future extensions, but they are not required for a credible Theme 3.2 submission.

### Why this is the right move

- It fits the existing repo much better than multi-agent or long-horizon.
- The environment is already partially observable and reward-shaped.
- You can spend effort on training proof and presentation instead of rewriting the whole benchmark.
- It avoids diluting the submission with half-built tool-use or memory features.

## 2. Goal State

By submission time, the repo should clearly show:

1. a strong Round 2 problem statement
2. a valid OpenEnv environment hosted on HF Spaces
3. a minimal TRL or Unsloth training script/notebook
4. real training evidence with plots
5. a README that tells a clean before/after story
6. a short blog/video/slides link

## 3. Workstreams

## A. Reframe the Submission

Objective:
Make the project read like a Round 2 Personalized Tasks submission, not a Round 1 benchmark repo.

Files to update:

- `README.md`
- `openenv.yaml`
- optionally `SUBMISSION_NEXT_STEPS.md`

Changes:

1. Rewrite the top of `README.md` around the Round 2 problem statement.
2. Add a section called `Why This Matters for Online Therapist Agents`.
3. Explain the hidden state clearly: `trust`, `distress`, `openness`, `revealed`, `stage`.
4. Add a section called `What the Agent Learns`.
5. Add a section called `Before vs After Training`.
6. Add links to the HF Space, Colab notebook, and blog/video.

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
- Do not block on adding tools, memory, or a richer simulator first.

What the training script should do:

1. connect to the environment, not a static dataset
2. run episodes against the benchmark
3. collect rewards and losses
4. save checkpoints or adapters
5. write metrics to a file you can plot later

Definition of done:

- You can run one command or one notebook and produce real training logs.

## D. Generate Training Evidence

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
- after training: agent reaches stage transitions more reliably and uses safety escalation correctly on the hard task

Definition of done:

- The README contains at least one plot and one before/after example.

## E. Improve Storytelling

Objective:
Convert technical work into something judges remember.

Assets to create:

- short Hugging Face blog post or README-style article
- or a `<2 min` YouTube demo
- optional 3-5 slide deck

Suggested narrative:

1. LLMs often sound empathetic without behaving like a good online therapist.
2. This environment makes therapist-style support sequential, partially observable, and safety-sensitive.
3. The reward teaches timing, pacing, disclosure handling, and escalation, not just nice wording.
4. Training improves completion and safety-aware behavior.
5. The current benchmark is intentionally narrow: no tools, no persistent memory, and a deterministic simulator for reproducibility.

Definition of done:

- All external materials are linked from `README.md`.

## F. Host and Demo

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

1. Lock the Round 2 framing around Personalized Tasks.
2. Add the scope boundaries explicitly so you do not overclaim.
3. Verify OpenEnv compliance against the latest release.
4. Add the minimal training script.
5. Run one real training job and save metrics.
6. Generate plots and before/after comparisons.
7. Rewrite the README around the new story.
8. Publish the Space and external explainer.

## 6. Practical Timeline

### Phase 1: Today

Deliverables:

- final theme selection
- explicit scope limits
- Round 2 README outline
- confirmed environment framing
- confirmed OpenEnv compliance gaps

### Phase 2: Next Build Session

Deliverables:

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
| Story feels too similar to Round 1 | Medium | Reframe around therapist-style capability learning, hidden state, and safety-sensitive planning. |
| Therapist framing sounds overclaimed or unsafe | Medium | Use therapist-style / therapy-support wording and emphasize escalation to real-world help. |
| The task feels too short-horizon | Medium | Emphasize partial observability, delayed effects, and safety timing rather than claiming long-horizon planning. |
| Judges want a richer world than you built | Medium | Be explicit that this is a narrow but trainable personalized conversation environment, then show strong training evidence. |

## 8. What Not To Do

Avoid these traps:

1. Do not re-pitch this as multi-agent unless you actually redesign the environment.
2. Do not claim long-horizon planning as the primary theme.
3. Do not imply the environment has real tool use or persistent memory when it does not.
4. Do not spend most of your time polishing the UI before training evidence exists.
5. Do not submit with only benchmark numbers and no actual training proof.
6. Do not overcomplicate the first training pipeline; minimal and real beats ambitious and broken.

## 9. Submission Checklist

- Theme clearly stated as Personalized Tasks
- Niche clearly stated as online therapist-style support
- Scope limits clearly stated
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

**Add the minimal training pipeline and metrics output first.**

That is the highest-leverage gap. The current repo already has enough environment logic to tell a credible story, but it does not yet have the training evidence judges explicitly require.
