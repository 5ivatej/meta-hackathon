# Technical Detail

This document explains the project from scratch to finish: what problem it targets, how the environment works, how the two-model training pipeline works, and how to present it clearly.

## 1. Problem Statement

This project builds an OpenEnv-compatible environment for training a therapist-style conversational agent.

The target capability is not generic chat. It is:

- multi-session emotional support
- partial observability
- delayed consequences
- memory continuity across sessions
- safe progression from empathy to reflection to planning

The key Round 2 framing is:

- **Theme 2**: long-horizon planning and instruction following
- **Theme 3.2**: personalized task handling

The system does **not** use external tools. Instead, it focuses on long-horizon therapeutic continuity:

- remembering prior disclosures
- carrying unresolved threads across sessions
- tracking whether the agent follows through
- rewarding progress that only pays off later

## 2. High-Level Architecture

There are four main layers.

1. **Environment**
   - Simulates the seeker, task, session structure, and reward.
   - Main files: [src/env.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/env.py), [src/tasks.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/tasks.py), [src/grader.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/grader.py), [src/seeker.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/seeker.py)

2. **Agentic controller + memory**
   - Maintains durable memory and chooses a conversational skill each turn.
   - Main files: [src/agentic.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/agentic.py), [src/runner.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/runner.py)

3. **Training pipeline**
   - Trains a separate reward model and a separate policy model.
   - Main files: [train_reward_model.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/train_reward_model.py), [train_trl.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/train_trl.py), [src/training_utils.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/training_utils.py)

4. **Serving / deployment**
   - Exposes the environment over HTTP and packages everything in Docker for local or Colab use.
   - Main files: [server/app.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/server/app.py), [docker-compose.yml](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/docker-compose.yml), [Dockerfile](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/Dockerfile)

## 3. Environment Design

### 3.1 What the agent sees

Each step returns an `Observation` object defined in [src/models.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/models.py).

The observation includes:

- the seeker’s latest utterance
- the public stage hint
- task id and scenario brief
- session index and total sessions
- remaining turns in the episode and in the current session
- rolling memory summary
- previous session outcome
- current goal hint
- cost budget usage
- time budget usage

This is important because the environment is no longer a short single-session chat. The agent must act with continuity.

### 3.2 What the hidden world contains

The hidden world is stored in `ESCEnv` in [src/env.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/env.py).

It tracks:

- seeker latent state
- trust
- distress
- openness / reveal progress
- current stage
- alliance and continuity statistics
- session progression
- unfinished threads
- memory summary
- cost and time budgets

This hidden state is what makes the environment partially observable.

### 3.3 Tasks

The benchmark currently contains three therapy-style arcs in [src/tasks.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/tasks.py):

- `work_stress_venting`
- `guarded_relationship`
- `crisis_fragile_trust`

Each task defines:

- the persona
- surface concern
- hidden true issue
- disclosure threshold
- max turns
- sessions total
- session turn limit
- cost and time budgets
- working goals
- follow-up openers for later sessions
- success thresholds

That gives the environment both short-term and long-term structure.

## 4. Seeker Dynamics

The seeker is not an LLM. It is a deterministic simulator defined by structured persona parameters and transition logic in [src/seeker.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/seeker.py).

At each turn:

1. The agent sends a message.
2. The environment extracts conversational features from that message.
3. The seeker state transitions based on those features.
4. The seeker produces the next utterance.

The extracted features include signals such as:

- empathy
- validation
- advice
- open question
- safety reference
- dismissiveness
- interrogation

This design makes the environment stable and trainable. It avoids needing a second LLM inside the environment loop.

## 5. Long-Horizon Extensions

This Round 2 version extends the original shorter environment in four major ways.

### 5.1 Multi-session episodes

Episodes span multiple therapy sessions instead of a single short interaction.

`ESCEnv.step()` advances session state when the per-session turn limit is reached. Later sessions can reopen old issues using `session_openers`.

### 5.2 Persistent memory and rolling summaries

The old pattern of prompting from only the last few turns has been replaced by durable memory plus local recent exchange.

`AgentMemory` in [src/agentic.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/agentic.py) stores:

- rolling summary
- last session outcome
- current goal hint
- unresolved threads
- risk markers
- recent turns
- budget usage

The prompt is built from:

- long-term memory summary
- current goals
- unresolved therapeutic threads
- recent local context

### 5.3 Resume / restart durability

The system supports pause and resume via [src/runner.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/runner.py).

`RunnerCheckpoint` stores:

- environment session token
- serialized agent memory
- last observation
- short history tail
- cumulative reward
- step count

This allows long-running episodes to survive worker restarts.

### 5.4 Budget-aware long-horizon control

The environment tracks:

- episode budget spent
- episode budget limit
- episode time spent
- episode time limit

This matters because long-horizon agents often fail through drift, repetition, or runaway verbosity rather than a single catastrophic step.

## 6. Agentic Controller

The repo includes a deterministic skill-routed controller in [src/agentic.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/agentic.py).

It has two roles.

### 6.1 Skill routing

`SkillRouter` chooses the conversational move for the current context, such as:

- empathize
- validate
- reflect
- plan
- safety escalate

This controller acts as a structured teacher and keeps the training data coherent.

### 6.2 Skill realization

Each skill has:

- a deterministic `render(...)` method
- an `llm_instruction(...)` method

The deterministic version is used to generate strong baseline behavior and training targets. The LLM instruction version is used when prompting the learned policy model.

## 7. Reward Logic

The reward system lives in [src/grader.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/grader.py).

It combines three kinds of signal.

### 7.1 Immediate reward

Immediate reward measures whether the current reply is locally appropriate:

- empathy / validation fit
- trust increase
- distress reduction
- stage progress
- reveal bonus

### 7.2 Future-oriented reward

This is the paper-inspired part.

Instead of only rewarding the present turn, the grader estimates whether the reply improves the **future ceiling** of the conversation. It does this by comparing short oracle lookahead from the pre-step and post-step seeker states.

So the reward asks:

> Did this reply make better future trajectories possible?

That is why the reward is closer to therapeutic trajectory quality than to simple lexical matching.

### 7.3 Long-horizon reward and penalties

Round 2 adds a long-horizon context term.

It rewards:

- continuity hits
- goal follow-through
- good session transitions
- resume continuity
- budget efficiency

It penalizes:

- drift
- repetition
- runaway budget or time usage
- dismissive or premature responses

The final step reward is a weighted combination of immediate reward, future-oriented reward, and long-horizon reward minus penalties.

## 8. Two-Model Training Setup

This project now uses two separate learned models.

### 8.1 Reward model

The reward model is trained by [train_reward_model.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/train_reward_model.py).

Its job is:

- input: context + candidate therapist response
- output: scalar future-oriented quality score

Recommended practical model:

- `distilroberta-base`

How reward-model data is created:

1. Roll out the environment with the deterministic teacher.
2. At each state, generate multiple candidate responses from:
   - deterministic skills
   - simple baselines
3. Clone the environment state.
4. Step each candidate into the cloned environment.
5. Record the resulting `future_oriented` score and step reward.

That logic lives in [src/training_utils.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/training_utils.py), especially `collect_reward_dataset(...)`.

So the learned reward model is trained on labels produced by the environment’s built-in future-oriented grader.

### 8.2 Policy model

The policy model is trained by [train_trl.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/train_trl.py).

Its job is:

- input: memory-aware prompt for the current therapy state
- output: the next therapist response

Recommended practical models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

The script supports two dataset modes.

1. **Teacher dataset**
   - Uses the deterministic controller’s chosen response directly.

2. **Reward-guided dataset**
   - Uses the learned reward model to rerank multiple candidate responses.
   - The best-scoring candidate becomes the target response for policy training.

This second mode is what makes the pipeline closer to the original paper motivation.

## 9. Full Training Flow

This is the end-to-end story.

### Stage A: environment exists

The environment provides:

- task
- latent seeker state
- multi-session progression
- memory summary
- reward decomposition

### Stage B: generate reward-model dataset

`collect_reward_dataset(...)` does:

1. Reset a task in the env.
2. Observe the current therapy state.
3. Generate multiple candidate therapist replies.
4. Simulate each candidate in a cloned env.
5. Record the future-oriented score for each candidate.

Output:

- supervised regression dataset for reward-model training

### Stage C: train reward model

`train_reward_model.py`:

1. loads `REWARD_MODEL_NAME`
2. tokenizes the reward dataset
3. trains a sequence regressor
4. writes model artifacts to `REWARD_MODEL_OUTPUT_DIR`
5. writes metrics and plots to `results/`

Output artifacts:

- `results/reward_model_metrics.json`
- `results/reward_model_loss.png`
- `results/reward_dataset_preview.json`

### Stage D: generate policy dataset

`train_trl.py` can now use the trained reward model.

For each env state:

1. build a memory-aware policy prompt
2. generate candidate responses
3. score them with the learned reward model
4. keep the best candidate
5. store prompt/completion training pairs

### Stage E: train policy model

`train_trl.py` then:

1. loads `POLICY_MODEL_NAME`
2. converts prompt/completion pairs into SFT data
3. trains using:
   - `trl.SFTTrainer` if available
   - otherwise `transformers.Trainer`
4. saves the fine-tuned policy model

Output artifacts:

- `results/training_metrics.json`
- `results/loss_curve.png`
- `results/reward_curve.png`
- `results/before_after.md`
- `results/policy_dataset_preview.json`

### Stage F: evaluate before vs after

The policy training script evaluates the base policy and the fine-tuned policy inside the same environment.

It reports:

- average score
- success rate
- average steps
- per-task before/after comparison

This is the key evidence judges want: not just “we trained,” but “the trained model improved inside the environment.”

## 10. Inference and Deployment Path

### 10.1 HTTP server

[server/app.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/server/app.py) exposes the OpenEnv interface:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`

The env state is persisted in a signed compressed cookie, so the server stays replica-friendly.

### 10.2 HTTP client

[src/client.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/client.py) mirrors the env API and also exports/imports the session token.

### 10.3 Durable runner

[src/runner.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/src/runner.py) wraps the client and `AgentMemory` together so a full episode can be paused and resumed.

### 10.4 LLM inference script

[inference.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/inference.py) is the external-model inference path.

It does not directly free-run a model.
Instead it:

1. queries the env over HTTP
2. uses the deterministic controller to choose the correct high-level skill
3. builds a memory-aware prompt
4. asks an external LLM to lightly polish the draft response
5. rejects unsafe or stage-breaking rewrites

This makes inference more stable and benchmark-friendly.

## 11. Docker and Colab Workflow

The container workflow is controlled by:

- [docker-compose.yml](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/docker-compose.yml)
- [.env.example](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/.env.example)
- [COLAB_README.md](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/COLAB_README.md)

Two model names are controlled by `.env`:

```env
POLICY_MODEL_NAME=distilgpt2
REWARD_MODEL_NAME=distilroberta-base
```

The recommended practical swap is:

```env
POLICY_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
REWARD_MODEL_NAME=distilroberta-base
```

Run order:

1. `docker compose up --build env-server`
2. `docker compose --profile reward up --build reward-trainer`
3. `docker compose --profile train up --build trainer`

## 12. What Is Novel Here

The main innovation is not “chatbot with a reward.”

It is the combination of:

- therapist-style personalized task framing
- multi-session continuity
- durable memory with rolling summaries
- future-oriented reward
- separate learned reward and policy models
- budget-aware long-horizon scoring

This makes the environment more than a static conversation simulator. It becomes a trainable benchmark for whether a model can sustain therapeutic continuity over time.

## 13. Suggested Presentation Story

For the presentation, the clean story is:

### Slide 1: capability gap

Current LLM assistants can sound empathetic for one turn, but they often fail at:

- long-term continuity
- delayed follow-through
- safe pacing
- remembering what mattered across sessions

### Slide 2: our environment

We built a therapist-style multi-session benchmark where the agent must:

- build trust
- surface the hidden issue
- carry memory across sessions
- manage safety when needed
- stay within cost/time budgets

### Slide 3: why this is hard

The agent cannot solve the task with one good sentence.
It must avoid:

- premature advice
- repetition
- losing the thread
- forgetting prior disclosure

### Slide 4: reward design

Explain the three reward layers:

- immediate turn quality
- future-oriented trajectory quality
- long-horizon continuity and budget discipline

### Slide 5: two-model training

Explain the pipeline:

1. env generates candidate-response data
2. reward model learns to score future-oriented quality
3. policy model learns responses favored by that reward model
4. we evaluate before vs after in the same environment

### Slide 6: results

Show:

- reward-model loss curve
- policy loss curve
- before/after reward plot
- one short transcript example

### Slide 7: why it matters

This is a benchmark for long-horizon personalized support agents, not just single-turn style matching.

## 14. Short Verbal Summary

If you need a 20-second version for judges:

> We built an OpenEnv environment for training a therapist-style assistant across multiple sessions. The environment tracks hidden seeker state, memory continuity, delayed consequences, and safety-sensitive progression. We train a separate reward model to predict future-oriented conversational quality, then use that reward model to guide policy-model training and show before/after improvement inside the same benchmark.

## 15. Practical Caveats

To be precise during the presentation:

- this is a therapist-style support environment, not a licensed therapist
- the seeker is a structured simulator, not a second LLM
- the reward model learns from env-generated future-oriented labels
- the current version focuses on conversational continuity, not external tool use

That framing is both honest and strong.
