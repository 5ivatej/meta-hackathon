# Colab Runbook

This file is the shortest path to running the environment and training pipeline in Google Colab with Docker Compose.

## 1. What This Setup Does

There are two Compose services:

- `env-server`: runs the OpenEnv-compatible FastAPI environment
- `reward-trainer`: trains the future-oriented reward model
- `trainer`: trains the policy model, optionally using the trained reward model

The policy model name and reward model name are both controlled from `.env`.

## 2. Important Files

- [docker-compose.yml](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/docker-compose.yml)
- [.env.example](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/.env.example)
- [Dockerfile](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/Dockerfile)
- [train_trl.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/train_trl.py)

## 3. Default `.env` Values

Current defaults:

```env
PORT=7860
HF_HOME=/root/.cache/huggingface
POLICY_MODEL_NAME=distilgpt2
REWARD_MODEL_NAME=distilroberta-base
REWARD_TRAIN_EPISODES_PER_TASK=8
REWARD_TRAIN_EPOCHS=1
REWARD_TRAIN_BATCH_SIZE=8
REWARD_TRAIN_MAX_LENGTH=1024
REWARD_MODEL_OUTPUT_DIR=artifacts/reward_model
POLICY_TRAIN_EPISODES_PER_TASK=8
POLICY_TRAIN_EPOCHS=1
POLICY_TRAIN_BATCH_SIZE=2
POLICY_TRAIN_MAX_LENGTH=1024
POLICY_MODEL_OUTPUT_DIR=artifacts/policy_model
TRAIN_RESULTS_DIR=results
```

If you want different models, change:

```env
POLICY_MODEL_NAME=your-policy-model
REWARD_MODEL_NAME=your-reward-model
```

Good practical starting point:

```env
POLICY_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
REWARD_MODEL_NAME=distilroberta-base
```

## 4. Colab Setup

In Colab, after cloning the repo and moving into it:

```bash
!pwd
!ls
```

If Docker is available in your Colab environment, use the commands below directly.

## 5. Run The Environment Server

```bash
!docker compose up --build env-server
```

This starts the server on port `7860` by default.

## 6. Train The Reward Model

In a separate cell:

```bash
!docker compose --profile reward up --build reward-trainer
```

This runs `train_reward_model.py` and writes a trained reward model under:

```bash
artifacts/reward_model/
```

## 7. Train The Policy Model

Then run:

```bash
!docker compose --profile train up --build trainer
```

This runs `train_trl.py`, which now uses:

- `POLICY_MODEL_NAME` as the trainable policy model
- `REWARD_MODEL_OUTPUT_DIR` as the learned reward model path

The policy training command is:

```bash
python3 train_trl.py \
  --model-name ${POLICY_MODEL_NAME} \
  --reward-model-path ${REWARD_MODEL_OUTPUT_DIR} \
  --episodes-per-task ${POLICY_TRAIN_EPISODES_PER_TASK} \
  --epochs ${POLICY_TRAIN_EPOCHS} \
  --batch-size ${POLICY_TRAIN_BATCH_SIZE} \
  --max-length ${POLICY_TRAIN_MAX_LENGTH} \
  --output-dir ${POLICY_MODEL_OUTPUT_DIR} \
  --results-dir ${TRAIN_RESULTS_DIR}
```

## 8. Expected Outputs

After training, check:

```bash
!ls results
!ls artifacts
```

Expected files:

- `results/training_metrics.json`
- `results/reward_model_metrics.json`
- `results/reward_model_loss.png`
- `results/loss_curve.png`
- `results/reward_curve.png`
- `results/before_after.md`
- `results/reward_dataset_preview.json`
- `results/policy_dataset_preview.json`
- `artifacts/reward_model/` with reward model files
- `artifacts/policy_model/` with policy model files

## 9. Useful Variants

### Train with more data

Edit `.env`:

```env
REWARD_TRAIN_EPISODES_PER_TASK=16
POLICY_TRAIN_EPISODES_PER_TASK=16
```

### Train longer

```env
REWARD_TRAIN_EPOCHS=2
POLICY_TRAIN_EPOCHS=2
```

### Change model

```env
POLICY_MODEL_NAME=distilgpt2
REWARD_MODEL_NAME=distilroberta-base
```

Replace with your preferred local/Hugging Face model name.

## 10. Common Issues

### `git add .` is slow

Model outputs are written under `artifacts/`, and that directory is ignored by `.gitignore`.

### Training fails on downloads

If the selected model is not already cached, `transformers` will try to download it.
That means Colab needs outbound network access for the model pull.

### `trl` / `transformers` / `accelerate` mismatch

The training script already includes compatibility fallbacks for the common issues we hit locally, but if Colab has a very different preinstalled stack, dependency upgrades may still be needed.

## 11. Minimal Recommended Flow

1. Edit `.env` if you want different `POLICY_MODEL_NAME` or `REWARD_MODEL_NAME`
2. Run:

```bash
!docker compose up --build env-server
```

3. Run:

```bash
!docker compose --profile reward up --build reward-trainer
```

4. Then run:

```bash
!docker compose --profile train up --build trainer
```

5. Inspect:

```bash
!cat results/before_after.md
```

6. Download or view:

- `results/loss_curve.png`
- `results/reward_curve.png`

## 12. Practical Note

If Docker is not available in the Colab runtime you choose, the fallback is to run the same script directly:

```bash
!python3 train_reward_model.py --model-name distilroberta-base --episodes-per-task 8 --epochs 1 --batch-size 8
!python3 train_trl.py --model-name distilgpt2 --reward-model-path artifacts/reward_model --episodes-per-task 8 --epochs 1 --batch-size 2
```

But if Docker Compose works, use the compose path so the environment is reproducible.
