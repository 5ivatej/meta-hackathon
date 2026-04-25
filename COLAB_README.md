# Colab Runbook

This file is the shortest path to running the environment and training pipeline in Google Colab with Docker Compose.

## 1. What This Setup Does

There are two Compose services:

- `env-server`: runs the OpenEnv-compatible FastAPI environment
- `trainer`: runs `train_trl.py` using the values from `.env`

The model name is controlled from `.env`.

## 2. Important Files

- [docker-compose.yml](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/docker-compose.yml)
- [.env.example](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/.env.example)
- [Dockerfile](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/Dockerfile)
- [train_trl.py](/Users/5ivatej/Desktop/meta-hackathon/meta-hackathon/train_trl.py)

## 3. Default `.env` Values

Current defaults:

```env
PORT=7860
MODEL_NAME=distilgpt2
TRAIN_EPISODES_PER_TASK=8
TRAIN_EPOCHS=1
TRAIN_BATCH_SIZE=2
TRAIN_MAX_LENGTH=1024
TRAIN_OUTPUT_DIR=artifacts/trl_sft
TRAIN_RESULTS_DIR=results
```

If you want a different model, change:

```env
MODEL_NAME=your-model-name
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

## 6. Run Training

In a separate cell:

```bash
!docker compose --profile train up --build trainer
```

This runs:

```bash
python3 train_trl.py \
  --model-name ${MODEL_NAME} \
  --episodes-per-task ${TRAIN_EPISODES_PER_TASK} \
  --epochs ${TRAIN_EPOCHS} \
  --batch-size ${TRAIN_BATCH_SIZE} \
  --max-length ${TRAIN_MAX_LENGTH} \
  --output-dir ${TRAIN_OUTPUT_DIR} \
  --results-dir ${TRAIN_RESULTS_DIR}
```

## 7. Expected Outputs

After training, check:

```bash
!ls results
!ls artifacts
```

Expected files:

- `results/training_metrics.json`
- `results/loss_curve.png`
- `results/reward_curve.png`
- `results/before_after.md`
- `results/teacher_dataset_preview.json`
- `artifacts/trl_sft/` with model files

## 8. Useful Variants

### Train with more data

Edit `.env`:

```env
TRAIN_EPISODES_PER_TASK=16
```

### Train longer

```env
TRAIN_EPOCHS=2
```

### Change model

```env
MODEL_NAME=distilgpt2
```

Replace with your preferred local/Hugging Face model name.

## 9. Common Issues

### `git add .` is slow

Model outputs are written under `artifacts/`, and that directory is ignored by `.gitignore`.

### Training fails on downloads

If the selected model is not already cached, `transformers` will try to download it.
That means Colab needs outbound network access for the model pull.

### `trl` / `transformers` / `accelerate` mismatch

The training script already includes compatibility fallbacks for the common issues we hit locally, but if Colab has a very different preinstalled stack, dependency upgrades may still be needed.

## 10. Minimal Recommended Flow

1. Edit `.env` if you want a different `MODEL_NAME`
2. Run:

```bash
!docker compose up --build env-server
```

3. Run:

```bash
!docker compose --profile train up --build trainer
```

4. Inspect:

```bash
!cat results/before_after.md
```

5. Download or view:

- `results/loss_curve.png`
- `results/reward_curve.png`

## 11. Practical Note

If Docker is not available in the Colab runtime you choose, the fallback is to run the same script directly:

```bash
!python3 train_trl.py --model-name distilgpt2 --episodes-per-task 8 --epochs 1 --batch-size 2
```

But if Docker Compose works, use the compose path so the environment is reproducible.
