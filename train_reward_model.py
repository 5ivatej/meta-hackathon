"""Train a future-oriented reward model from environment-generated scores."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.training_utils import RewardSample, collect_reward_dataset


_unwrap_signature = inspect.signature(Accelerator.unwrap_model)
if "keep_torch_compile" not in _unwrap_signature.parameters:
    _orig_unwrap_model = Accelerator.unwrap_model

    def _compat_unwrap_model(self, model, *args, keep_torch_compile=None, **kwargs):
        del keep_torch_compile
        return _orig_unwrap_model(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _compat_unwrap_model


def split_samples(samples: Sequence[Any], eval_ratio: float = 0.2) -> tuple[List[Any], List[Any]]:
    if not samples:
        return [], []
    split_idx = max(1, math.floor(len(samples) * (1.0 - eval_ratio)))
    return list(samples[:split_idx]), list(samples[split_idx:])


def dataset_from_samples(samples: Sequence[RewardSample]) -> Dataset:
    rows = [{"text": sample.text, "label": float(sample.future_oriented)} for sample in samples]
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = batch["label"]
        return encoded

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text", "label"])
    tokenized.set_format(type="torch")
    return tokenized


def compute_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    preds = np.squeeze(predictions)
    labels = np.squeeze(labels)
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    return {"mse": mse, "mae": mae}


def train_reward_model(
    model_name: str,
    output_dir: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_length: int,
    epochs: float,
    batch_size: int,
) -> tuple[Any, Any, List[Dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token or tokenizer.sep_token
    model.config.pad_token_id = tokenizer.pad_token_id

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    tokenized_eval = tokenize_dataset(eval_dataset, tokenizer, max_length=max_length)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        learning_rate=2e-5,
        remove_unused_columns=False,
    )
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.model, tokenizer, list(trainer.state.log_history)


def save_plot(log_history: Sequence[Dict[str, Any]], output_path: Path) -> None:
    losses = [(entry.get("step"), entry.get("loss")) for entry in log_history if "loss" in entry and "step" in entry]
    if not losses:
        return
    xs = [step for step, _ in losses]
    ys = [loss for _, loss in losses]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Reward Model Loss")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the future-oriented reward model.")
    parser.add_argument("--model-name", default="distilroberta-base", help="Base reward model name or local path.")
    parser.add_argument("--episodes-per-task", type=int, default=8, help="Environment rollout episodes per task.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Training sequence length.")
    parser.add_argument("--output-dir", default="artifacts/reward_model", help="Reward model output directory.")
    parser.add_argument("--results-dir", default="results", help="Metrics output directory.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_reward_dataset(episodes_per_task=args.episodes_per_task)
    train_samples, eval_samples = split_samples(samples)
    train_dataset = dataset_from_samples(train_samples)
    eval_dataset = dataset_from_samples(eval_samples or train_samples[:1])

    _, _, log_history = train_reward_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    metrics = {
        "reward_model_name": args.model_name,
        "episodes_per_task": args.episodes_per_task,
        "sample_count": len(samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "log_history": log_history,
    }
    (results_dir / "reward_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plot(log_history, results_dir / "reward_model_loss.png")
    (results_dir / "reward_dataset_preview.json").write_text(
        json.dumps([asdict(sample) for sample in samples[:10]], indent=2),
        encoding="utf-8",
    )

    print(f"Wrote reward model metrics to {results_dir / 'reward_model_metrics.json'}")
    print(f"Wrote reward model loss plot to {results_dir / 'reward_model_loss.png'}")


if __name__ == "__main__":
    main()
