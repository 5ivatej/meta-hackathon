"""Stage 2: train a scalar future-reward model from simulated data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import RewardModelConfig
from .io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a scalar future-oriented reward model.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    args = parser.parse_args()

    config = RewardModelConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_ratio=args.eval_ratio,
        freeze_backbone=args.freeze_backbone,
    )
    _train_reward_model(config=config, input_jsonl=args.input_jsonl)


def _train_reward_model(config: RewardModelConfig, input_jsonl: str) -> None:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    raw_records = read_jsonl(input_jsonl)
    if not raw_records:
        raise SystemExit(f"No records found in {input_jsonl}")

    examples = [_convert_sim_record_to_regression_example(record) for record in raw_records]
    dataset = Dataset.from_list(examples).shuffle(seed=42)
    if len(dataset) > 10 and config.eval_ratio > 0:
        split = dataset.train_test_split(test_size=config.eval_ratio, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=1,
        problem_type="regression",
    )
    if config.freeze_backbone:
        _freeze_backbone_parameters(model)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50 if eval_dataset is not None else None,
        save_steps=50,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_regression_metrics if eval_dataset is not None else None,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    _write_reward_model_audit(
        trainer=trainer,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        output_dir=config.output_dir,
    )

    metadata_path = Path(config.output_dir) / "reward_model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_jsonl": input_jsonl,
                "model_name": config.model_name,
                "freeze_backbone": config.freeze_backbone,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _convert_sim_record_to_regression_example(record: dict[str, Any]) -> dict[str, Any]:
    prompt = str(record.get("prompt_for_policy") or "").strip()
    response = str(record.get("response") or "").strip()
    reward = float(record.get("scalar_reward", 0.0))
    text = f"{prompt}\n\nAssistant response:\n{response}"
    return {"text": text, "label": reward}


def _compute_regression_metrics(eval_prediction: Any) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_prediction
    preds = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    preds = np.clip(preds, 0.0, 1.0)
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    corr = float(np.corrcoef(preds, labels)[0, 1]) if len(preds) > 1 else 0.0
    return {"mse": mse, "mae": mae, "corr": corr}


def _freeze_backbone_parameters(model) -> None:
    head_markers = ("score", "classifier", "classification_head")
    for name, param in model.named_parameters():
        if any(marker in name for marker in head_markers):
            param.requires_grad = True
        else:
            param.requires_grad = False


def _write_reward_model_audit(trainer, eval_dataset, train_dataset, output_dir: str) -> None:
    import numpy as np

    audit_dataset = eval_dataset if eval_dataset is not None else train_dataset
    prediction_output = trainer.predict(audit_dataset)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    preds = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    preds = np.clip(preds, 0.0, 1.0)

    records = []
    for idx, (pred, label) in enumerate(zip(preds, labels)):
        row = audit_dataset[idx]
        records.append(
            {
                "index": idx,
                "text": row.get("text"),
                "target_reward": float(label),
                "predicted_reward": float(pred),
                "absolute_error": float(abs(pred - label)),
            }
        )

    audit_path = Path(output_dir) / "reward_model_audit.jsonl"
    write_jsonl(audit_path, records)

    summary = {
        "num_examples": len(records),
        "mse": float(np.mean([(record["predicted_reward"] - record["target_reward"]) ** 2 for record in records] or [0.0])),
        "mae": float(np.mean([record["absolute_error"] for record in records] or [0.0])),
        "top_overestimates": sorted(
            records,
            key=lambda item: item["predicted_reward"] - item["target_reward"],
            reverse=True,
        )[:20],
        "top_underestimates": sorted(
            records,
            key=lambda item: item["predicted_reward"] - item["target_reward"],
        )[:20],
    }
    summary_path = Path(output_dir) / "reward_model_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
