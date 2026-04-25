"""Stage 2: train a future-oriented reward model from simulated data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import RewardModelConfig
from .io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary future-oriented reward model.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-threshold", type=float, default=0.65)
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
        positive_threshold=args.positive_threshold,
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

    examples = [_convert_sim_record_to_classifier_example(record, config.positive_threshold) for record in raw_records]
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

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
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
        compute_metrics=_compute_classifier_metrics if eval_dataset is not None else None,
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
                "positive_threshold": config.positive_threshold,
                "model_name": config.model_name,
                "freeze_backbone": config.freeze_backbone,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _convert_sim_record_to_classifier_example(record: dict[str, Any], positive_threshold: float) -> dict[str, Any]:
    prompt = str(record.get("prompt_for_policy") or "").strip()
    response = str(record.get("response") or "").strip()
    reward = float(record.get("scalar_reward", 0.0))
    text = f"{prompt}\n\nAssistant response:\n{response}"
    return {"text": text, "label": 1 if reward >= positive_threshold else 0}


def _compute_classifier_metrics(eval_prediction: Any) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_prediction
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    positive_rate = float(preds.mean()) if len(preds) else 0.0
    return {"accuracy": accuracy, "positive_rate": positive_rate}


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
    probs = _positive_class_probabilities(logits)
    preds = np.argmax(logits, axis=-1)

    records = []
    for idx, (prob, pred, label) in enumerate(zip(probs, preds, labels)):
        row = audit_dataset[idx]
        records.append(
            {
                "index": idx,
                "text": row.get("text"),
                "label": int(label),
                "predicted_label": int(pred),
                "positive_probability": float(prob),
                "is_correct": bool(int(pred) == int(label)),
            }
        )

    audit_path = Path(output_dir) / "reward_model_audit.jsonl"
    write_jsonl(audit_path, records)

    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for record in records:
        pred = record["predicted_label"]
        label = record["label"]
        if pred == 1 and label == 1:
            confusion["tp"] += 1
        elif pred == 0 and label == 0:
            confusion["tn"] += 1
        elif pred == 1 and label == 0:
            confusion["fp"] += 1
        else:
            confusion["fn"] += 1

    summary = {
        "num_examples": len(records),
        "accuracy": float(sum(1 for record in records if record["is_correct"]) / max(1, len(records))),
        "avg_positive_probability": float(sum(record["positive_probability"] for record in records) / max(1, len(records))),
        "confusion_matrix": confusion,
        "top_false_positives": sorted(
            [record for record in records if record["predicted_label"] == 1 and record["label"] == 0],
            key=lambda item: item["positive_probability"],
            reverse=True,
        )[:20],
        "top_false_negatives": sorted(
            [record for record in records if record["predicted_label"] == 0 and record["label"] == 1],
            key=lambda item: item["positive_probability"],
        )[:20],
    }
    summary_path = Path(output_dir) / "reward_model_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _positive_class_probabilities(logits) -> Any:
    import numpy as np

    if len(logits.shape) == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    return probs[:, 1]
