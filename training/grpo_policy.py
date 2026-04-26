"""Stage 3: GRPO policy optimization with future reward + think-format reward."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Iterable, List

from .config import GRPOTrainingConfig
from .datasets import build_prompt_records_from_simulation, load_seed_examples
from .io import read_jsonl


THINK_RESPONSE_PATTERN = re.compile(
    r"^\s*<think>.*?</think>\s*<response>.*?</response>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO with learned future reward + think format reward.")
    parser.add_argument("--prompt-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reward-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--think-format-weight", type=float, default=0.2)
    args = parser.parse_args()

    config = GRPOTrainingConfig(
        model_name=args.model_name,
        reward_model_dir=args.reward_model_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        temperature=args.temperature,
        think_format_weight=args.think_format_weight,
    )
    _train_grpo(config=config, prompt_jsonl=args.prompt_jsonl)


def _train_grpo(config: GRPOTrainingConfig, prompt_jsonl: str) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    prompt_records = _load_prompt_records(prompt_jsonl)
    dataset = Dataset.from_list(prompt_records)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_dir)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_dir)
    reward_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model.to(device)

    def future_reward_func(prompts, completions, **kwargs):
        completion_texts = [_normalize_completion_text(completion) for completion in completions]
        response_texts = [_extract_response_body(text) for text in completion_texts]
        combined = [
            f"{prompt}\n\nAssistant response:\n{response}"
            for prompt, response in zip(prompts, response_texts)
        ]
        batch = reward_tokenizer(
            combined,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length + config.max_completion_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            logits = reward_model(**batch).logits
            preds = logits.view(-1).float().clamp(0.0, 1.0)
        return [float(x) for x in preds.cpu().tolist()]

    def think_format_reward_func(completions, **kwargs):
        completion_texts = [_normalize_completion_text(completion) for completion in completions]
        return [
            float(config.think_format_weight) if THINK_RESPONSE_PATTERN.match(text) else 0.0
            for text in completion_texts
        ]

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        report_to="none",
        logging_steps=5,
        save_steps=50,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        remove_unused_columns=False,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[future_reward_func, think_format_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config.output_dir)


def _load_prompt_records(path: str) -> List[dict[str, Any]]:
    raw_records = read_jsonl(path)
    prompt_records = build_prompt_records_from_simulation(raw_records)
    if not prompt_records:
        raise SystemExit(f"No prompt records found in {path}")
    return prompt_records


def _normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content") or "")
    return str(completion)


def _extract_response_body(text: str) -> str:
    match = re.search(r"<response>(.*?)</response>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


if __name__ == "__main__":
    main()
