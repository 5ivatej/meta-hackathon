"""Train the policy model, optionally guided by a learned reward model."""
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
import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.agentic import AgentMemory, SkillDecision, SkillRouter, build_default_skills
from src.env import ESCEnv
from src.models import Action
from src.training_utils import (
    EvalEpisode,
    PolicySample,
    build_policy_prompt,
    build_reward_text,
    candidate_responses,
    collect_policy_teacher_dataset,
    summarize_eval,
)
from src.tasks import TASKS

try:
    from trl import SFTConfig, SFTTrainer

    HAS_TRL = True
except Exception:
    SFTConfig = None
    SFTTrainer = None
    HAS_TRL = False


_unwrap_signature = inspect.signature(Accelerator.unwrap_model)
if "keep_torch_compile" not in _unwrap_signature.parameters:
    _orig_unwrap_model = Accelerator.unwrap_model

    def _compat_unwrap_model(self, model, *args, keep_torch_compile=None, **kwargs):
        del keep_torch_compile
        return _orig_unwrap_model(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _compat_unwrap_model


def build_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def dataset_from_samples(samples: Sequence[PolicySample], tokenizer) -> Dataset:
    eos = tokenizer.eos_token or ""
    rows = [{"text": f"{sample.prompt}\n\nAssistant: {sample.completion}{eos}"} for sample in samples]
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
        return encoded

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


def split_samples(samples: Sequence[Any], eval_ratio: float = 0.2) -> tuple[List[Any], List[Any]]:
    if not samples:
        return [], []
    split_idx = max(1, math.floor(len(samples) * (1.0 - eval_ratio)))
    return list(samples[:split_idx]), list(samples[split_idx:])


def load_reward_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def score_candidates_with_reward_model(
    reward_model,
    reward_tokenizer,
    observation,
    memory: AgentMemory,
    candidates: Sequence[tuple[str, str]],
) -> List[tuple[str, str, float]]:
    scored: List[tuple[str, str, float]] = []
    device = next(reward_model.parameters()).device
    for source, candidate in candidates:
        text = build_reward_text(observation, memory, candidate)
        inputs = reward_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits
        score = float(logits.squeeze().item())
        scored.append((source, candidate, score))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored


def collect_reward_guided_policy_dataset(
    episodes_per_task: int,
    reward_model_path: str,
) -> List[PolicySample]:
    reward_model, reward_tokenizer = load_reward_model(reward_model_path)
    env = ESCEnv()
    router = SkillRouter()
    skills = build_default_skills()
    samples: List[PolicySample] = []

    for task_id in TASKS:
        for _ in range(episodes_per_task):
            memory = AgentMemory()
            memory.reset(task_id)
            obs = env.reset(task_id=task_id).observation

            while True:
                memory.observe(obs)
                decision = router.choose(obs, memory)
                skill = skills[decision.skill_name]
                prompt = build_policy_prompt(
                    observation=obs,
                    memory=memory,
                    decision=decision,
                    skill_instruction=skill.llm_instruction(obs, memory, decision),
                )
                scored_candidates = score_candidates_with_reward_model(
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    observation=obs,
                    memory=memory,
                    candidates=candidate_responses(obs, memory),
                )
                best_source, best_message, _best_score = scored_candidates[0]
                samples.append(
                    PolicySample(
                        task_id=task_id,
                        prompt=prompt,
                        completion=best_message,
                        session_index=obs.session_index,
                        turn=obs.turn,
                        source=f"reward_model:{best_source}",
                    )
                )
                memory.remember(decision.skill_name, best_message)
                result = env.step(Action(message=best_message))
                obs = result.observation
                if result.done:
                    break

    return samples


def train_model(
    model_name: str,
    output_dir: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_length: int,
    epochs: float,
    batch_size: int,
) -> tuple[Any, Any, List[Dict[str, Any]], str]:
    model, tokenizer = build_model_and_tokenizer(model_name)
    training_backend = "trl_sft" if HAS_TRL else "transformers_trainer"

    if HAS_TRL and SFTTrainer is not None and SFTConfig is not None:
        config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to=[],
            max_length=max_length,
            learning_rate=2e-5,
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    else:
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
            "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
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
    return trainer.model, tokenizer, list(trainer.state.log_history), training_backend


def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 96) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first_line = text.split("\n")[0].strip()
    return first_line or "That sounds really hard. I'm here with you. What feels most important to say right now?"


def evaluate_model(model, tokenizer) -> List[EvalEpisode]:
    env = ESCEnv()
    router = SkillRouter()
    skills = build_default_skills()
    episodes: List[EvalEpisode] = []

    for task_id in TASKS:
        memory = AgentMemory()
        memory.reset(task_id)
        obs = env.reset(task_id=task_id).observation
        transcript = [f"Seeker: {obs.seeker_utterance}"]

        while True:
            memory.observe(obs)
            decision = router.choose(obs, memory)
            skill = skills[decision.skill_name]
            prompt = build_policy_prompt(
                observation=obs,
                memory=memory,
                decision=decision,
                skill_instruction=skill.llm_instruction(obs, memory, decision),
            )
            message = generate_reply(model, tokenizer, prompt)
            memory.remember(decision.skill_name, message)
            result = env.step(Action(message=message))
            transcript.append(f"Agent: {message}")
            transcript.append(f"Seeker: {result.observation.seeker_utterance}")
            obs = result.observation
            if result.done:
                final = result.info.get("final", {})
                episodes.append(
                    EvalEpisode(
                        task_id=task_id,
                        score=float(final.get("score", 0.0)),
                        success=bool(final.get("success", 0.0) >= 1.0),
                        steps=obs.turn,
                        completion=float(final.get("completion", 0.0)),
                        final_resolution=float(final.get("final_resolution", 0.0)),
                        transcript=transcript,
                    )
                )
                break

    return episodes


def save_loss_plot(log_history: Sequence[Dict[str, Any]], output_path: Path) -> None:
    losses = [(entry.get("step"), entry.get("loss")) for entry in log_history if "loss" in entry and "step" in entry]
    if not losses:
        return
    xs = [step for step, _ in losses]
    ys = [loss for _, loss in losses]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Policy Training Loss")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_reward_plot(before: Dict[str, Any], after: Dict[str, Any], output_path: Path) -> None:
    labels = [ep["task_id"] for ep in before["episodes"]]
    before_scores = [ep["score"] for ep in before["episodes"]]
    after_scores = [ep["score"] for ep in after["episodes"]]
    xs = range(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([x - width / 2 for x in xs], before_scores, width=width, label="before")
    plt.bar([x + width / 2 for x in xs], after_scores, width=width, label="after")
    plt.xticks(list(xs), labels, rotation=15)
    plt.ylabel("score")
    plt.title("Policy Before vs After Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_before_after_md(before: Dict[str, Any], after: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Before / After Policy Training",
        "",
        "| Task | Before score | After score | Before success | After success |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    after_by_task = {ep["task_id"]: ep for ep in after["episodes"]}
    for ep in before["episodes"]:
        other = after_by_task[ep["task_id"]]
        lines.append(
            f"| {ep['task_id']} | {ep['score']:.3f} | {other['score']:.3f} | "
            f"{int(ep['success'])} | {int(other['success'])} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the policy model, optionally guided by a reward model.")
    parser.add_argument("--model-name", default="distilgpt2", help="Base policy model name or local path.")
    parser.add_argument("--reward-model-path", default="", help="Path to a trained reward model. If set, use it to rerank candidates.")
    parser.add_argument("--episodes-per-task", type=int, default=8, help="Teacher rollout episodes per task.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Training sequence length.")
    parser.add_argument("--output-dir", default="artifacts/policy_model", help="Fine-tuned policy model output directory.")
    parser.add_argument("--results-dir", default="results", help="Metrics and plot output directory.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.reward_model_path and Path(args.reward_model_path).exists():
        samples = collect_reward_guided_policy_dataset(
            episodes_per_task=args.episodes_per_task,
            reward_model_path=args.reward_model_path,
        )
        dataset_source = "reward_model_reranked"
    else:
        samples = collect_policy_teacher_dataset(episodes_per_task=args.episodes_per_task)
        dataset_source = "teacher"

    train_samples, eval_samples = split_samples(samples)
    tokenizer_probe = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer_probe.pad_token is None:
        tokenizer_probe.pad_token = tokenizer_probe.eos_token
    train_dataset = dataset_from_samples(train_samples, tokenizer_probe)
    eval_dataset = dataset_from_samples(eval_samples or train_samples[:1], tokenizer_probe)

    before_model, before_tokenizer = build_model_and_tokenizer(args.model_name)
    before_eval = summarize_eval(evaluate_model(before_model, before_tokenizer))
    del before_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_model, trained_tokenizer, log_history, backend = train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    after_eval = summarize_eval(evaluate_model(trained_model, trained_tokenizer))

    metrics = {
        "policy_model_name": args.model_name,
        "reward_model_path": args.reward_model_path,
        "dataset_source": dataset_source,
        "training_backend": backend,
        "episodes_per_task": args.episodes_per_task,
        "sample_count": len(samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "before_eval": before_eval,
        "after_eval": after_eval,
        "log_history": log_history,
    }

    (results_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_loss_plot(log_history, results_dir / "loss_curve.png")
    save_reward_plot(before_eval, after_eval, results_dir / "reward_curve.png")
    write_before_after_md(before_eval, after_eval, results_dir / "before_after.md")
    (results_dir / "policy_dataset_preview.json").write_text(
        json.dumps([asdict(sample) for sample in samples[:10]], indent=2),
        encoding="utf-8",
    )

    print(f"Wrote policy metrics to {results_dir / 'training_metrics.json'}")
    print(f"Wrote policy plots to {results_dir / 'loss_curve.png'} and {results_dir / 'reward_curve.png'}")
    print(f"Wrote policy report to {results_dir / 'before_after.md'}")


if __name__ == "__main__":
    main()
