"""Collect environment-generated data and train a reply model.

This script uses the long-horizon runtime in-process:

1. Roll out the deterministic skill-routed teacher against the environment.
2. Build prompt/completion examples from the actual environment state.
3. Fine-tune a causal LM with TRL `SFTTrainer` when available.
4. Fall back to plain `transformers.Trainer` when TRL is unavailable.
5. Evaluate the untrained and trained model on the same environment contract.
6. Write metrics and plots to `results/`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence
import inspect

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.agentic import AgentMemory, SkillDecision, SkillRouter, build_default_skills
from src.env import ESCEnv
from src.models import Action, Observation
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


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the response generator inside a therapist-style support agent.

    A deterministic controller has already selected the right conversational
    move. Your job is to produce the next response while preserving the memory,
    pacing, and safety needs of the ongoing therapy arc.

    Rules:
    - Stay warm, brief, and human.
    - Preserve continuity with prior sessions.
    - Ask at most one question.
    - Do not jump to advice before trust is built.
    - Keep safety follow-through explicit when the context requires it.
    """
).strip()


@dataclass
class Sample:
    task_id: str
    prompt: str
    completion: str
    session_index: int
    turn: int


@dataclass
class EvalEpisode:
    task_id: str
    score: float
    success: bool
    steps: int
    completion: float
    final_resolution: float
    transcript: List[str]


def build_training_prompt(
    observation: Observation,
    memory: AgentMemory,
    decision: SkillDecision,
    skill_instruction: str,
) -> str:
    return textwrap.dedent(
        f"""
        {SYSTEM_PROMPT}

        Selected skill: {decision.skill_name}
        Why this skill was selected: {decision.rationale}
        Skill directive: {skill_instruction}

        Scenario: {observation.scenario_brief}
        Public stage hint: {observation.stage_hint}
        Session: {observation.session_index}/{observation.sessions_total}
        Turn: {observation.turn}
        Remaining turns: {observation.remaining_turns}
        Remaining turns in session: {observation.remaining_session_turns}

        Durable memory and recent exchange:
        {memory.prompt_context(observation)}

        Seeker just said:
        "{observation.seeker_utterance}"

        Write the next reply now.
        """
    ).strip()


def collect_teacher_dataset(episodes_per_task: int) -> List[Sample]:
    env = ESCEnv()
    router = SkillRouter()
    skills = build_default_skills()
    samples: List[Sample] = []

    for task_id in TASKS:
        for _ in range(episodes_per_task):
            memory = AgentMemory()
            memory.reset(task_id)
            obs = env.reset(task_id=task_id).observation

            while True:
                memory.observe(obs)
                decision = router.choose(obs, memory)
                skill = skills[decision.skill_name]
                message = skill.render(obs, memory, decision)
                prompt = build_training_prompt(
                    observation=obs,
                    memory=memory,
                    decision=decision,
                    skill_instruction=skill.llm_instruction(obs, memory, decision),
                )
                samples.append(
                    Sample(
                        task_id=task_id,
                        prompt=prompt,
                        completion=message,
                        session_index=obs.session_index,
                        turn=obs.turn,
                    )
                )
                memory.remember(decision.skill_name, message)
                result = env.step(Action(message=message))
                obs = result.observation
                if result.done:
                    break

    return samples


def dataset_from_samples(samples: Sequence[Sample], tokenizer) -> Dataset:
    eos = tokenizer.eos_token or ""
    rows = [{"text": f"{sample.prompt}\n\nAssistant: {sample.completion}{eos}"} for sample in samples]
    return Dataset.from_list(rows)


def build_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


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
            prompt = build_training_prompt(
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


def summarize_eval(episodes: Sequence[EvalEpisode]) -> Dict[str, Any]:
    return {
        "avg_score": mean(ep.score for ep in episodes) if episodes else 0.0,
        "success_rate": mean(1.0 if ep.success else 0.0 for ep in episodes) if episodes else 0.0,
        "avg_steps": mean(ep.steps for ep in episodes) if episodes else 0.0,
        "episodes": [asdict(ep) for ep in episodes],
    }


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
    plt.title("Training Loss")
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
    plt.title("Before vs After Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_before_after_md(before: Dict[str, Any], after: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Before / After Training",
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
    lines.append("## Transcript Excerpts")
    lines.append("")
    for ep in after["episodes"]:
        lines.append(f"### {ep['task_id']}")
        lines.append("")
        for line in ep["transcript"][:10]:
            lines.append(f"- {line}")
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def split_samples(samples: Sequence[Sample], eval_ratio: float = 0.2) -> tuple[List[Sample], List[Sample]]:
    if not samples:
        return [], []
    split_idx = max(1, math.floor(len(samples) * (1.0 - eval_ratio)))
    return list(samples[:split_idx]), list(samples[split_idx:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a long-horizon therapist reply model with TRL/Transformers.")
    parser.add_argument("--model-name", default="distilgpt2", help="Base model name or local path.")
    parser.add_argument("--episodes-per-task", type=int, default=8, help="Teacher rollout episodes per task.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Training sequence length.")
    parser.add_argument("--output-dir", default="artifacts/trl_sft", help="Fine-tuned model output directory.")
    parser.add_argument("--results-dir", default="results", help="Metrics and plot output directory.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_teacher_dataset(episodes_per_task=args.episodes_per_task)
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
        "model_name": args.model_name,
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
    (results_dir / "teacher_dataset_preview.json").write_text(
        json.dumps([asdict(sample) for sample in samples[:10]], indent=2),
        encoding="utf-8",
    )

    print(f"Wrote metrics to {results_dir / 'training_metrics.json'}")
    print(f"Wrote plots to {results_dir / 'loss_curve.png'} and {results_dir / 'reward_curve.png'}")
    print(f"Wrote before/after report to {results_dir / 'before_after.md'}")


if __name__ == "__main__":
    main()
