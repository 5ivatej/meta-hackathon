"""Seed dataset helpers for simulation and training."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, List

from src.tasks import TASKS

from .io import read_jsonl


@dataclass
class SeedExample:
    example_id: str
    source: str
    scenario_brief: str
    desired_outcome: str
    context_turns: List[dict[str, str]]
    task_id: str | None = None
    ground_truth_response: str | None = None
    emotion_type: str | None = None
    problem_type: str | None = None


def load_seed_examples(
    source: str,
    input_jsonl: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    max_examples: int | None = None,
) -> List[SeedExample]:
    if source == "jsonl":
        if not input_jsonl:
            raise SystemExit("--input-jsonl is required when --examples-source=jsonl")
        return _limit(_load_seed_examples_from_jsonl(input_jsonl), max_examples)
    if source == "tasks":
        return _limit(_load_seed_examples_from_tasks(), max_examples)
    if source == "esconv_hf":
        return _limit(_load_seed_examples_from_esconv_hf(dataset_name=dataset_name, dataset_split=dataset_split), max_examples)
    if source == "extes_hf":
        return _limit(_load_seed_examples_from_extes_hf(dataset_name=dataset_name, dataset_split=dataset_split), max_examples)
    if source == "extes_jsonl":
        if not input_jsonl:
            raise SystemExit("--input-jsonl is required when --examples-source=extes_jsonl")
        return _limit(_load_seed_examples_from_extes_jsonl(input_jsonl), max_examples)
    raise SystemExit(f"Unsupported example source: {source}")


def _load_seed_examples_from_jsonl(path: str) -> List[SeedExample]:
    examples: List[SeedExample] = []
    for idx, record in enumerate(read_jsonl(path)):
        example_id = str(record.get("example_id") or record.get("id") or f"jsonl-{idx}")
        scenario_brief = str(record.get("scenario_brief") or record.get("context") or "")
        context_turns = _coerce_context_turns(record.get("context_turns"))
        opening_user_message = str(record.get("opening_user_message") or record.get("user_message") or record.get("opening") or "")
        if not context_turns and opening_user_message:
            context_turns = [{"role": "user", "content": opening_user_message}]
        desired_outcome = str(
            record.get("desired_outcome")
            or record.get("target_outcome")
            or record.get("goal")
            or "Help the user feel heard, understood, and more emotionally grounded."
        )
        if not scenario_brief or not context_turns:
            raise SystemExit(f"Seed example {example_id} is missing required fields.")
        examples.append(
            SeedExample(
                example_id=example_id,
                source="jsonl",
                scenario_brief=scenario_brief,
                desired_outcome=desired_outcome,
                context_turns=context_turns,
                task_id=record.get("task_id"),
                ground_truth_response=record.get("ground_truth_response"),
                emotion_type=record.get("emotion_type"),
                problem_type=record.get("problem_type"),
            )
        )
    return examples


def _load_seed_examples_from_tasks() -> List[SeedExample]:
    examples: List[SeedExample] = []
    for task_id, task in TASKS.items():
        examples.append(
            SeedExample(
                example_id=f"task-{task_id}",
                source="tasks",
                task_id=task_id,
                scenario_brief=task.persona.scenario_brief,
                desired_outcome=(
                    f"Reach {task.required_final_stage} with trust >= {task.min_final_trust:.2f}, "
                    f"distress <= {task.max_final_distress:.2f}, and keep the conversation emotionally safe."
                ),
                context_turns=[{"role": "user", "content": task.persona.surface_concern}],
            )
        )
    return examples


def _load_seed_examples_from_esconv_hf(dataset_name: str | None, dataset_split: str) -> List[SeedExample]:
    from datasets import load_dataset

    name = dataset_name or "thu-coai/esconv"
    dataset = load_dataset(name, split=dataset_split)
    examples: List[SeedExample] = []
    for row_index, row in enumerate(dataset):
        dialog = row.get("dialog") or row.get("dialogue") or []
        context_turns: List[dict[str, str]] = []
        for turn_index, turn in enumerate(dialog):
            speaker = str(turn.get("speaker") or turn.get("role") or "").lower()
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            normalized_role = "assistant" if speaker in {"sys", "supporter", "assistant"} else "user"
            if normalized_role == "assistant" and context_turns:
                scenario = _build_dataset_scenario_brief(row)
                desired_outcome = _build_dataset_desired_outcome(row)
                examples.append(
                    SeedExample(
                        example_id=f"esconv-{row_index}-turn-{turn_index}",
                        source="esconv_hf",
                        scenario_brief=scenario,
                        desired_outcome=desired_outcome,
                        context_turns=list(context_turns),
                        ground_truth_response=text,
                        emotion_type=_optional_str(row.get("emotion_type")),
                        problem_type=_optional_str(row.get("problem_type")),
                    )
                )
            context_turns.append({"role": normalized_role, "content": text})
    return examples


def _load_seed_examples_from_extes_jsonl(path: str) -> List[SeedExample]:
    examples: List[SeedExample] = []
    for idx, record in enumerate(read_jsonl(path)):
        if record.get("example"):
            parsed = _parse_extes_serialized_example(str(record["example"]))
            examples.append(
                SeedExample(
                    example_id=str(record.get("example_id") or record.get("id") or f"extes-{idx}"),
                    source="extes_jsonl",
                    scenario_brief=str(record.get("scenario_brief") or parsed["scenario_brief"]),
                    desired_outcome=str(record.get("desired_outcome") or parsed["desired_outcome"]),
                    context_turns=parsed["context_turns"],
                    ground_truth_response=str(record.get("ground_truth_response") or parsed["ground_truth_response"]),
                    emotion_type=_optional_str(record.get("emotion_type")),
                    problem_type=_optional_str(record.get("problem_type")),
                )
            )
            continue
        context_turns = _coerce_context_turns(record.get("context_turns") or record.get("dialogue") or record.get("dialog"))
        if not context_turns:
            opening = str(record.get("opening_user_message") or record.get("user_message") or "")
            if opening:
                context_turns = [{"role": "user", "content": opening}]
        scenario_brief = str(record.get("scenario_brief") or record.get("situation") or record.get("context") or "")
        desired_outcome = str(
            record.get("desired_outcome")
            or record.get("goal")
            or "Provide emotionally supportive help and move the conversation toward emotional relief or resolution."
        )
        if not scenario_brief or not context_turns:
            raise SystemExit(f"ExTES-style record {idx} is missing `scenario_brief` or dialogue context.")
        examples.append(
            SeedExample(
                example_id=str(record.get("example_id") or record.get("id") or f"extes-{idx}"),
                source="extes_jsonl",
                scenario_brief=scenario_brief,
                desired_outcome=desired_outcome,
                context_turns=context_turns,
                ground_truth_response=record.get("ground_truth_response"),
                emotion_type=_optional_str(record.get("emotion_type")),
                problem_type=_optional_str(record.get("problem_type")),
            )
        )
    return examples


def _load_seed_examples_from_extes_hf(dataset_name: str | None, dataset_split: str) -> List[SeedExample]:
    from datasets import load_dataset

    name = dataset_name or "ailover/ExTES"
    dataset = load_dataset(name, split=dataset_split)
    examples: List[SeedExample] = []
    for row_index, row in enumerate(dataset):
        serialized = str(row.get("example") or "").strip()
        if not serialized:
            continue
        parsed = _parse_extes_serialized_example(serialized)
        examples.append(
            SeedExample(
                example_id=f"extes-{row_index}",
                source="extes_hf",
                scenario_brief=parsed["scenario_brief"],
                desired_outcome=parsed["desired_outcome"],
                context_turns=parsed["context_turns"],
                ground_truth_response=parsed["ground_truth_response"],
            )
        )
    return examples


def build_prompt_records_from_simulation(records: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    prompts: List[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        prompt = str(record.get("prompt_for_policy") or "").strip()
        if not prompt:
            continue
        key = (prompt, str(record.get("example_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        prompts.append(
            {
                "prompt": prompt,
                "example_id": record.get("example_id"),
                "task_id": record.get("env_task_id") or record.get("task_id"),
                "source": record.get("source"),
                "emotion_type": record.get("emotion_type"),
                "problem_type": record.get("problem_type"),
            }
        )
    return prompts


def infer_task_id_for_seed(seed: SeedExample) -> str:
    if seed.task_id and seed.task_id in TASKS:
        return seed.task_id

    text = " ".join(
        part
        for part in [
            seed.scenario_brief,
            seed.desired_outcome,
            seed.emotion_type or "",
            seed.problem_type or "",
            " ".join(turn.get("content", "") for turn in seed.context_turns[-4:]),
        ]
        if part
    ).lower()

    crisis_markers = (
        "dark thoughts",
        "suicid",
        "self-harm",
        "hospital",
        "late at night",
        "panic",
        "unsafe",
        "overwhelmed",
    )
    relationship_markers = (
        "partner",
        "relationship",
        "separat",
        "divorce",
        "break up",
        "spare room",
        "at home",
    )
    work_markers = (
        "manager",
        "deadline",
        "coworker",
        "burnout",
        "work stress",
        "job",
        "office",
        "inbox",
    )

    if any(marker in text for marker in crisis_markers):
        return "crisis_fragile_trust"
    if any(marker in text for marker in relationship_markers):
        return "guarded_relationship"
    if any(marker in text for marker in work_markers):
        return "work_stress_venting"
    return "work_stress_venting"


def _coerce_context_turns(raw_turns: Any) -> List[dict[str, str]]:
    context_turns: List[dict[str, str]] = []
    if not isinstance(raw_turns, list):
        return context_turns
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("speaker") or "").lower()
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if not content:
            continue
        normalized_role = "assistant" if role in {"sys", "system", "assistant", "supporter"} else "user"
        context_turns.append({"role": normalized_role, "content": content})
    return context_turns


def _build_dataset_scenario_brief(row: dict[str, Any]) -> str:
    parts: List[str] = []
    if row.get("problem_type"):
        parts.append(f"Problem type: {row['problem_type']}.")
    if row.get("emotion_type"):
        parts.append(f"Emotion type: {row['emotion_type']}.")
    if row.get("situation"):
        parts.append(str(row["situation"]).strip())
    return " ".join(part for part in parts if part).strip() or "Emotional support dialogue context."


def _build_dataset_desired_outcome(row: dict[str, Any]) -> str:
    return (
        "Help the user feel heard, emotionally supported, and move the conversation toward relief or a constructive next step."
    )


def _parse_extes_serialized_example(example: str) -> dict[str, Any]:
    cleaned = example.strip()
    cleaned = cleaned.replace("<s>", "").replace("</s>", "").strip()

    if "[/INST]" not in cleaned:
        raise SystemExit("ExTES serialized example is missing `[/INST]` and cannot be parsed.")

    instruction_block, assistant_text = cleaned.rsplit("[/INST]", 1)
    assistant_text = assistant_text.strip()
    instruction_text = instruction_block.split("[INST]")[-1].strip()

    request_marker = "Please respond the following request:"
    lowered_instruction = instruction_text.lower()
    lowered_marker = request_marker.lower()
    marker_index = lowered_instruction.rfind(lowered_marker)
    if marker_index >= 0:
        system_prefix = instruction_text[:marker_index].strip()
        user_text = instruction_text[marker_index + len(request_marker):].strip()
    else:
        system_prefix = ""
        user_text = instruction_text.strip()

    system_prefix = re.sub(r"\s+", " ", system_prefix).strip()
    user_text = re.sub(r"\s+", " ", user_text).strip()
    assistant_text = re.sub(r"\s+", " ", assistant_text).strip()

    if not user_text or not assistant_text:
        raise SystemExit("ExTES serialized example did not contain a usable user prompt and assistant response.")

    scenario_brief = (
        system_prefix
        or "Extended emotional support dialogue example derived from an instruction-style ESC dataset."
    )
    desired_outcome = "Provide an emotionally supportive response that is empathetic, relevant, and encouraging."

    return {
        "scenario_brief": scenario_brief,
        "desired_outcome": desired_outcome,
        "context_turns": [{"role": "user", "content": user_text}],
        "ground_truth_response": assistant_text,
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _limit(examples: List[SeedExample], max_examples: int | None) -> List[SeedExample]:
    if max_examples is None or max_examples <= 0:
        return examples
    return examples[:max_examples]
