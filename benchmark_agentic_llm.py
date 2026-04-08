"""Run a skill-routed LLM agent against the ESC environment.

This script shares the same deterministic skill router as `benchmark_agentic.py`
but lets an LLM realize each selected skill turn by turn. It writes Markdown and
JSON artifacts so the submission can show an agentic baseline with explicit
routing traces.

Required environment variables:
    API_BASE_URL
    MODEL_NAME
    ESC_ENV_URL

Authentication variables:
    HF_TOKEN or OPENAI_API_KEY or API_KEY

Example:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=gpt-4.1-mini
    export HF_TOKEN=<your-token>
    export ESC_ENV_URL=http://127.0.0.1:7860
    python3 benchmark_agentic_llm.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from openai import OpenAI

from src.agentic import AgentMemory, SkillRouter, build_default_skills
from src.client import ESCHttpClient
from src.models import Action, Observation

TASK_IDS = ["work_stress_venting", "guarded_relationship", "crisis_fragile_trust"]
TEMPERATURE = 0.5
MAX_TOKENS = 220

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the response generator inside a skill-routed emotional-support agent.

    A controller will choose one conversational skill for each turn. Follow that
    selected skill closely while still sounding natural and human.

    Global rules:
    - Keep replies warm, brief, and conversational (1-3 sentences).
    - Ask at most one question.
    - Do not mention the router, skill names, or any internal policy logic.
    - Do not give advice before trust is built.
    - In crisis scenarios, keep the tone calm and supportive rather than alarmist.

    Reply with ONLY the next message to the seeker.
    """
).strip()


@dataclass
class AgenticLLMEpisodeSummary:
    task_id: str
    model: str
    endpoint_type: str
    steps: int
    score: float
    success: bool
    completion: float
    avg_step_reward: float
    avg_immediate: float
    avg_future_oriented: float
    avg_penalties: float
    final_resolution: float
    had_safety_reference: bool
    skill_counts: Dict[str, int] = field(default_factory=dict)
    skill_trace: List[str] = field(default_factory=list)
    transcript: List[str] = field(default_factory=list)


def classify_endpoint(api_base_url: str) -> str:
    lowered = api_base_url.lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return "local"
    if "huggingface" in lowered:
        return "huggingface"
    return "remote"


def build_user_prompt(
    observation: Observation,
    history: List[str],
    skill_name: str,
    skill_instruction: str,
    rationale: str,
) -> str:
    history_block = "\n".join(history[-8:]) if history else "(first turn)"
    return textwrap.dedent(
        f"""
        Selected skill: {skill_name}
        Why this skill was selected: {rationale}
        Skill directive: {skill_instruction}

        Scenario: {observation.scenario_brief}
        Public stage hint: {observation.stage_hint}
        Turn: {observation.turn}
        Remaining turns: {observation.remaining_turns}

        Recent exchange:
        {history_block}

        Seeker just said:
        "{observation.seeker_utterance}"

        Write the next reply now.
        """
    ).strip()


def call_llm(client: OpenAI, model_name: str, user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        return "That sounds really heavy, and I'm here with you. What feels most important to say right now?"
    return text


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}\n"
            f"Set it, then rerun `py -3 benchmark_agentic_llm.py`."
        )
    return value


def resolve_api_key() -> str:
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing HF_TOKEN, OPENAI_API_KEY, or API_KEY.")
    return api_key


async def run_task(
    openai_client: OpenAI,
    env_client: ESCHttpClient,
    model_name: str,
    endpoint_type: str,
    task_id: str,
) -> AgenticLLMEpisodeSummary:
    router = SkillRouter()
    skills = build_default_skills()
    memory = AgentMemory()
    memory.reset(task_id)

    reset = await env_client.reset(task_id=task_id)
    obs = reset.observation
    history: List[str] = [f"Seeker: {obs.seeker_utterance}"]
    rewards: List[float] = []
    immediate_scores: List[float] = []
    future_scores: List[float] = []
    penalties: List[float] = []
    transcript: List[str] = [f"Seeker: {obs.seeker_utterance}"]
    skill_trace: List[str] = []
    final: Dict[str, Any] = {}
    last_result = None

    while True:
        memory.observe(obs)
        decision = router.choose(obs, memory)
        skill = skills[decision.skill_name]
        prompt = build_user_prompt(
            observation=obs,
            history=history,
            skill_name=decision.skill_name,
            skill_instruction=skill.llm_instruction(obs, memory, decision),
            rationale=decision.rationale,
        )
        message = call_llm(openai_client, model_name, prompt)
        memory.remember(decision.skill_name, message)
        skill_trace.append(
            f"Turn {obs.turn + 1} [{obs.stage_hint}] -> {decision.skill_name}: {decision.rationale}"
        )

        result = await env_client.step(Action(message=message))
        last_result = result

        rewards.append(float(result.reward))
        reward_detail = result.reward_detail or {}
        immediate_scores.append(float(reward_detail.get("immediate", 0.0)))
        future_scores.append(float(reward_detail.get("future_oriented", 0.0)))
        penalties.append(float(reward_detail.get("penalties", 0.0)))

        transcript.append(f"Agent: {message}")
        transcript.append(f"Seeker: {result.observation.seeker_utterance}")
        history.extend(transcript[-2:])

        obs = result.observation
        if result.done:
            final = result.info.get("final", {})
            break

    assert last_result is not None
    return AgenticLLMEpisodeSummary(
        task_id=task_id,
        model=model_name,
        endpoint_type=endpoint_type,
        steps=obs.turn,
        score=float(final.get("score", 0.0)),
        success=bool(final.get("success", 0.0) >= 1.0),
        completion=float(final.get("completion", 0.0)),
        avg_step_reward=mean(rewards) if rewards else 0.0,
        avg_immediate=mean(immediate_scores) if immediate_scores else 0.0,
        avg_future_oriented=mean(future_scores) if future_scores else 0.0,
        avg_penalties=mean(penalties) if penalties else 0.0,
        final_resolution=float(final.get("final_resolution", 0.0)),
        had_safety_reference=bool(last_result.info.get("had_safety_reference", False)),
        skill_counts=dict(memory.skill_counts),
        skill_trace=skill_trace,
        transcript=transcript,
    )


def render_markdown(
    episodes: List[AgenticLLMEpisodeSummary],
    generated_at: str,
    env_url: str,
) -> str:
    avg_score = mean(ep.score for ep in episodes) if episodes else 0.0
    avg_success = mean(1.0 if ep.success else 0.0 for ep in episodes) if episodes else 0.0
    model_name = episodes[0].model if episodes else "unknown"
    endpoint_type = episodes[0].endpoint_type if episodes else "unknown"

    skill_totals: Dict[str, int] = {}
    for episode in episodes:
        for skill_name, count in episode.skill_counts.items():
            skill_totals[skill_name] = skill_totals.get(skill_name, 0) + count

    lines: List[str] = []
    lines.append("# Agentic LLM Benchmark Results")
    lines.append("")
    lines.append(f"_Generated: {generated_at}_")
    lines.append("")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Endpoint type: `{endpoint_type}`")
    lines.append(f"- Environment URL: `{env_url}`")
    lines.append(f"- Average score: `{avg_score:.3f}`")
    lines.append(f"- Success rate: `{avg_success:.2f}`")
    lines.append("")
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append("| Task | Score | Success | Completion | Steps | Avg reward | Final resolution | Safety ref |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for episode in episodes:
        lines.append(
            "| "
            f"{episode.task_id} | "
            f"{episode.score:.3f} | "
            f"{int(episode.success)} | "
            f"{episode.completion:.1f} | "
            f"{episode.steps} | "
            f"{episode.avg_step_reward:.3f} | "
            f"{episode.final_resolution:.3f} | "
            f"{int(episode.had_safety_reference)} |"
        )
    lines.append("")
    lines.append("## Skill Usage Totals")
    lines.append("")
    lines.append("| Skill | Total turns |")
    lines.append("| --- | ---: |")
    for skill_name, count in sorted(skill_totals.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {skill_name} | {count} |")
    lines.append("")
    lines.append("## Skill Trace Excerpts")
    lines.append("")
    for episode in episodes:
        lines.append(f"### {episode.task_id}")
        lines.append("")
        for line in episode.skill_trace[:8]:
            lines.append(f"- {line}")
        lines.append("")
    lines.append("## Transcript Excerpts")
    lines.append("")
    for episode in episodes:
        lines.append(f"### {episode.task_id}")
        lines.append("")
        for line in episode.transcript[:10]:
            lines.append(f"- {line}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


async def async_main(output: str, json_output: str) -> None:
    api_base_url = require_env("API_BASE_URL")
    model_name = require_env("MODEL_NAME")
    api_key = resolve_api_key()
    env_url = require_env("ESC_ENV_URL")

    endpoint_type = classify_endpoint(api_base_url)
    openai_client = OpenAI(base_url=api_base_url, api_key=api_key)
    env_client = ESCHttpClient.from_url(env_url)

    try:
        episodes = [
            await run_task(
                openai_client,
                env_client,
                model_name=model_name,
                endpoint_type=endpoint_type,
                task_id=task_id,
            )
            for task_id in TASK_IDS
        ]
    finally:
        await env_client.close()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    markdown = render_markdown(episodes, generated_at=generated_at, env_url=env_url)

    md_path = Path(output)
    json_path = Path(json_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps([asdict(ep) for ep in episodes], indent=2), encoding="utf-8")

    print(f"Wrote Markdown report to {md_path}")
    print(f"Wrote JSON report to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the skill-routed LLM benchmark.")
    parser.add_argument(
        "--output",
        default="results/agentic_llm_benchmark.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--json-output",
        default="results/agentic_llm_benchmark.json",
        help="JSON output path.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args.output, args.json_output))


if __name__ == "__main__":
    main()
