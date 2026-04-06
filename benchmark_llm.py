"""Run an LLM baseline against the ESC environment and write Markdown results.

This script is intentionally separate from `inference.py`:

- `inference.py` keeps the hackathon-required stdout contract.
- `benchmark_llm.py` is for your own benchmarking workflow and writes
  reusable Markdown + JSON artifacts.

Required environment variables:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN or API_KEY
    ESC_ENV_URL

Example:
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    set HF_TOKEN=...
    set ESC_ENV_URL=http://localhost:7860
    py -3 benchmark_llm.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from openai import OpenAI

from src.client import ESCHttpClient
from src.models import Action

TASK_IDS = ["work_stress_venting", "guarded_relationship", "crisis_fragile_trust"]
TEMPERATURE = 0.6
MAX_TOKENS = 220

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an emotionally attuned peer supporter chatting with someone who is
    going through a hard time. Your job is NOT to fix their problem. Your job
    is to make them feel heard, safe, and understood first — and only move
    toward gentle exploration or light action once trust is established.

    Principles:
    - Lead with empathy and validation. Reflect what you hear.
    - Do NOT give advice until the person has clearly shared what's really
      going on and feels heard.
    - Ask at most one open-ended question per reply. Never interrogate.
    - Never be dismissive, minimising, or instructive in a judgmental tone.
    - Keep replies warm, brief (1-3 sentences), and human.
    - In high-distress / crisis scenarios, gently reference professional
      support (a therapist, crisis line) only after rapport is built.

    Reply with ONLY your next message to the seeker.
    """
).strip()


@dataclass
class LLMEpisodeSummary:
    task_id: str
    model: str
    steps: int
    score: float
    success: bool
    completion: float
    avg_step_reward: float
    avg_immediate: float
    avg_future_oriented: float
    avg_penalties: float
    final_resolution: float
    transcript: List[str]


def build_user_prompt(
    scenario_brief: str,
    stage_hint: str,
    turn: int,
    remaining: int,
    seeker_utterance: str,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-8:]) if history else "(first turn)"
    return textwrap.dedent(
        f"""
        Scenario: {scenario_brief}
        Conversation stage (public hint): {stage_hint}
        Turn: {turn}
        Remaining turns: {remaining}

        Recent exchange:
        {history_block}

        Seeker just said:
        "{seeker_utterance}"

        Write your next reply (1-3 sentences, warm, no advice unless rapport is clearly established):
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
        return "That sounds really hard. I'm here with you. Can you tell me more about what's feeling heaviest right now?"
    return text


async def run_task(
    openai_client: OpenAI,
    env_client: ESCHttpClient,
    model_name: str,
    task_id: str,
) -> LLMEpisodeSummary:
    reset = await env_client.reset(task_id=task_id)
    obs = reset.observation
    history: List[str] = [f"Seeker: {obs.seeker_utterance}"]
    rewards: List[float] = []
    immediate_scores: List[float] = []
    future_scores: List[float] = []
    penalties: List[float] = []
    transcript: List[str] = [f"Seeker: {obs.seeker_utterance}"]
    final: Dict[str, Any] = {}

    while True:
        prompt = build_user_prompt(
            scenario_brief=obs.scenario_brief,
            stage_hint=obs.stage_hint,
            turn=obs.turn,
            remaining=obs.remaining_turns,
            seeker_utterance=obs.seeker_utterance,
            history=history,
        )
        message = call_llm(openai_client, model_name, prompt)
        result = await env_client.step(Action(message=message))

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

    return LLMEpisodeSummary(
        task_id=task_id,
        model=model_name,
        steps=obs.turn,
        score=float(final.get("score", 0.0)),
        success=bool(final.get("success", 0.0) >= 1.0),
        completion=float(final.get("completion", 0.0)),
        avg_step_reward=mean(rewards) if rewards else 0.0,
        avg_immediate=mean(immediate_scores) if immediate_scores else 0.0,
        avg_future_oriented=mean(future_scores) if future_scores else 0.0,
        avg_penalties=mean(penalties) if penalties else 0.0,
        final_resolution=float(final.get("final_resolution", 0.0)),
        transcript=transcript,
    )


def render_markdown(episodes: List[LLMEpisodeSummary], generated_at: str, env_url: str) -> str:
    avg_score = mean(ep.score for ep in episodes) if episodes else 0.0
    avg_success = mean(1.0 if ep.success else 0.0 for ep in episodes) if episodes else 0.0
    model_name = episodes[0].model if episodes else "unknown"

    lines: List[str] = []
    lines.append("# LLM Benchmark Results")
    lines.append("")
    lines.append(f"_Generated: {generated_at}_")
    lines.append("")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Environment URL: `{env_url}`")
    lines.append(f"- Average score: `{avg_score:.3f}`")
    lines.append(f"- Success rate: `{avg_success:.2f}`")
    lines.append("")
    lines.append("| Task | Score | Success | Completion | Steps | Avg step reward | Final resolution |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ep in episodes:
        lines.append(
            "| "
            f"{ep.task_id} | "
            f"{ep.score:.3f} | "
            f"{int(ep.success)} | "
            f"{ep.completion:.1f} | "
            f"{ep.steps} | "
            f"{ep.avg_step_reward:.3f} | "
            f"{ep.final_resolution:.3f} |"
        )
    lines.append("")
    lines.append("## Transcript Excerpts")
    lines.append("")
    for ep in episodes:
        lines.append(f"### {ep.task_id}")
        lines.append("")
        for line in ep.transcript[:10]:
            lines.append(f"- {line}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}\n"
            f"Set it, then rerun `py -3 benchmark_llm.py`."
        )
    return value


async def async_main(output: str, json_output: str) -> None:
    api_base_url = require_env("API_BASE_URL")
    model_name = require_env("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing HF_TOKEN or API_KEY.")
    env_url = require_env("ESC_ENV_URL")

    openai_client = OpenAI(base_url=api_base_url, api_key=api_key)
    env_client = ESCHttpClient.from_url(env_url)

    try:
        episodes = [
            await run_task(openai_client, env_client, model_name=model_name, task_id=task_id)
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
    parser = argparse.ArgumentParser(description="Run the LLM baseline and write results.")
    parser.add_argument("--output", default="results/llm_benchmark.md", help="Markdown output path.")
    parser.add_argument("--json-output", default="results/llm_benchmark.json", help="JSON output path.")
    args = parser.parse_args()
    asyncio.run(async_main(args.output, args.json_output))


if __name__ == "__main__":
    main()
