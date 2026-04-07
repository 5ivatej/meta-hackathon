"""Run local benchmarks for the explicit skill-routed agentic policies.

This script keeps the environment unchanged and benchmarks policy-side agentic
extensions on top of it. The main goal is to show that a small skill router can
compose reusable conversational skills while still solving the benchmark.

Example:
    py -3 benchmark_agentic.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.agentic import SkillRoutedDeterministicPolicy
from src.baselines import StageAwareHeuristicBaseline
from src.env import ESCEnv
from src.models import Action
from src.tasks import TASKS


@dataclass
class AgenticEpisodeSummary:
    policy: str
    task_id: str
    difficulty: str
    steps: int
    score: float
    success: bool
    completion: float
    avg_step_reward: float
    final_resolution: float
    had_safety_reference: bool
    skill_counts: Dict[str, int] = field(default_factory=dict)
    skill_trace: List[str] = field(default_factory=list)
    transcript_excerpt: List[str] = field(default_factory=list)


def _extract_skill_trace(policy: Any) -> tuple[Dict[str, int], List[str]]:
    if not hasattr(policy, "decision_log") or not hasattr(policy, "memory"):
        return {}, []

    counts = dict(getattr(policy.memory, "skill_counts", {}))
    trace: List[str] = []
    for entry in getattr(policy, "decision_log", [])[:6]:
        turn = int(entry.get("turn", "0")) + 1
        trace.append(
            f"Turn {turn} [{entry.get('stage', '')}] -> {entry.get('skill', '')}: {entry.get('reason', '')}"
        )
    return counts, trace


def run_episode(env: ESCEnv, policy: Any, task_id: str) -> AgenticEpisodeSummary:
    task = TASKS[task_id]
    policy.reset(task_id)
    reset = env.reset(task_id=task_id)
    obs = reset.observation

    rewards: List[float] = []
    transcript_excerpt: List[str] = [f"Seeker: {obs.seeker_utterance}"]
    last_result = None

    while True:
        message = policy.act(obs)
        transcript_excerpt.append(f"Agent: {message}")
        result = env.step(Action(message=message))
        last_result = result

        rewards.append(float(result.reward))
        obs = result.observation
        transcript_excerpt.append(f"Seeker: {obs.seeker_utterance}")

        if result.done:
            break

    assert last_result is not None
    final = last_result.info.get("final", {})
    skill_counts, skill_trace = _extract_skill_trace(policy)

    return AgenticEpisodeSummary(
        policy=policy.name,
        task_id=task_id,
        difficulty=task.difficulty,
        steps=obs.turn,
        score=float(final.get("score", 0.0)),
        success=bool(final.get("success", 0.0) >= 1.0),
        completion=float(final.get("completion", 0.0)),
        avg_step_reward=mean(rewards) if rewards else 0.0,
        final_resolution=float(final.get("final_resolution", 0.0)),
        had_safety_reference=bool(last_result.info.get("had_safety_reference", False)),
        skill_counts=skill_counts,
        skill_trace=skill_trace,
        transcript_excerpt=transcript_excerpt[:10],
    )


def summarize_by_policy(episodes: List[AgenticEpisodeSummary]) -> List[Dict[str, Any]]:
    by_name: Dict[str, List[AgenticEpisodeSummary]] = {}
    for episode in episodes:
        by_name.setdefault(episode.policy, []).append(episode)

    rows: List[Dict[str, Any]] = []
    for policy, group in by_name.items():
        rows.append(
            {
                "policy": policy,
                "avg_score": mean(ep.score for ep in group),
                "success_rate": mean(1.0 if ep.success else 0.0 for ep in group),
                "avg_steps": mean(ep.steps for ep in group),
                "avg_completion": mean(ep.completion for ep in group),
                "avg_resolution": mean(ep.final_resolution for ep in group),
            }
        )
    rows.sort(key=lambda row: row["avg_score"], reverse=True)
    return rows


def aggregate_skill_counts(episodes: List[AgenticEpisodeSummary], policy_name: str) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for episode in episodes:
        if episode.policy != policy_name:
            continue
        for skill_name, count in episode.skill_counts.items():
            totals[skill_name] = totals.get(skill_name, 0) + count
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))


def render_markdown(episodes: List[AgenticEpisodeSummary], generated_at: str) -> str:
    summary_rows = summarize_by_policy(episodes)
    skill_totals = aggregate_skill_counts(episodes, "skill_routed_deterministic")
    reference_row = next((row for row in summary_rows if row["policy"] == "stage_aware_heuristic"), None)
    agentic_row = next((row for row in summary_rows if row["policy"] == "skill_routed_deterministic"), None)

    lines: List[str] = []
    lines.append("# Agentic Benchmark Results")
    lines.append("")
    lines.append(f"_Generated: {generated_at}_")
    lines.append("")
    lines.append("This report isolates the policy-side skills/agents story. The environment and rubric are unchanged.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Policy | Avg score | Success rate | Avg steps | Completion | Final resolution |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['policy']} | "
            f"{row['avg_score']:.3f} | "
            f"{row['success_rate']:.2f} | "
            f"{row['avg_steps']:.2f} | "
            f"{row['avg_completion']:.2f} | "
            f"{row['avg_resolution']:.3f} |"
        )
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    if reference_row is not None and agentic_row is not None:
        delta = agentic_row["avg_score"] - reference_row["avg_score"]
        lines.append(
            f"- The explicit skill-routed policy scored `{agentic_row['avg_score']:.3f}`, "
            f"for a delta of `{delta:+.3f}` versus the non-agentic staged heuristic."
        )
    lines.append("- The skill-routed policy keeps the benchmark deterministic while making the policy decomposition visible to judges.")
    lines.append("- Safety escalation remains a policy-side decision; the hard task still requires the environment-level safety-aware finish.")
    lines.append("")
    lines.append("## Skill Usage Totals")
    lines.append("")
    lines.append("| Skill | Total turns |")
    lines.append("| --- | ---: |")
    for skill_name, count in skill_totals.items():
        lines.append(f"| {skill_name} | {count} |")
    lines.append("")
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append("| Task | Difficulty | Policy | Score | Success | Completion | Steps | Safety ref |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for episode in sorted(episodes, key=lambda item: (item.task_id, item.policy)):
        lines.append(
            "| "
            f"{episode.task_id} | "
            f"{episode.difficulty} | "
            f"{episode.policy} | "
            f"{episode.score:.3f} | "
            f"{int(episode.success)} | "
            f"{episode.completion:.1f} | "
            f"{episode.steps} | "
            f"{int(episode.had_safety_reference)} |"
        )
    lines.append("")
    lines.append("## Skill Trace Excerpts")
    lines.append("")
    for episode in episodes:
        if not episode.skill_trace:
            continue
        lines.append(f"### {episode.task_id} - {episode.policy}")
        lines.append("")
        lines.append(
            f"- Score: `{episode.score:.3f}` | Success: `{episode.success}` | "
            f"Completion: `{episode.completion:.1f}`"
        )
        for trace in episode.skill_trace:
            lines.append(f"- {trace}")
        lines.append("")
    lines.append("## Transcript Excerpts")
    lines.append("")
    for episode in episodes:
        lines.append(f"### {episode.task_id} - {episode.policy}")
        lines.append("")
        for line in episode.transcript_excerpt:
            lines.append(f"- {line}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic local benchmark suite.")
    parser.add_argument(
        "--output",
        default="results/agentic_benchmarks.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--json-output",
        default="results/agentic_benchmarks.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    env = ESCEnv()
    policies = [
        StageAwareHeuristicBaseline(),
        SkillRoutedDeterministicPolicy(),
    ]

    episodes: List[AgenticEpisodeSummary] = []
    for policy in policies:
        for task_id in TASKS:
            episodes.append(run_episode(env, policy, task_id))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    markdown = render_markdown(episodes, generated_at=generated_at)

    md_path = Path(args.output)
    json_path = Path(args.json_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps([asdict(ep) for ep in episodes], indent=2), encoding="utf-8")

    print(f"Wrote Markdown report to {md_path}")
    print(f"Wrote JSON report to {json_path}")


if __name__ == "__main__":
    main()
