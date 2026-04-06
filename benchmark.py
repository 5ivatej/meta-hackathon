"""Run deterministic local benchmarks for the ESC submission.

This script benchmarks the in-process environment against a small baseline
ladder and writes:

- a Markdown report for the README / submission
- a JSON artifact for later comparisons

Example:
    py -3 benchmark.py
    py -3 benchmark.py --output results/local_benchmarks.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.baselines import BaselinePolicy, make_default_baselines
from src.env import ESCEnv
from src.models import Action
from src.tasks import TASKS


@dataclass
class EpisodeSummary:
    baseline: str
    task_id: str
    difficulty: str
    steps: int
    score: float
    success: bool
    completion: float
    avg_step_reward: float
    avg_immediate: float
    avg_future_oriented: float
    avg_penalties: float
    final_resolution: float
    efficiency: float
    final_stage: str
    had_safety_reference: bool
    transcript_excerpt: List[str]


def run_episode(env: ESCEnv, baseline: BaselinePolicy, task_id: str) -> EpisodeSummary:
    task = TASKS[task_id]
    baseline.reset(task_id)
    reset = env.reset(task_id=task_id)
    obs = reset.observation

    rewards: List[float] = []
    immediate_scores: List[float] = []
    future_scores: List[float] = []
    penalties: List[float] = []
    transcript_excerpt: List[str] = [f"Seeker: {obs.seeker_utterance}"]
    last_result = None

    while True:
        message = baseline.act(obs)
        transcript_excerpt.append(f"Agent: {message}")
        result = env.step(Action(message=message))
        last_result = result

        rewards.append(float(result.reward))
        immediate_scores.append(float(result.reward_detail.immediate))
        future_scores.append(float(result.reward_detail.future_oriented))
        penalties.append(float(result.reward_detail.penalties))

        obs = result.observation
        transcript_excerpt.append(f"Seeker: {obs.seeker_utterance}")

        if result.done:
            break

    assert last_result is not None
    final = last_result.info.get("final", {})

    return EpisodeSummary(
        baseline=baseline.name,
        task_id=task_id,
        difficulty=task.difficulty,
        steps=obs.turn,
        score=float(final.get("score", 0.0)),
        success=bool(final.get("success", 0.0) >= 1.0),
        completion=float(final.get("completion", 0.0)),
        avg_step_reward=mean(rewards) if rewards else 0.0,
        avg_immediate=mean(immediate_scores) if immediate_scores else 0.0,
        avg_future_oriented=mean(future_scores) if future_scores else 0.0,
        avg_penalties=mean(penalties) if penalties else 0.0,
        final_resolution=float(final.get("final_resolution", 0.0)),
        efficiency=float(final.get("efficiency", 0.0)),
        final_stage=str(last_result.info.get("stage", "")),
        had_safety_reference=bool(last_result.info.get("had_safety_reference", False)),
        transcript_excerpt=transcript_excerpt[:8],
    )


def summarize_by_baseline(episodes: List[EpisodeSummary]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_name: Dict[str, List[EpisodeSummary]] = {}
    for ep in episodes:
        by_name.setdefault(ep.baseline, []).append(ep)

    for baseline, eps in by_name.items():
        rows.append(
            {
                "baseline": baseline,
                "avg_score": mean(ep.score for ep in eps),
                "success_rate": mean(1.0 if ep.success else 0.0 for ep in eps),
                "avg_steps": mean(ep.steps for ep in eps),
                "avg_completion": mean(ep.completion for ep in eps),
                "avg_immediate": mean(ep.avg_immediate for ep in eps),
                "avg_future_oriented": mean(ep.avg_future_oriented for ep in eps),
                "avg_penalties": mean(ep.avg_penalties for ep in eps),
            }
        )
    rows.sort(key=lambda row: row["avg_score"], reverse=True)
    return rows


def render_markdown(episodes: List[EpisodeSummary], generated_at: str) -> str:
    baseline_rows = summarize_by_baseline(episodes)

    hardest_task = min(
        TASKS,
        key=lambda task_id: mean(ep.score for ep in episodes if ep.task_id == task_id),
    )
    best_baseline = baseline_rows[0]["baseline"]
    generic_row = next((row for row in baseline_rows if row["baseline"] == "generic_template"), None)

    lines: List[str] = []
    lines.append("# Local Benchmark Results")
    lines.append("")
    lines.append(f"_Generated: {generated_at}_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Baseline | Avg score | Success rate | Avg steps | Completion | Avg immediate | Avg future | Avg penalties |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in baseline_rows:
        lines.append(
            "| "
            f"{row['baseline']} | "
            f"{row['avg_score']:.3f} | "
            f"{row['success_rate']:.2f} | "
            f"{row['avg_steps']:.2f} | "
            f"{row['avg_completion']:.2f} | "
            f"{row['avg_immediate']:.3f} | "
            f"{row['avg_future_oriented']:.3f} | "
            f"{row['avg_penalties']:.3f} |"
        )
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    lines.append(f"- Best deterministic baseline: `{best_baseline}`.")
    lines.append(f"- Hardest task under current local baselines: `{hardest_task}`.")
    if generic_row is not None and generic_row["avg_completion"] == 0.0:
        lines.append("- The generic empathetic template no longer completes tasks successfully, which is exactly what we want from the hardened rubric.")
    lines.append("- `avg_immediate` vs `avg_future` provides a lightweight rubric ablation lens: weak baselines can sound safe locally but still fail completion and final score.")
    lines.append("")
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append("| Task | Difficulty | Baseline | Score | Success | Completion | Steps | Final resolution | Final stage | Safety ref |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for ep in sorted(episodes, key=lambda item: (item.task_id, item.score), reverse=False):
        lines.append(
            "| "
            f"{ep.task_id} | "
            f"{ep.difficulty} | "
            f"{ep.baseline} | "
            f"{ep.score:.3f} | "
            f"{int(ep.success)} | "
            f"{ep.completion:.1f} | "
            f"{ep.steps} | "
            f"{ep.final_resolution:.3f} | "
            f"{ep.final_stage} | "
            f"{int(ep.had_safety_reference)} |"
        )
    lines.append("")
    lines.append("## Transcript Excerpts")
    lines.append("")
    for ep in episodes:
        lines.append(f"### {ep.task_id} · {ep.baseline}")
        lines.append("")
        lines.append(
            f"- Score: `{ep.score:.3f}` | Success: `{ep.success}` | "
            f"Completion: `{ep.completion:.1f}` | Steps: `{ep.steps}`"
        )
        for turn in ep.transcript_excerpt:
            lines.append(f"- {turn}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic local benchmarks.")
    parser.add_argument(
        "--output",
        default="results/local_benchmarks.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--json-output",
        default="results/local_benchmarks.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    env = ESCEnv()
    baselines = make_default_baselines()

    episodes: List[EpisodeSummary] = []
    for baseline in baselines:
        for task_id in TASKS:
            episodes.append(run_episode(env, baseline, task_id))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    markdown = render_markdown(episodes, generated_at=generated_at)

    md_path = Path(args.output)
    json_path = Path(args.json_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps([asdict(ep) for ep in episodes], indent=2),
        encoding="utf-8",
    )

    print(f"Wrote Markdown report to {md_path}")
    print(f"Wrote JSON report to {json_path}")


if __name__ == "__main__":
    main()
