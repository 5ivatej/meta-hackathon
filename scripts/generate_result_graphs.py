from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "docs" / "assets"

BG = "#fcfbf8"
PANEL = "#ffffff"
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#d6d3d1"
TEAL = "#1f6f78"
ORANGE = "#d97706"
RED = "#b42318"
GREEN = "#15803d"
SLATE = "#475569"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}"/>',
    ]


def _text(x: float, y: float, value: str, size: int = 14, weight: str = "400", fill: str = INK, anchor: str = "start") -> str:
    safe = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{safe}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", radius: float = 12) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def _polyline(points: Iterable[tuple[float, float]], stroke: str, width: float = 3.0, fill: str = "none") -> str:
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{data}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'


def build_grpo_reward_curve() -> None:
    data = _load_json(RESULTS_DIR / "trainer_state.json")
    logs = [row for row in data.get("log_history", []) if "step" in row and "reward" in row]
    width, height = 960, 540
    left, right, top, bottom = 90, 40, 80, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    steps = [row["step"] for row in logs]
    rewards = [row["reward"] for row in logs]
    stds = [row["reward_std"] for row in logs]
    losses = [row["loss"] for row in logs]
    y_max = max(max(r + s for r, s in zip(rewards, stds)), 0.3)
    y_max = round(y_max + 0.05, 2)

    def sx(step: float) -> float:
        return left + (step - min(steps)) / (max(steps) - min(steps)) * chart_w

    def sy(value: float) -> float:
        return top + (1 - value / y_max) * chart_h

    svg = _svg_header(width, height)
    svg.append(_text(40, 42, "GRPO Future-Reward Curve", 28, "700"))
    svg.append(_text(40, 66, "Logged mean reward with one-standard-deviation band across the 20-step reduced run.", 15, "400", MUTED))
    svg.append(_rect(left - 20, top - 24, chart_w + 40, chart_h + 44, PANEL, "#ece7df", 18))

    for tick in range(0, int(y_max * 100) + 1, 10):
        value = tick / 100
        y = sy(value)
        svg.append(_line(left, y, left + chart_w, y, GRID, 1, "4 6"))
        svg.append(_text(left - 14, y + 5, f"{value:.1f}", 12, "400", MUTED, "end"))

    for step in steps:
        x = sx(step)
        svg.append(_line(x, top, x, top + chart_h, GRID, 1, "4 6"))
        svg.append(_text(x, top + chart_h + 26, str(step), 12, "400", MUTED, "middle"))

    svg.append(_text(left + chart_w / 2, height - 24, "GRPO step", 14, "600", MUTED, "middle"))
    svg.append(_text(28, top + chart_h / 2, "Reward", 14, "600", MUTED))

    band_points_top = [(sx(step), sy(reward + std)) for step, reward, std in zip(steps, rewards, stds)]
    band_points_bottom = [(sx(step), sy(max(0.0, reward - std))) for step, reward, std in reversed(list(zip(steps, rewards, stds)))]
    band = " ".join(f"{x:.2f},{y:.2f}" for x, y in band_points_top + band_points_bottom)
    svg.append(f'<polygon points="{band}" fill="{TEAL}" opacity="0.15"/>')
    svg.append(_polyline([(sx(step), sy(reward)) for step, reward in zip(steps, rewards)], TEAL, 4))

    for step, reward in zip(steps, rewards):
        svg.append(f'<circle cx="{sx(step):.2f}" cy="{sy(reward):.2f}" r="5" fill="{TEAL}"/>')

    peak_idx = max(range(len(rewards)), key=lambda i: rewards[i])
    peak_x, peak_y = sx(steps[peak_idx]), sy(rewards[peak_idx])
    svg.append(_line(peak_x, peak_y - 8, peak_x + 96, peak_y - 58, ORANGE, 2))
    svg.append(_text(peak_x + 104, peak_y - 62, f"Peak: {rewards[peak_idx]:.3f}", 14, "700", ORANGE))
    svg.append(_text(peak_x + 104, peak_y - 42, f"step {steps[peak_idx]}", 13, "400", MUTED))

    legend_x = width - 260
    legend_y = 98
    svg.append(_rect(legend_x, legend_y, 200, 92, "#f8fafc", "#e5e7eb", 14))
    svg.append(_line(legend_x + 16, legend_y + 24, legend_x + 52, legend_y + 24, TEAL, 4))
    svg.append(_text(legend_x + 62, legend_y + 29, "mean reward", 13, "600"))
    svg.append(f'<rect x="{legend_x + 16}" y="{legend_y + 42}" width="36" height="12" fill="{TEAL}" opacity="0.15"/>')
    svg.append(_text(legend_x + 62, legend_y + 53, "± 1 std", 13, "600"))
    svg.append(_text(legend_x + 16, legend_y + 76, f"final loss: {losses[-1]:.4f}", 13, "400", MUTED))

    (ASSETS_DIR / "grpo_reward_curve.svg").write_text("\n".join(svg + ["</svg>"]))


def build_task_summary_chart() -> None:
    rows = _load_jsonl(RESULTS_DIR / "candidate_rewards.jsonl")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)

    tasks = list(grouped)
    task_labels = {
        "work_stress_venting": "work stress",
        "guarded_relationship": "guarded relationship",
        "crisis_fragile_trust": "crisis fragile trust",
    }
    mean_rewards = [mean(item["scalar_reward"] for item in grouped[task]) for task in tasks]
    completion_rates = [sum(1 for item in grouped[task] if item.get("completed")) / len(grouped[task]) for task in tasks]
    counts = [len(grouped[task]) for task in tasks]

    width, height = 1080, 560
    left, right, top, bottom = 90, 50, 92, 96
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_group = chart_w / len(tasks)
    bar_w = 84

    svg = _svg_header(width, height)
    svg.append(_text(40, 42, "Simulation Reward Summary by Task", 28, "700"))
    svg.append(_text(40, 66, "Mean scalar reward and completion rate across 24 candidate-response evaluations.", 15, "400", MUTED))
    svg.append(_rect(left - 20, top - 24, chart_w + 40, chart_h + 44, PANEL, "#ece7df", 18))

    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1 - value) * chart_h
        svg.append(_line(left, y, left + chart_w, y, GRID, 1, "4 6"))
        svg.append(_text(left - 14, y + 5, f"{value:.2f}", 12, "400", MUTED, "end"))

    for i, task in enumerate(tasks):
        center = left + bar_group * (i + 0.5)
        reward_h = mean_rewards[i] * chart_h
        comp_h = completion_rates[i] * chart_h
        reward_x = center - bar_w - 10
        comp_x = center + 10
        reward_y = top + chart_h - reward_h
        comp_y = top + chart_h - comp_h

        svg.append(_rect(reward_x, reward_y, bar_w, reward_h, TEAL, "none", 10))
        svg.append(_rect(comp_x, comp_y, bar_w, comp_h, ORANGE, "none", 10))
        svg.append(_text(reward_x + bar_w / 2, reward_y - 10, f"{mean_rewards[i]:.3f}", 12, "700", TEAL, "middle"))
        svg.append(_text(comp_x + bar_w / 2, comp_y - 10, f"{completion_rates[i]*100:.0f}%", 12, "700", ORANGE, "middle"))
        svg.append(_text(center, top + chart_h + 28, task_labels.get(task, task.replace("_", " ")), 12, "600", INK, "middle"))
        svg.append(_text(center, top + chart_h + 62, f"n={counts[i]}", 12, "400", MUTED, "middle"))

    legend_x = width - 270
    legend_y = 104
    svg.append(_rect(legend_x, legend_y, 210, 84, "#f8fafc", "#e5e7eb", 14))
    svg.append(_rect(legend_x + 16, legend_y + 18, 24, 12, TEAL, "none", 4))
    svg.append(_text(legend_x + 52, legend_y + 29, "mean scalar reward", 13, "600"))
    svg.append(_rect(legend_x + 16, legend_y + 46, 24, 12, ORANGE, "none", 4))
    svg.append(_text(legend_x + 52, legend_y + 57, "completion rate", 13, "600"))

    (ASSETS_DIR / "simulation_task_summary.svg").write_text("\n".join(svg + ["</svg>"]))


def build_reward_model_audit_chart() -> None:
    summary = _load_json(RESULTS_DIR / "reward_model_audit_summary.json")
    examples = summary.get("top_underestimates", [])[:3]
    labels = [f"ex {i + 1}" for i in range(len(examples))]
    targets = [float(item["target_reward"]) for item in examples]
    preds = [float(item["predicted_reward"]) for item in examples]
    errors = [float(item["absolute_error"]) for item in examples]

    width, height = 980, 560
    left, right, top, bottom = 90, 40, 96, 96
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_group = chart_w / max(1, len(labels))
    bar_w = 90

    svg = _svg_header(width, height)
    svg.append(_text(40, 42, "Reward Model Audit Snapshot", 28, "700"))
    svg.append(_text(40, 66, "Target versus predicted reward on the reduced audit set.", 15, "400", MUTED))
    svg.append(_rect(left - 20, top - 24, chart_w + 40, chart_h + 44, PANEL, "#ece7df", 18))

    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1 - value) * chart_h
        svg.append(_line(left, y, left + chart_w, y, GRID, 1, "4 6"))
        svg.append(_text(left - 14, y + 5, f"{value:.2f}", 12, "400", MUTED, "end"))

    for i, label in enumerate(labels):
        center = left + bar_group * (i + 0.5)
        tx = center - bar_w - 10
        px = center + 10
        target_h = targets[i] * chart_h
        pred_h = preds[i] * chart_h
        ty = top + chart_h - target_h
        py = top + chart_h - pred_h
        svg.append(_rect(tx, ty, bar_w, target_h, GREEN, "none", 10))
        svg.append(_rect(px, py, bar_w, max(pred_h, 4), RED, "none", 10))
        svg.append(_text(tx + bar_w / 2, ty - 10, f"{targets[i]:.2f}", 12, "700", GREEN, "middle"))
        svg.append(_text(px + bar_w / 2, py - 10, f"{preds[i]:.2f}", 12, "700", RED, "middle"))
        svg.append(_text(center, top + chart_h + 28, label, 13, "600", INK, "middle"))
        svg.append(_text(center, top + chart_h + 50, f"err={errors[i]:.2f}", 12, "400", MUTED, "middle"))

    metrics_x = width - 250
    metrics_y = 112
    svg.append(_rect(metrics_x, metrics_y, 190, 110, "#f8fafc", "#e5e7eb", 14))
    svg.append(_text(metrics_x + 16, metrics_y + 28, f"examples: {summary['num_examples']}", 13, "600"))
    svg.append(_text(metrics_x + 16, metrics_y + 54, f"MSE: {summary['mse']:.3f}", 13, "600"))
    svg.append(_text(metrics_x + 16, metrics_y + 80, f"MAE: {summary['mae']:.3f}", 13, "600"))
    svg.append(_text(metrics_x + 16, metrics_y + 100, "Predictions collapsed to 0.0", 12, "400", MUTED))

    (ASSETS_DIR / "reward_model_audit.svg").write_text("\n".join(svg + ["</svg>"]))


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    build_grpo_reward_curve()
    build_task_summary_chart()
    build_reward_model_audit_chart()
    print("Generated:")
    for name in [
        "docs/assets/grpo_reward_curve.svg",
        "docs/assets/simulation_task_summary.svg",
        "docs/assets/reward_model_audit.svg",
    ]:
        print(name)


if __name__ == "__main__":
    main()
