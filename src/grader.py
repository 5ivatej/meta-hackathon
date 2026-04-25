"""Reward and grading logic for the ESC environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .seeker import (
    Features,
    SeekerState,
    Stage,
    resolution_score,
    simulate_oracle_rollout,
    stage_progress,
)

LOOKAHEAD_K = 3
W_IMMEDIATE = 0.40
W_FUTURE = 0.45
W_LONG_HORIZON = 0.15
DISMISSIVE_PENALTY = 0.6
PREMATURE_ADVICE_PENALTY = 0.25
BARE_PENALTY = 0.15
INTERROGATION_PENALTY = 0.15
REPETITION_PENALTY = 0.18


@dataclass
class GradeBreakdown:
    value: float
    immediate: float
    future_oriented: float
    penalties: float
    components: Dict[str, float]


def _stage_fit_score(stage: Stage, f: Features) -> float:
    if stage in (Stage.OPENING, Stage.EXPLORING):
        fit = 0.5 * min(1.0, f.empathy) + 0.3 * min(1.0, f.open_question) + 0.2 * min(1.0, f.validation)
        fit -= 0.4 * min(1.0, f.advice)
    elif stage == Stage.REFLECTING:
        fit = 0.5 * min(1.0, f.validation) + 0.4 * min(1.0, f.empathy) + 0.1 * min(1.0, f.open_question)
        fit -= 0.2 * min(1.0, f.advice)
    elif stage == Stage.PLANNING:
        fit = 0.4 * min(1.0, f.open_question) + 0.3 * min(1.0, f.advice) + 0.3 * min(1.0, f.empathy)
    else:
        fit = 0.5 * min(1.0, f.empathy) + 0.3 * min(1.0, f.safety) + 0.2 * min(1.0, f.validation)
    return max(0.0, min(1.0, fit))


def _immediate_reward(pre_state: SeekerState, post_state: SeekerState, f: Features) -> float:
    stage_fit = _stage_fit_score(pre_state.stage, f)
    trust_delta = max(0.0, post_state.trust - pre_state.trust)
    distress_relief = max(0.0, pre_state.distress - post_state.distress)
    stage_advance = max(0.0, stage_progress(post_state.stage) - stage_progress(pre_state.stage))
    reveal_bonus = 0.2 if (post_state.revealed and not pre_state.revealed) else 0.0
    return max(
        0.0,
        min(
            1.0,
            0.45 * stage_fit
            + 0.20 * trust_delta * 2.0
            + 0.20 * distress_relief * 2.0
            + 0.10 * stage_advance
            + 0.05
            + reveal_bonus,
        ),
    )


def _future_oriented_reward(pre_state: SeekerState, post_state: SeekerState) -> float:
    pre_ceiling = simulate_oracle_rollout(pre_state.snapshot(), LOOKAHEAD_K)
    post_ceiling = simulate_oracle_rollout(post_state.snapshot(), LOOKAHEAD_K)
    delta = post_ceiling - pre_ceiling
    return max(0.0, min(1.0, 0.5 + 1.25 * delta))


def _penalties(flags: Dict[str, bool]) -> float:
    p = 0.0
    if flags.get("dismissed"):
        p += DISMISSIVE_PENALTY
    if flags.get("advice_too_early"):
        p += PREMATURE_ADVICE_PENALTY
    if flags.get("bare_reply"):
        p += BARE_PENALTY
    if flags.get("interrogated"):
        p += INTERROGATION_PENALTY
    if flags.get("repetitive"):
        p += REPETITION_PENALTY
    return p


def _long_horizon_reward(context: Optional[Dict[str, float]]) -> tuple[float, Dict[str, float]]:
    if not context:
        return 0.0, {}

    continuity = 0.18 if context.get("continuity_hit", 0.0) > 0 else (-0.10 if context.get("continuity_expected", 0.0) > 0 else 0.0)
    goal_follow = 0.10 if context.get("goal_hit", 0.0) > 0 else 0.0
    session_transition = 0.08 * context.get("session_transition_bonus", 0.0)
    resume_bonus = 0.08 * context.get("resume_continuity", 0.0)
    budget_bonus = 0.08 * max(0.0, 1.0 - context.get("budget_ratio", 0.0))
    drift_penalty = 0.14 * context.get("drift", 0.0)
    repetition_penalty = 0.10 * context.get("repetition_window", 0.0)
    runaway_penalty = 0.20 * max(0.0, context.get("budget_ratio", 0.0) - 1.0)
    runaway_penalty += 0.20 * max(0.0, context.get("time_ratio", 0.0) - 1.0)

    total = continuity + goal_follow + session_transition + resume_bonus + budget_bonus
    total -= drift_penalty + repetition_penalty + runaway_penalty
    components = {
        "memory_continuity": continuity,
        "goal_follow": goal_follow,
        "session_transition_bonus": session_transition,
        "resume_continuity": resume_bonus,
        "budget_bonus": budget_bonus,
        "drift_penalty": drift_penalty,
        "repetition_window_penalty": repetition_penalty,
        "runaway_budget_penalty": runaway_penalty,
    }
    return total, components


def grade_step(
    pre_state: SeekerState,
    post_state: SeekerState,
    features: Features,
    flags: Dict[str, bool],
    long_horizon_context: Optional[Dict[str, float]] = None,
) -> GradeBreakdown:
    imm = _immediate_reward(pre_state, post_state, features)
    fut = _future_oriented_reward(pre_state, post_state)
    pen = _penalties(flags)
    long_horizon_value, long_horizon_components = _long_horizon_reward(long_horizon_context)
    combined = W_IMMEDIATE * imm + W_FUTURE * fut + W_LONG_HORIZON * long_horizon_value - pen
    value = max(0.0, min(1.0, combined))
    components = {
        "stage_fit": _stage_fit_score(pre_state.stage, features),
        "trust_delta": post_state.trust - pre_state.trust,
        "distress_delta": pre_state.distress - post_state.distress,
        "resolution_score_post": resolution_score(post_state),
        "pre_oracle_ceiling": simulate_oracle_rollout(pre_state.snapshot(), LOOKAHEAD_K),
        "post_oracle_ceiling": simulate_oracle_rollout(post_state.snapshot(), LOOKAHEAD_K),
        "long_horizon_value": long_horizon_value,
    }
    components.update(long_horizon_components)
    return GradeBreakdown(
        value=value,
        immediate=imm,
        future_oriented=fut,
        penalties=pen,
        components=components,
    )


def final_task_score(
    cumulative_reward: float,
    steps_taken: int,
    max_turns: int,
    final_state: SeekerState,
    success_threshold: float,
    completed: bool,
    alliance_strength: float = 0.0,
    stability: float = 0.0,
    adherence: float = 0.0,
    continuity: float = 0.0,
    budget_ratio: float = 0.0,
) -> Dict[str, float]:
    avg_reward = cumulative_reward / max(1, steps_taken)
    final_res = resolution_score(final_state)
    usage = steps_taken / max_turns
    efficiency = 1.0 if usage <= 0.6 else max(0.65, 1.0 - 0.75 * (usage - 0.6))
    budget_efficiency = max(0.0, min(1.0, 1.0 - max(0.0, budget_ratio - 1.0)))
    completion = 1.0 if completed else 0.0
    score = (
        0.22 * avg_reward
        + 0.28 * final_res
        + 0.15 * efficiency
        + 0.12 * alliance_strength
        + 0.10 * stability
        + 0.05 * adherence
        + 0.04 * continuity
        + 0.04 * budget_efficiency
        + 0.10 * completion
    )
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "avg_reward": avg_reward,
        "final_resolution": final_res,
        "efficiency": efficiency,
        "alliance_strength": alliance_strength,
        "stability": stability,
        "adherence": adherence,
        "continuity": continuity,
        "budget_efficiency": budget_efficiency,
        "completion": completion,
        "success": 1.0 if (completed and score >= success_threshold) else 0.0,
    }
