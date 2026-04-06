"""Reward / grading module.

Implements the hybrid reward used by the ESC environment:

    step_reward = clip(immediate + future_oriented - penalties,  0, 1)

- immediate            : stage-appropriate empathy/validation/open-question signal
- future_oriented      : RLFF-ESC style lookahead — projects the oracle policy
                         k steps forward from the *post-action* state and
                         compares the projected resolution_score against the
                         pre-action ceiling. Rewards actions that *preserve or
                         advance* the attainable resolution, not just ones
                         that look good this turn.
- penalties            : dismissive language, premature advice, repetitive
                         bare replies, interrogation.

This shaping gives the agent dense, varying signal across the trajectory
(required by the rubric: "signal over the full trajectory, not just
binary end-of-episode").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .seeker import (
    Features,
    SeekerState,
    Stage,
    resolution_score,
    simulate_oracle_rollout,
    stage_progress,
)

# Hyper-parameters — tuned to keep step reward in [0, 1] under normal play.
LOOKAHEAD_K = 3
W_IMMEDIATE = 0.45
W_FUTURE = 0.55
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
    """How appropriate are the agent's features for the current stage?"""
    if stage in (Stage.OPENING, Stage.EXPLORING):
        # Reward empathy + open questions; punish early advice strongly.
        fit = 0.5 * min(1.0, f.empathy) + 0.3 * min(1.0, f.open_question) + 0.2 * min(1.0, f.validation)
        fit -= 0.4 * min(1.0, f.advice)
    elif stage == Stage.REFLECTING:
        fit = 0.5 * min(1.0, f.validation) + 0.4 * min(1.0, f.empathy) + 0.1 * min(1.0, f.open_question)
        fit -= 0.2 * min(1.0, f.advice)
    elif stage == Stage.PLANNING:
        # Advice is finally okay here.
        fit = 0.4 * min(1.0, f.open_question) + 0.3 * min(1.0, f.advice) + 0.3 * min(1.0, f.empathy)
    else:  # CLOSING
        fit = 0.5 * min(1.0, f.empathy) + 0.3 * min(1.0, f.safety) + 0.2 * min(1.0, f.validation)
    return max(0.0, min(1.0, fit))


def _immediate_reward(pre_state: SeekerState, post_state: SeekerState, f: Features) -> float:
    """Turn-level reward: stage fit + trust delta + distress delta."""
    stage_fit = _stage_fit_score(pre_state.stage, f)
    trust_delta = max(0.0, post_state.trust - pre_state.trust)
    distress_relief = max(0.0, pre_state.distress - post_state.distress)
    stage_advance = max(
        0.0, stage_progress(post_state.stage) - stage_progress(pre_state.stage)
    )
    reveal_bonus = 0.2 if (post_state.revealed and not pre_state.revealed) else 0.0
    return max(
        0.0,
        min(
            1.0,
            0.45 * stage_fit
            + 0.20 * trust_delta * 2.0  # scale small deltas
            + 0.20 * distress_relief * 2.0
            + 0.10 * stage_advance
            + 0.05 * 1.0  # small baseline for any non-destructive turn
            + reveal_bonus,
        ),
    )


def _future_oriented_reward(pre_state: SeekerState, post_state: SeekerState) -> float:
    """RLFF-ESC style: does this action *preserve / advance* future resolution?

    We roll the oracle policy k steps from both the pre- and post-action states
    and take the (clipped) delta. Positive delta = the action moved the
    attainable future forward; negative = the agent damaged trajectory
    potential and must recover.
    """
    pre_ceiling = simulate_oracle_rollout(pre_state.snapshot(), LOOKAHEAD_K)
    post_ceiling = simulate_oracle_rollout(post_state.snapshot(), LOOKAHEAD_K)
    delta = post_ceiling - pre_ceiling
    # Map delta in roughly [-0.4, +0.4] to [0, 1] with 0 at delta=0.
    return max(0.0, min(1.0, 0.5 + 1.25 * delta))


def _penalties(flags: Dict[str, bool], f: Features) -> float:
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


def grade_step(
    pre_state: SeekerState,
    post_state: SeekerState,
    features: Features,
    flags: Dict[str, bool],
) -> GradeBreakdown:
    imm = _immediate_reward(pre_state, post_state, features)
    fut = _future_oriented_reward(pre_state, post_state)
    pen = _penalties(flags, features)
    combined = W_IMMEDIATE * imm + W_FUTURE * fut - pen
    value = max(0.0, min(1.0, combined))
    components = {
        "stage_fit": _stage_fit_score(pre_state.stage, features),
        "trust_delta": post_state.trust - pre_state.trust,
        "distress_delta": pre_state.distress - post_state.distress,
        "resolution_score_post": resolution_score(post_state),
        "pre_oracle_ceiling": simulate_oracle_rollout(pre_state.snapshot(), LOOKAHEAD_K),
        "post_oracle_ceiling": simulate_oracle_rollout(post_state.snapshot(), LOOKAHEAD_K),
    }
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
) -> Dict[str, float]:
    """Compute the final [0,1] task score used by the grader."""
    # Component 1: average shaped reward over the trajectory (already in [0,1]).
    avg_reward = cumulative_reward / max(1, steps_taken)
    # Component 2: final resolution_score.
    final_res = resolution_score(final_state)
    # Component 3: efficiency — finishing sooner is slightly better, but never
    # negative. Flat 1.0 if used ≤ 60% of budget, linearly decays to 0.7 at max.
    usage = steps_taken / max_turns
    efficiency = 1.0 if usage <= 0.6 else max(0.7, 1.0 - 0.75 * (usage - 0.6))
    completion = 1.0 if completed else 0.0
    score = (
        0.30 * avg_reward
        + 0.45 * final_res
        + 0.10 * efficiency
        + 0.15 * completion
    )
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "avg_reward": avg_reward,
        "final_resolution": final_res,
        "efficiency": efficiency,
        "completion": completion,
        "success": 1.0 if (completed and score >= success_threshold) else 0.0,
    }
