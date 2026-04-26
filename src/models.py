"""Typed Pydantic models for the ESC OpenEnv environment."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """Agent action: a free-text conversational reply to the seeker."""

    message: str = Field(..., description="Agent's reply to the seeker.")


class Observation(BaseModel):
    """What the agent sees each turn."""

    seeker_utterance: str = Field(..., description="The seeker's latest message.")
    turn: int = Field(..., description="1-indexed conversation turn.")
    remaining_turns: int = Field(..., description="Turns left before forced close.")
    stage_hint: str = Field(
        ...,
        description=(
            "Coarse public hint about conversational phase: one of "
            "'opening', 'exploring', 'reflecting', 'planning', 'closing'."
        ),
    )
    task_id: str = Field(..., description="Currently active task id.")
    scenario_brief: str = Field(
        ...,
        description="One-line scenario framing shown once at reset (kept in obs for convenience).",
    )
    session_index: int = Field(default=1, description="1-indexed therapy session number.")
    sessions_total: int = Field(default=1, description="Total sessions planned in the current episode.")
    remaining_session_turns: int = Field(
        default=0,
        description="Turns left in the current session before the next session transition.",
    )
    memory_summary: str = Field(
        default="",
        description="Rolling summary of important prior context carried across sessions.",
    )
    last_session_outcome: str = Field(
        default="",
        description="Compact description of how the previous session ended.",
    )
    current_goal_hint: str = Field(
        default="",
        description="High-level goal the agent should keep in mind for continuity.",
    )
    episode_budget_spent: float = Field(
        default=0.0,
        description="Approximate budget usage consumed so far in the episode.",
    )
    episode_budget_limit: float = Field(
        default=0.0,
        description="Budget cap for the full episode.",
    )
    episode_time_spent: float = Field(
        default=0.0,
        description="Approximate time budget consumed so far in the episode.",
    )
    episode_time_limit: float = Field(
        default=0.0,
        description="Time cap for the full episode.",
    )


class Reward(BaseModel):
    """Detailed reward breakdown for a single step.

    The scalar `value` is what the agent sees. The decomposition is exposed
    for transparency and debugging.
    """

    value: float = Field(..., ge=0.0, le=1.0, description="Clipped step reward in [0,1].")
    immediate: float = Field(..., description="Immediate turn-level component (empathy, stage-fit).")
    future_oriented: float = Field(
        ...,
        description=(
            "Future-oriented component: k-step lookahead over the deterministic "
            "seeker dynamics, comparing this action's projected resolution "
            "progress against the oracle ceiling (RLFF-ESC style)."
        ),
    )
    penalties: float = Field(..., description="Summed penalties (dismissive, premature advice, loops).")
    components: Dict[str, float] = Field(default_factory=dict, description="Sub-component breakdown.")


class StepResult(BaseModel):
    """Envelope returned by env.step()."""

    observation: Observation
    reward: float
    reward_detail: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Envelope returned by env.reset()."""

    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    """Public view of environment state returned by env.state()."""

    task_id: str
    turn: int
    max_turns: int
    done: bool
    cumulative_reward: float
    session_index: int = 1
    sessions_total: int = 1
    remaining_session_turns: int = 0
    memory_summary: str = ""
    current_goal_hint: str = ""
    last_session_outcome: str = ""
    episode_budget_spent: float = 0.0
    episode_budget_limit: float = 0.0
    episode_time_spent: float = 0.0
    episode_time_limit: float = 0.0
    transcript: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {'role': 'seeker'|'agent', 'text': str} entries.",
    )


# ------- Request schemas for the HTTP server -------


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description="Optional task id. If omitted, defaults to 'work_stress_venting'.",
    )
    seed: Optional[int] = Field(default=None, description="Optional seed (reserved; env is deterministic).")


class StepRequest(BaseModel):
    action: Action
