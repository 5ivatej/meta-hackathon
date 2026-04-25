"""Paper-style critic grounded in environment rollouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import SimulationConfig
from .datasets import SeedExample
from .llm import chat_json
from .memory import EpisodeMemory
from .prompts import build_critic_messages


@dataclass
class CriticJudgment:
    score: float
    goal_achieved: bool
    rationale: str


@dataclass
class FutureRewardResult:
    reward: float
    terminal_score: float
    goal_achieved: bool
    steps_used: int
    rationale: str
    env_final_score: float
    env_final_success: float
    env_cumulative_reward: float
    env_stage: str


class FutureRewardCritic:
    """Separated critic role used to estimate future-oriented reward."""

    def __init__(self, client, config: SimulationConfig) -> None:
        self.client = client
        self.config = config

    def evaluate(
        self,
        seed: SeedExample,
        transcript: List[dict[str, str]],
        memory: EpisodeMemory,
        env_state: dict,
        rollout_step_rewards: List[float],
        final_info: dict,
    ) -> CriticJudgment:
        payload = chat_json(
            client=self.client,
            model=self.config.critic_model,
            messages=build_critic_messages(
                seed=seed,
                transcript=transcript,
                memory=memory,
                env_state=env_state,
                rollout_step_rewards=rollout_step_rewards,
                final_info=final_info,
            ),
            temperature=0.0,
            max_tokens=256,
        )
        score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
        goal_achieved = bool(payload.get("goal_achieved", payload.get("completed", False)))
        rationale = str(payload.get("rationale") or "")
        return CriticJudgment(score=score, goal_achieved=goal_achieved, rationale=rationale)

    def compute_future_reward(self, terminal_score: float) -> float:
        return max(0.0, min(1.0, terminal_score))
