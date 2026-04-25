"""Paper-style critic and future-reward computation."""
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
    ) -> CriticJudgment:
        payload = chat_json(
            client=self.client,
            model=self.config.critic_model,
            messages=build_critic_messages(seed=seed, transcript=transcript, memory=memory),
            temperature=0.0,
            max_tokens=256,
        )
        score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
        goal_achieved = bool(payload.get("goal_achieved", payload.get("completed", False)))
        rationale = str(payload.get("rationale") or "")
        if score >= self.config.critic_completion_threshold:
            goal_achieved = True
        return CriticJudgment(score=score, goal_achieved=goal_achieved, rationale=rationale)

    def compute_future_reward(self, terminal_score: float, steps_used: int) -> float:
        avg_turn = max(1, steps_used)
        return terminal_score + (self.config.success_turn_bonus / avg_turn)
