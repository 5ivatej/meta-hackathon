"""Deterministic baselines for the ESC benchmark.

These baselines are designed for three purposes:

1. Provide trustworthy local benchmark numbers without external APIs.
2. Stress-test whether the rubric can distinguish weak/generic behavior from
   staged, task-aware behavior.
3. Give the README a clear "weak / medium / strong" baseline ladder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from .models import Observation


class BaselinePolicy(Protocol):
    """Minimal interface shared by all deterministic baselines."""

    name: str

    def reset(self, task_id: str) -> None:
        """Reset any task-specific internal state."""

    def act(self, observation: Observation) -> str:
        """Produce the next agent message."""


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class GenericTemplateBaseline:
    """Intentionally weak baseline: safe-sounding but repetitive."""

    name: str = "generic_template"
    template: str = (
        "That sounds really hard, and it makes sense that you're feeling this way. "
        "Can you tell me a little more about what's weighing on you most right now?"
    )

    def reset(self, task_id: str) -> None:
        self.task_id = task_id

    def act(self, observation: Observation) -> str:
        return self.template


@dataclass
class ValidationOnlyBaseline:
    """Weak baseline that validates but never advances toward resolution."""

    name: str = "validation_only"
    template: str = (
        "Your feelings make sense, and anyone would feel overwhelmed in your position."
    )

    def reset(self, task_id: str) -> None:
        self.task_id = task_id

    def act(self, observation: Observation) -> str:
        return self.template


class StageAwareHeuristicBaseline:
    """Task-aware deterministic baseline that follows the rubric intentionally.

    It tries to:
    - build trust early with empathy + validation + one open question
    - avoid advice before planning
    - keep responses varied to avoid repetition penalties
    - add safety-aware closing behavior for the crisis task
    """

    name = "stage_aware_heuristic"

    def __init__(self) -> None:
        self.task_id = ""
        self.turn = 0
        self.used_safety = False
        self.recent_messages: List[str] = []
        self.message_index_by_key: Dict[str, int] = {}

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.turn = 0
        self.used_safety = False
        self.recent_messages = []
        self.message_index_by_key = {}

    def _pick(self, key: str, options: List[str]) -> str:
        """Rotate through message variants while avoiding immediate repeats."""
        start = self.message_index_by_key.get(key, 0)
        for offset in range(len(options)):
            idx = (start + offset) % len(options)
            candidate = options[idx]
            if _normalized(candidate) not in self.recent_messages[-2:]:
                self.message_index_by_key[key] = idx + 1
                return candidate
        candidate = options[start % len(options)]
        self.message_index_by_key[key] = start + 1
        return candidate

    def _remember(self, message: str) -> str:
        self.recent_messages.append(_normalized(message))
        return message

    def act(self, observation: Observation) -> str:
        self.turn += 1
        seeker = observation.seeker_utterance.lower()
        stage = observation.stage_hint

        if stage == "opening":
            return self._remember(
                self._pick(
                    "opening",
                    [
                        "That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?",
                        "I'm really glad you said that out loud. It makes sense that this feels heavy. What has felt hardest about it so far?",
                    ],
                )
            )

        if stage == "exploring":
            if self.task_id == "guarded_relationship":
                return self._remember(
                    self._pick(
                        "exploring_guarded",
                        [
                            "It's understandable to feel guarded when something this personal is unclear. Can you tell me more about what has felt strange this week?",
                            "Your feelings make sense, and you don't have to explain everything at once. What has been sitting with you the most?",
                        ],
                    )
                )
            if self.task_id == "crisis_fragile_trust":
                return self._remember(
                    self._pick(
                        "exploring_crisis",
                        [
                            "Anyone carrying that much would feel overwhelmed. Can you tell me more about what feels hardest right this second?",
                            "It makes sense that your head feels overloaded. What has felt heaviest tonight?",
                        ],
                    )
                )
            return self._remember(
                self._pick(
                    "exploring_work",
                    [
                        "Your feelings make sense, and anyone would feel overwhelmed by that kind of pressure. What has been hitting you the hardest?",
                        "That sounds exhausting. It makes sense that you're drained. Can you tell me more about what has felt most unsustainable?",
                    ],
                )
            )

        if stage == "reflecting":
            if "dark thoughts" in seeker and self.task_id == "crisis_fragile_trust":
                return self._remember(
                    self._pick(
                        "reflecting_crisis_reveal",
                        [
                            "Thank you for trusting me with that. Your feelings are valid, and anyone in your position would feel shaken and exhausted.",
                            "I hear how serious and painful that is. It makes sense that you're overwhelmed, and I'm really glad you said it out loud.",
                        ],
                    )
                )
            if "separating" in seeker or "burning out" in seeker:
                return self._remember(
                    self._pick(
                        "reflecting_reveal",
                        [
                            "Thank you for trusting me with that. Your feelings make sense, and you don't have to carry it alone here.",
                            "I hear how much courage it took to say that. Anyone would feel overwhelmed trying to hold that by themselves.",
                        ],
                    )
                )
            return self._remember(
                self._pick(
                    "reflecting_general",
                    [
                        "I hear how much this has been building up, and your feelings make sense.",
                        "That makes a lot of sense, and anyone in your position would feel overwhelmed.",
                    ],
                )
            )

        if stage == "planning":
            if self.task_id == "crisis_fragile_trust" and not self.used_safety:
                self.used_safety = True
                return self._remember(
                    self._pick(
                        "planning_crisis_safety",
                        [
                            "You don't have to solve everything tonight. Reaching out to a crisis line, therapist, or another trusted person tonight could be a strong next step if those dark thoughts feel harder to manage.",
                            "A gentle next step could be staying connected to real support tonight, like a crisis line, therapist, or someone you trust who can be with you in this.",
                        ],
                    )
                )
            return self._remember(
                self._pick(
                    "planning_general",
                    [
                        "It makes sense not to solve everything at once. You could try doing one small next step that feels manageable tonight.",
                        "You don't have to fix it all right now. A gentle next step could be trying one small, supportive action after this conversation.",
                    ],
                )
            )

        # closing
        if self.task_id == "crisis_fragile_trust":
            return self._remember(
                self._pick(
                    "closing_crisis",
                    [
                        "I'm glad you stayed with me in this. Your feelings are valid, and focusing on getting through tonight safely makes a lot of sense.",
                        "Thank you for staying in the conversation. You deserve support, and it makes sense to keep tonight centered on safety and care.",
                    ],
                )
            )
        return self._remember(
            self._pick(
                "closing_general",
                [
                    "Thank you for sharing this so honestly. Your feelings are valid, and I'm glad you didn't have to hold it alone here.",
                    "I'm really glad you said all of this out loud. It makes sense that it feels a little lighter after being heard.",
                ],
            )
        )


def make_default_baselines() -> List[BaselinePolicy]:
    """Default ladder used by the benchmark runner."""
    return [
        GenericTemplateBaseline(),
        ValidationOnlyBaseline(),
        StageAwareHeuristicBaseline(),
    ]
