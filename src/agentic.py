"""Agentic skill-routed policies for the ESC benchmark.

The environment itself stays deterministic and tool-free. This module adds an
explicit policy-side "agent" layer made of reusable conversational skills plus
deterministic routing logic. That gives the submission a clean skills/agents
story without weakening the reproducibility of the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol

from .models import Observation


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_any(text: str, markers: List[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


REVEAL_MARKERS: Dict[str, List[str]] = {
    "work_stress_venting": ["burning out"],
    "guarded_relationship": ["separating"],
    "crisis_fragile_trust": ["dark thoughts", "that's what's actually going on"],
}


@dataclass
class SkillDecision:
    skill_name: str
    rationale: str


@dataclass
class AgentMemory:
    task_id: str = ""
    turns_seen: int = 0
    used_safety: bool = False
    seeker_revealed: bool = False
    recent_messages: List[str] = field(default_factory=list)
    recent_skills: List[str] = field(default_factory=list)
    message_index_by_key: Dict[str, int] = field(default_factory=dict)
    skill_counts: Dict[str, int] = field(default_factory=dict)

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.turns_seen = 0
        self.used_safety = False
        self.seeker_revealed = False
        self.recent_messages = []
        self.recent_skills = []
        self.message_index_by_key = {}
        self.skill_counts = {}

    def observe(self, observation: Observation) -> None:
        self.task_id = observation.task_id
        self.turns_seen = observation.turn
        markers = REVEAL_MARKERS.get(observation.task_id, [])
        if _contains_any(observation.seeker_utterance, markers):
            self.seeker_revealed = True

    def remember(self, skill_name: str, message: str) -> None:
        self.recent_messages.append(_normalized(message))
        self.recent_skills.append(skill_name)
        self.skill_counts[skill_name] = self.skill_counts.get(skill_name, 0) + 1
        if skill_name == "safety_escalate":
            self.used_safety = True


class ConversationSkill(Protocol):
    name: str
    brief: str

    def render(self, observation: Observation, memory: AgentMemory, decision: SkillDecision) -> str:
        """Produce the next deterministic message."""

    def llm_instruction(
        self,
        observation: Observation,
        memory: AgentMemory,
        decision: SkillDecision,
    ) -> str:
        """Return a short instruction block for an LLM-backed agent."""


class BaseSkill:
    name = ""
    brief = ""

    def _pick(self, memory: AgentMemory, key: str, options: List[str]) -> str:
        start = memory.message_index_by_key.get(key, 0)
        for offset in range(len(options)):
            idx = (start + offset) % len(options)
            candidate = options[idx]
            if _normalized(candidate) not in memory.recent_messages[-2:]:
                memory.message_index_by_key[key] = idx + 1
                return candidate
        candidate = options[start % len(options)]
        memory.message_index_by_key[key] = start + 1
        return candidate

    def llm_instruction(
        self,
        observation: Observation,
        memory: AgentMemory,
        decision: SkillDecision,
    ) -> str:
        return self.brief


class EmpathizeSkill(BaseSkill):
    name = "empathize"
    brief = (
        "Lead with empathy and emotional attunement. Reflect the weight of what "
        "they are carrying, keep it warm, and ask at most one open question."
    )

    def render(self, observation: Observation, memory: AgentMemory, decision: SkillDecision) -> str:
        if observation.task_id == "crisis_fragile_trust":
            return self._pick(
                memory,
                "empathize_crisis",
                [
                    "That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?",
                    "I'm really glad you said that out loud. It makes sense that this feels heavy. What has felt hardest about it so far?",
                ],
            )
        if observation.task_id == "guarded_relationship":
            return self._pick(
                memory,
                "empathize_guarded",
                [
                    "That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?",
                    "I'm really glad you said that out loud. It makes sense that this feels heavy. What has felt hardest about it so far?",
                ],
            )
        return self._pick(
            memory,
            "empathize_work",
            [
                "That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?",
                "I'm really glad you said that out loud. It makes sense that this feels heavy. What has felt hardest about it so far?",
            ],
        )


class ValidateSkill(BaseSkill):
    name = "validate"
    brief = (
        "Reflect and validate what they shared. If they just disclosed the core "
        "issue, acknowledge the trust it took to say it. Do not pivot into advice."
    )

    def render(self, observation: Observation, memory: AgentMemory, decision: SkillDecision) -> str:
        seeker = observation.seeker_utterance.lower()
        if observation.stage_hint == "closing":
            if observation.task_id == "crisis_fragile_trust":
                return self._pick(
                    memory,
                    "validate_closing_crisis",
                    [
                        "I'm glad you stayed with me in this. Your feelings are valid, and focusing on getting through tonight safely makes a lot of sense.",
                        "Thank you for staying in the conversation. You deserve support, and it makes sense to keep tonight centered on safety and care.",
                    ],
                )
            return self._pick(
                memory,
                "validate_closing_general",
                [
                    "Thank you for sharing this so honestly. Your feelings are valid, and I'm glad you didn't have to hold it alone here.",
                    "I'm really glad you said all of this out loud. It makes sense that it feels a little lighter after being heard.",
                ],
            )

        if "dark thoughts" in seeker and observation.task_id == "crisis_fragile_trust":
            if observation.task_id == "crisis_fragile_trust":
                return self._pick(
                    memory,
                    "validate_reveal_crisis",
                    [
                        "Thank you for trusting me with that. Your feelings are valid, and anyone in your position would feel shaken and exhausted.",
                        "I hear how serious and painful that is. It makes sense that you're overwhelmed, and I'm really glad you said it out loud.",
                    ],
                )
        if "separating" in seeker or "burning out" in seeker:
            return self._pick(
                memory,
                "validate_reveal_general",
                [
                    "Thank you for trusting me with that. Your feelings make sense, and you don't have to carry it alone here.",
                    "I hear how much courage it took to say that. Anyone would feel overwhelmed trying to hold that by themselves.",
                ],
            )

        return self._pick(
            memory,
            "validate_general",
            [
                "I hear how much this has been building up, and your feelings make sense.",
                "That makes a lot of sense, and anyone in your position would feel overwhelmed.",
            ],
        )


class ExploreSkill(BaseSkill):
    name = "explore"
    brief = (
        "Gently invite more disclosure with one open-ended question. Stay warm and "
        "curious, and do not jump to solutions."
    )

    def render(self, observation: Observation, memory: AgentMemory, decision: SkillDecision) -> str:
        if observation.task_id == "guarded_relationship":
            return self._pick(
                memory,
                "explore_guarded",
                [
                    "It's understandable to feel guarded when something this personal is unclear. Can you tell me more about what has felt strange this week?",
                    "Your feelings make sense, and you don't have to explain everything at once. What has been sitting with you the most?",
                ],
            )
        if observation.task_id == "crisis_fragile_trust":
            return self._pick(
                memory,
                "explore_crisis",
                [
                    "Anyone carrying that much would feel overwhelmed. Can you tell me more about what feels hardest right this second?",
                    "It makes sense that your head feels overloaded. What has felt heaviest tonight?",
                ],
            )
        return self._pick(
            memory,
            "explore_work",
            [
                "Your feelings make sense, and anyone would feel overwhelmed by that kind of pressure. What has been hitting you the hardest?",
                "That sounds exhausting. It makes sense that you're drained. Can you tell me more about what has felt most unsustainable?",
            ],
        )


class PlanSkill(BaseSkill):
    name = "plan"
    brief = (
        "Offer one gentle, manageable next step now that trust is built. Keep it "
        "tentative and supportive, not prescriptive or list-heavy."
    )

    def render(self, observation: Observation, memory: AgentMemory, decision: SkillDecision) -> str:
        return self._pick(
            memory,
            "plan_general",
            [
                "It makes sense not to solve everything at once. You could try doing one small next step that feels manageable tonight.",
                "You don't have to fix it all right now. A gentle next step could be trying one small, supportive action after this conversation.",
            ],
        )


class SafetyEscalateSkill(BaseSkill):
    name = "safety_escalate"
    brief = (
        "Stay warm and calm while explicitly encouraging real-world support tonight, "
        "such as a crisis line, therapist, or trusted person who can stay connected."
    )

    def render(self, observation: Observation, memory: AgentMemory, decision: SkillDecision) -> str:
        return self._pick(
            memory,
            "safety_escalate",
            [
                "You don't have to solve everything tonight. Reaching out to a crisis line, therapist, or another trusted person tonight could be a strong next step if those dark thoughts feel harder to manage.",
                "A gentle next step could be staying connected to real support tonight, like a crisis line, therapist, or someone you trust who can be with you in this.",
            ],
        )


class SkillRouter:
    """Deterministic routing logic over a small reusable skill library."""

    def choose(self, observation: Observation, memory: AgentMemory) -> SkillDecision:
        stage = observation.stage_hint

        if stage == "opening":
            return SkillDecision(
                skill_name="empathize",
                rationale="Early turns should prioritize attunement and psychological safety.",
            )

        if stage == "exploring":
            return SkillDecision(
                skill_name="explore",
                rationale="This phase is for careful disclosure, so the agent should keep exploring with one warm question.",
            )

        if stage == "reflecting":
            return SkillDecision(
                skill_name="validate",
                rationale="This stage rewards reflection and trust-building more than solutioning.",
            )

        if stage == "planning":
            if observation.task_id == "crisis_fragile_trust" and not memory.used_safety:
                return SkillDecision(
                    skill_name="safety_escalate",
                    rationale="Planning on the hard task should include safety support before anything else.",
                )
            return SkillDecision(
                skill_name="plan",
                rationale="Trust is established enough to move toward one gentle next step.",
            )

        return SkillDecision(
            skill_name="validate",
            rationale="Closing turns should stabilize the seeker with affirmation and reflection.",
        )


class SkillRoutedDeterministicPolicy:
    """Deterministic agentic baseline with explicit skill routing."""

    name = "skill_routed_deterministic"

    def __init__(self) -> None:
        self.router = SkillRouter()
        self.skills = build_default_skills()
        self.memory = AgentMemory()
        self.last_decision: SkillDecision | None = None
        self.decision_log: List[Dict[str, str]] = []

    def reset(self, task_id: str) -> None:
        self.memory.reset(task_id)
        self.last_decision = None
        self.decision_log = []

    def act(self, observation: Observation) -> str:
        self.memory.observe(observation)
        decision = self.router.choose(observation, self.memory)
        skill = self.skills[decision.skill_name]
        message = skill.render(observation, self.memory, decision)
        self.memory.remember(decision.skill_name, message)
        self.last_decision = decision
        self.decision_log.append(
            {
                "turn": str(observation.turn),
                "stage": observation.stage_hint,
                "skill": decision.skill_name,
                "reason": decision.rationale,
                "message": message,
            }
        )
        return message


def build_default_skills() -> Dict[str, ConversationSkill]:
    skills: List[ConversationSkill] = [
        EmpathizeSkill(),
        ValidateSkill(),
        ExploreSkill(),
        PlanSkill(),
        SafetyEscalateSkill(),
    ]
    return {skill.name: skill for skill in skills}
