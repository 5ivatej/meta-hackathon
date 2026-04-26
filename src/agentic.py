"""Agentic skill-routed policies and durable policy memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    session_index: int = 1
    sessions_total: int = 1
    used_safety: bool = False
    seeker_revealed: bool = False
    rolling_summary: str = ""
    last_session_outcome: str = ""
    current_goal_hint: str = ""
    episode_budget_spent: float = 0.0
    episode_budget_limit: float = 0.0
    episode_time_spent: float = 0.0
    episode_time_limit: float = 0.0
    risk_markers: List[str] = field(default_factory=list)
    unresolved_threads: List[str] = field(default_factory=list)
    recent_messages: List[str] = field(default_factory=list)
    recent_skills: List[str] = field(default_factory=list)
    recent_turns: List[str] = field(default_factory=list)
    message_index_by_key: Dict[str, int] = field(default_factory=dict)
    skill_counts: Dict[str, int] = field(default_factory=dict)

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.turns_seen = 0
        self.session_index = 1
        self.sessions_total = 1
        self.used_safety = False
        self.seeker_revealed = False
        self.rolling_summary = ""
        self.last_session_outcome = ""
        self.current_goal_hint = ""
        self.episode_budget_spent = 0.0
        self.episode_budget_limit = 0.0
        self.episode_time_spent = 0.0
        self.episode_time_limit = 0.0
        self.risk_markers = []
        self.unresolved_threads = []
        self.recent_messages = []
        self.recent_skills = []
        self.recent_turns = []
        self.message_index_by_key = {}
        self.skill_counts = {}

    def observe(self, observation: Observation) -> None:
        self.task_id = observation.task_id
        self.turns_seen = observation.turn
        self.session_index = observation.session_index
        self.sessions_total = observation.sessions_total
        self.rolling_summary = observation.memory_summary or self.rolling_summary
        self.last_session_outcome = observation.last_session_outcome or self.last_session_outcome
        self.current_goal_hint = observation.current_goal_hint or self.current_goal_hint
        self.episode_budget_spent = float(observation.episode_budget_spent)
        self.episode_budget_limit = float(observation.episode_budget_limit)
        self.episode_time_spent = float(observation.episode_time_spent)
        self.episode_time_limit = float(observation.episode_time_limit)
        markers = REVEAL_MARKERS.get(observation.task_id, [])
        if _contains_any(observation.seeker_utterance, markers):
            self.seeker_revealed = True
        if "dark thoughts" in observation.seeker_utterance.lower():
            self._add_unique(self.risk_markers, "dark thoughts")
        if observation.current_goal_hint:
            self._merge_goal_hint(observation.current_goal_hint)
        self._append_turn(f"Seeker: {observation.seeker_utterance}")

    def remember(self, skill_name: str, message: str) -> None:
        normalized = _normalized(message)
        self.recent_messages.append(normalized)
        self.recent_messages = self.recent_messages[-8:]
        self.recent_skills.append(skill_name)
        self.recent_skills = self.recent_skills[-8:]
        self.skill_counts[skill_name] = self.skill_counts.get(skill_name, 0) + 1
        self._append_turn(f"Agent: {message}")
        if skill_name == "safety_escalate":
            self.used_safety = True
            self._add_unique(self.risk_markers, "safety follow-up")
        if "small next step" in normalized or "next step" in normalized:
            self._add_unique(self.unresolved_threads, "follow through on the agreed next step")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "AgentMemory":
        memory = cls()
        for key, value in data.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        return memory

    def prompt_context(self, observation: Observation) -> str:
        lines: List[str] = []
        if self.rolling_summary:
            lines.append("Therapy arc summary:")
            lines.append(self.rolling_summary)
        if self.last_session_outcome:
            lines.append("")
            lines.append("Last session outcome:")
            lines.append(self.last_session_outcome)
        if self.current_goal_hint:
            lines.append("")
            lines.append("Current goal:")
            lines.append(self.current_goal_hint)
        if self.unresolved_threads:
            lines.append("")
            lines.append("Unresolved threads:")
            for item in self.unresolved_threads[:3]:
                lines.append(f"- {item}")
        if self.risk_markers:
            lines.append("")
            lines.append("Risk / guardrail reminders:")
            for item in self.risk_markers[:3]:
                lines.append(f"- {item}")
        if observation.episode_budget_limit > 0 or observation.episode_time_limit > 0:
            lines.append("")
            lines.append(
                "Budget status: "
                f"cost={observation.episode_budget_spent:.1f}/{observation.episode_budget_limit:.1f}, "
                f"time={observation.episode_time_spent:.1f}/{observation.episode_time_limit:.1f}"
            )
        lines.append("")
        lines.append("Recent local exchange:")
        if self.recent_turns:
            lines.extend(self.recent_turns[-6:])
        else:
            lines.append("(first turn)")
        return "\n".join(lines).strip()

    def checkpoint_summary(self) -> str:
        return self.rolling_summary or self.current_goal_hint or ""

    def _append_turn(self, text: str) -> None:
        self.recent_turns.append(text)
        self.recent_turns = self.recent_turns[-6:]

    def _add_unique(self, target: List[str], value: str) -> None:
        if value not in target:
            target.append(value)
            del target[:-4]

    def _merge_goal_hint(self, hint: str) -> None:
        normalized = hint.strip()
        if normalized:
            self._add_unique(self.unresolved_threads, normalized)


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
