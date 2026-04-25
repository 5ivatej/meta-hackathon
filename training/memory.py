"""Persistent episode memory for longer-horizon conversations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import List


RISK_PATTERNS = {
    "self_harm": [r"\bdark thoughts\b", r"\bdon't want to be here\b", r"\bhurt myself\b"],
    "burnout": [r"\bburning out\b", r"\bburned out\b", r"\bcan't keep doing this\b"],
    "relationship_breakdown": [r"\bseparating\b", r"\bdivorce\b", r"\bbreaking up\b"],
}


def _extract_risk_flags(text: str) -> List[str]:
    lowered = text.lower()
    flags: List[str] = []
    for label, patterns in RISK_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            flags.append(label)
    return flags


@dataclass
class EpisodeMemory:
    summary: str = ""
    seeker_facts: List[str] = field(default_factory=list)
    unresolved_needs: List[str] = field(default_factory=list)
    commitments: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    recent_turns: List[str] = field(default_factory=list)
    current_goal_hint: str = ""
    last_session_outcome: str = ""
    env_task_id: str = ""
    session_index: int = 1
    sessions_total: int = 1
    budget_ratio: float = 0.0
    time_ratio: float = 0.0

    def clone(self) -> "EpisodeMemory":
        return deepcopy(self)

    def note_user_message(self, message: str) -> None:
        text = message.strip()
        if text:
            self.recent_turns.append(f"User: {text}")
            self.recent_turns = self.recent_turns[-10:]
            for flag in _extract_risk_flags(text):
                if flag not in self.risk_flags:
                    self.risk_flags.append(flag)

    def note_assistant_message(self, message: str) -> None:
        text = message.strip()
        if text:
            self.recent_turns.append(f"Assistant: {text}")
            self.recent_turns = self.recent_turns[-10:]

    def add_fact(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.seeker_facts:
            self.seeker_facts.append(fact)

    def add_unresolved_need(self, need: str) -> None:
        need = need.strip()
        if need and need not in self.unresolved_needs:
            self.unresolved_needs.append(need)

    def add_commitment(self, commitment: str) -> None:
        commitment = commitment.strip()
        if commitment and commitment not in self.commitments:
            self.commitments.append(commitment)

    def sync_from_env(self, env_state: dict, info: dict | None = None) -> None:
        self.env_task_id = str(env_state.get("task_id") or self.env_task_id)
        self.summary = str(env_state.get("memory_summary") or self.summary)
        self.current_goal_hint = str(env_state.get("current_goal_hint") or self.current_goal_hint)
        self.last_session_outcome = str(env_state.get("last_session_outcome") or self.last_session_outcome)
        self.session_index = int(env_state.get("session_index", self.session_index))
        self.sessions_total = int(env_state.get("sessions_total", self.sessions_total))

        unfinished_threads = [str(item) for item in env_state.get("unfinished_threads", []) or []]
        for need in unfinished_threads:
            self.add_unresolved_need(need)

        task_budget = float(env_state.get("episode_budget_limit") or 0.0)
        spent_budget = float(env_state.get("episode_budget_spent") or 0.0)
        if task_budget > 0:
            self.budget_ratio = spent_budget / task_budget

        task_time = float(env_state.get("episode_time_limit") or 0.0)
        spent_time = float(env_state.get("episode_time_spent") or 0.0)
        if task_time > 0:
            self.time_ratio = spent_time / task_time

        seeker_state = env_state.get("seeker") or {}
        if seeker_state:
            stage = str(seeker_state.get("stage") or "").strip()
            if stage:
                self.add_fact(f"Current stage: {stage}.")
            if bool(seeker_state.get("revealed")):
                self.add_fact("The core issue has been disclosed.")
            distress = seeker_state.get("distress")
            trust = seeker_state.get("trust")
            if distress is not None:
                self.add_fact(f"Distress estimate: {float(distress):.2f}.")
            if trust is not None:
                self.add_fact(f"Trust estimate: {float(trust):.2f}.")

        if info and bool(info.get("had_safety_reference")):
            if "safety_support_named" not in self.risk_flags:
                self.risk_flags.append("safety_support_named")

    def _structured_summary(self) -> str:
        parts: List[str] = []
        if self.env_task_id:
            parts.append(f"Task: {self.env_task_id}.")
        if self.session_index or self.sessions_total:
            parts.append(f"Session: {self.session_index}/{self.sessions_total}.")
        if self.current_goal_hint:
            parts.append("Goal hint: " + self.current_goal_hint)
        if self.last_session_outcome:
            parts.append("Last session: " + self.last_session_outcome)
        if self.seeker_facts:
            parts.append("Facts: " + "; ".join(self.seeker_facts[-4:]))
        if self.unresolved_needs:
            parts.append("Needs: " + "; ".join(self.unresolved_needs[-4:]))
        if self.commitments:
            parts.append("Commitments: " + "; ".join(self.commitments[-3:]))
        if self.risk_flags:
            parts.append("Risk flags: " + ", ".join(self.risk_flags))
        if self.budget_ratio or self.time_ratio:
            parts.append(
                "Resource usage: "
                f"budget={self.budget_ratio:.2f}, time={self.time_ratio:.2f}"
            )
        if self.recent_turns:
            parts.append("Recent: " + " | ".join(self.recent_turns[-4:]))
        return "\n".join(parts).strip()

    def refresh_summary(self) -> None:
        self.summary = self._structured_summary()

    def render_for_prompt(self) -> str:
        structured = self._structured_summary()
        narrative = self.summary.strip()
        if narrative and structured and narrative != structured:
            return f"Summary: {narrative}\n{structured}"
        if narrative:
            return narrative
        if structured:
            return structured
        return "No durable memory yet."
