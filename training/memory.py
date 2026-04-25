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

    def _structured_summary(self) -> str:
        parts: List[str] = []
        if self.seeker_facts:
            parts.append("Facts: " + "; ".join(self.seeker_facts[-4:]))
        if self.unresolved_needs:
            parts.append("Needs: " + "; ".join(self.unresolved_needs[-4:]))
        if self.commitments:
            parts.append("Commitments: " + "; ".join(self.commitments[-3:]))
        if self.risk_flags:
            parts.append("Risk flags: " + ", ".join(self.risk_flags))
        if self.recent_turns:
            parts.append("Recent: " + " | ".join(self.recent_turns[-4:]))
        return "\n".join(parts).strip()

    def refresh_summary(self) -> None:
        if not self.summary:
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
