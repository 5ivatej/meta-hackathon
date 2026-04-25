"""Paper-style user simulator with explicit rollout state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import SimulationConfig
from .datasets import SeedExample
from .llm import chat_json
from .memory import EpisodeMemory
from .prompts import build_summary_messages, build_user_sim_messages


@dataclass
class SimulatorTurn:
    user_message: str
    completed: bool
    learned_facts: List[str]
    unresolved_needs: List[str]


class UserSimulator:
    """LLM user simulator separated from the critic, following the paper's roles."""

    def __init__(self, client, config: SimulationConfig) -> None:
        self.client = client
        self.config = config

    def generate_turn(
        self,
        seed: SeedExample,
        transcript: List[dict[str, str]],
        memory: EpisodeMemory,
    ) -> SimulatorTurn:
        payload = chat_json(
            client=self.client,
            model=self.config.user_model,
            messages=build_user_sim_messages(
                seed=seed,
                transcript=transcript,
                memory=memory,
                max_recent_turns=self.config.max_recent_turns_in_prompt,
            ),
            temperature=self.config.temperature,
            max_tokens=self.config.max_completion_tokens,
        )
        user_message = str(payload.get("user_message") or "").strip()
        if not user_message:
            raise RuntimeError("User simulator returned an empty `user_message`.")
        learned_facts = [str(item) for item in payload.get("learned_facts", []) or []]
        unresolved_needs = [str(item) for item in payload.get("unresolved_needs", []) or []]
        return SimulatorTurn(
            user_message=user_message,
            completed=bool(payload.get("completed")),
            learned_facts=learned_facts,
            unresolved_needs=unresolved_needs,
        )

    def apply_turn_to_memory(self, memory: EpisodeMemory, turn: SimulatorTurn) -> None:
        memory.note_user_message(turn.user_message)
        for fact in turn.learned_facts:
            memory.add_fact(fact)
        for need in turn.unresolved_needs:
            memory.add_unresolved_need(need)

    def refresh_summary(
        self,
        seed: SeedExample,
        transcript: List[dict[str, str]],
        memory: EpisodeMemory,
    ) -> None:
        if not self.config.summary_model:
            memory.refresh_summary()
            return
        payload = chat_json(
            client=self.client,
            model=self.config.summary_model,
            messages=build_summary_messages(seed=seed, transcript=transcript, memory=memory),
            temperature=0.0,
            max_tokens=256,
        )
        memory.summary = str(payload.get("summary") or memory.summary)
        memory.seeker_facts = [str(x) for x in payload.get("seeker_facts", []) or memory.seeker_facts]
        memory.unresolved_needs = [str(x) for x in payload.get("unresolved_needs", []) or memory.unresolved_needs]
        memory.commitments = [str(x) for x in payload.get("commitments", []) or memory.commitments]
        memory.risk_flags = [str(x) for x in payload.get("risk_flags", []) or memory.risk_flags]

