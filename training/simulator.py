"""Environment-backed simulator used for training rollouts."""
from __future__ import annotations

from dataclasses import dataclass

from src.env import ESCEnv
from src.models import Action, Observation


@dataclass
class SimulatorTransition:
    observation: Observation
    reward: float
    reward_detail: dict
    done: bool
    info: dict
    env_state: dict
    user_message: str


class EnvironmentSimulator:
    """Environment-backed simulator aligned with the OpenEnv task semantics."""

    def __init__(self, task_id: str) -> None:
        self.env = ESCEnv()
        reset = self.env.reset(task_id=task_id)
        self.last_observation = reset.observation

    @classmethod
    def from_exported_state(cls, state: dict) -> "EnvironmentSimulator":
        simulator = cls.__new__(cls)
        simulator.env = ESCEnv.from_state(state)
        exported = simulator.env.export_state()
        simulator.last_observation = Observation(**exported["last_observation"])
        return simulator

    def export_state(self) -> dict:
        return self.env.export_state()

    def transcript_for_chat(self) -> list[dict[str, str]]:
        transcript = []
        for entry in self.export_state().get("transcript", []):
            role = "user" if entry.get("role") == "seeker" else "assistant"
            transcript.append({"role": role, "content": str(entry.get("text") or "")})
        return transcript

    def step_assistant(self, message: str) -> SimulatorTransition:
        result = self.env.step(Action(message=message))
        self.last_observation = result.observation
        transcript = self.export_state().get("transcript", [])
        last_user_message = ""
        for entry in reversed(transcript):
            if entry.get("role") == "seeker":
                last_user_message = str(entry.get("text") or "")
                break
        return SimulatorTransition(
            observation=result.observation,
            reward=float(result.reward),
            reward_detail=result.reward_detail.model_dump(),
            done=bool(result.done),
            info=dict(result.info),
            env_state=self.export_state(),
            user_message=last_user_message,
        )
