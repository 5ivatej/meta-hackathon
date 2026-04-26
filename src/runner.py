"""Durable episode runner and checkpoint serialization."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import zlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .agentic import AgentMemory
from .client import ESCHttpClient, ResetResponse, StepResponse
from .models import Action, Observation


RUNNER_SECRET = os.getenv("ESC_RUNNER_SECRET", os.getenv("ESC_SESSION_SECRET", "esc-openenv-dev-secret")).encode(
    "utf-8"
)


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload: str) -> str:
    return hmac.new(RUNNER_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass
class RunnerCheckpoint:
    task_id: str
    env_session_token: str = ""
    agent_memory: Dict[str, object] = field(default_factory=dict)
    last_observation: Optional[Dict[str, object]] = None
    history_tail: List[str] = field(default_factory=list)
    cumulative_reward: float = 0.0
    step_count: int = 0

    def to_token(self) -> str:
        raw = json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        payload = _urlsafe_b64encode(zlib.compress(raw, level=9))
        return f"{payload}.{_sign(payload)}"

    @classmethod
    def from_token(cls, token: str) -> "RunnerCheckpoint":
        payload, signature = token.rsplit(".", 1)
        if not hmac.compare_digest(signature, _sign(payload)):
            raise RuntimeError("Invalid runner checkpoint signature")
        data = json.loads(zlib.decompress(_urlsafe_b64decode(payload)).decode("utf-8"))
        return cls(**data)


class DurableEpisodeRunner:
    """Checkpointable wrapper around the HTTP env client plus agent memory."""

    def __init__(self, env_client: ESCHttpClient, memory: Optional[AgentMemory] = None) -> None:
        self.env_client = env_client
        self.memory = memory or AgentMemory()
        self.task_id = ""
        self.last_observation: Optional[Observation] = None
        self.history_tail: List[str] = []
        self.cumulative_reward: float = 0.0
        self.step_count: int = 0

    async def reset(self, task_id: str) -> ResetResponse:
        self.task_id = task_id
        self.memory.reset(task_id)
        self.history_tail = []
        self.cumulative_reward = 0.0
        self.step_count = 0
        result = await self.env_client.reset(task_id=task_id)
        self.last_observation = result.observation
        return result

    async def step(self, action: Action) -> StepResponse:
        result = await self.env_client.step(action)
        self.last_observation = result.observation
        self.cumulative_reward += float(result.reward)
        self.step_count += 1
        return result

    def push_history(self, line: str) -> None:
        self.history_tail.append(line)
        self.history_tail = self.history_tail[-8:]

    def checkpoint(self) -> str:
        checkpoint = RunnerCheckpoint(
            task_id=self.task_id,
            env_session_token=self.env_client.export_session_token() or "",
            agent_memory=self.memory.to_dict(),
            last_observation=self.last_observation.model_dump() if self.last_observation else None,
            history_tail=list(self.history_tail),
            cumulative_reward=self.cumulative_reward,
            step_count=self.step_count,
        )
        return checkpoint.to_token()

    @classmethod
    def from_checkpoint(
        cls,
        token: str,
        env_client: ESCHttpClient,
    ) -> "DurableEpisodeRunner":
        checkpoint = RunnerCheckpoint.from_token(token)
        env_client.import_session_token(checkpoint.env_session_token)
        runner = cls(env_client=env_client, memory=AgentMemory.from_dict(checkpoint.agent_memory))
        runner.task_id = checkpoint.task_id
        runner.history_tail = list(checkpoint.history_tail)
        runner.cumulative_reward = float(checkpoint.cumulative_reward)
        runner.step_count = int(checkpoint.step_count)
        if checkpoint.last_observation is not None:
            runner.last_observation = Observation(**checkpoint.last_observation)
        return runner
