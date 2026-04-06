"""Async HTTP client mirroring the OpenEnv env interface.

Used by inference.py to interact with the running FastAPI server (local or
HF Space deployment).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .models import Action, Observation


@dataclass
class StepResponse:
    observation: Observation
    reward: float
    reward_detail: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


@dataclass
class ResetResponse:
    observation: Observation
    info: Dict[str, Any]


class ESCHttpClient:
    """Thin async client for the ESC OpenEnv server."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    @classmethod
    def from_url(cls, base_url: str) -> "ESCHttpClient":
        return cls(base_url=base_url)

    async def reset(self, task_id: Optional[str] = None) -> ResetResponse:
        payload: Dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        r = await self._client.post("/reset", json=payload)
        r.raise_for_status()
        data = r.json()
        return ResetResponse(
            observation=Observation(**data["observation"]),
            info=data.get("info", {}),
        )

    async def step(self, action: Action) -> StepResponse:
        r = await self._client.post("/step", json={"action": action.model_dump()})
        r.raise_for_status()
        data = r.json()
        return StepResponse(
            observation=Observation(**data["observation"]),
            reward=float(data["reward"]),
            reward_detail=data.get("reward_detail", {}),
            done=bool(data["done"]),
            info=data.get("info", {}),
        )

    async def state(self) -> Dict[str, Any]:
        r = await self._client.get("/state")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()
