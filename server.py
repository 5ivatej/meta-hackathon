"""FastAPI HTTP server exposing the OpenEnv interface for the ESC environment.

Endpoints
---------
GET  /             → health check + metadata
POST /reset        → reset episode (optional task_id), returns initial Observation
POST /step         → take one step with {"action": {"message": "..."}}
GET  /state        → return current EnvState
GET  /tasks        → list available tasks + difficulties

The server holds a single in-process ESCEnv instance. For parallel eval,
deploy multiple replicas — the env itself has no shared state between
instances.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.env import ESCEnv
from src.models import ResetRequest, StepRequest

app = FastAPI(
    title="Emotional Support Conversations (OpenEnv)",
    version="0.1.0",
    description=(
        "An OpenEnv environment for open-ended emotional support "
        "conversations. Reward shaping inspired by RLFF-ESC "
        "(arXiv:2508.12935)."
    ),
)

_env = ESCEnv()


@app.get("/")
def root() -> dict:
    return {
        "name": "emotional-support-conversations",
        "version": "0.1.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks"],
        "tasks": [t["id"] for t in ESCEnv.list_tasks()],
    }


@app.get("/tasks")
def list_tasks() -> dict:
    return {"tasks": ESCEnv.list_tasks()}


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict:
    req = req or ResetRequest()
    try:
        result = _env.reset(task_id=req.task_id, seed=req.seed)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


@app.post("/step")
def step(req: StepRequest) -> dict:
    try:
        result = _env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    try:
        return _env.state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
