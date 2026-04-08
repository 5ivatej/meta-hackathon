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

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi import Request, Response

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

SESSION_COOKIE = "esc_session_id"
_envs: dict[str, ESCEnv] = {}


def _get_env_for_request(request: Request) -> ESCEnv:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or session_id not in _envs:
        raise RuntimeError("env.step() called before reset()")
    return _envs[session_id]


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
def reset(request: Request, response: Response, req: ResetRequest | None = None) -> dict:
    req = req or ResetRequest()
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        session_id = uuid4().hex
    env = _envs.get(session_id)
    if env is None:
        env = ESCEnv()
        _envs[session_id] = env
    try:
        result = env.reset(task_id=req.task_id, seed=req.seed)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    response.set_cookie(key=SESSION_COOKIE, value=session_id, httponly=True, samesite="lax")
    return result.model_dump()


@app.post("/step")
def step(req: StepRequest, request: Request) -> dict:
    try:
        result = _get_env_for_request(request).step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return result.model_dump()


@app.get("/state")
def state(request: Request) -> dict:
    try:
        return _get_env_for_request(request).state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
