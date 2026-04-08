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

import base64
import hashlib
import hmac
import json
import os
import zlib

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
SESSION_SECRET = os.getenv("ESC_SESSION_SECRET", "esc-openenv-dev-secret").encode("utf-8")


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload: str) -> str:
    return hmac.new(SESSION_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_env(env: ESCEnv) -> str:
    raw = json.dumps(env.export_state(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = _urlsafe_b64encode(zlib.compress(raw, level=9))
    return f"{payload}.{_sign(payload)}"


def _decode_env(token: str) -> ESCEnv:
    try:
        payload, signature = token.rsplit(".", 1)
    except ValueError as exc:
        raise RuntimeError("Invalid session token") from exc

    expected = _sign(payload)
    if not hmac.compare_digest(signature, expected):
        raise RuntimeError("Invalid session signature")

    try:
        compressed = _urlsafe_b64decode(payload)
        data = json.loads(zlib.decompress(compressed).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Invalid session payload") from exc

    return ESCEnv.from_state(data)


def _get_env_for_request(request: Request) -> ESCEnv:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise RuntimeError("env.step() called before reset()")
    return _decode_env(token)


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
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        try:
            env = _decode_env(token)
        except RuntimeError:
            env = ESCEnv()
    else:
        env = ESCEnv()
    try:
        result = env.reset(task_id=req.task_id, seed=req.seed)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    response.set_cookie(
        key=SESSION_COOKIE,
        value=_encode_env(env),
        httponly=True,
        samesite="lax",
    )
    return result.model_dump()


@app.post("/step")
def step(req: StepRequest, request: Request, response: Response) -> dict:
    try:
        env = _get_env_for_request(request)
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    response.set_cookie(
        key=SESSION_COOKIE,
        value=_encode_env(env),
        httponly=True,
        samesite="lax",
    )
    return result.model_dump()


@app.get("/state")
def state(request: Request) -> dict:
    try:
        return _get_env_for_request(request).state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
