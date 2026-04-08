"""FastAPI HTTP server exposing the OpenEnv interface for the ESC environment.

Endpoints
---------
GET  /             → health check + metadata
POST /reset        → reset episode (optional task_id), returns initial Observation
POST /step         → take one step with {"action": {"message": "..."}}
GET  /state        → return current EnvState
GET  /tasks        → list available tasks + difficulties

This server is stateless across replicas: the current environment snapshot is
stored in a signed, compressed cookie so any replica can serve the next
request as long as all nodes share the same session secret.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import zlib

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

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
UI_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Emotional Support Conversations</title>
    <style>
      :root {
        --bg: #f3efe7;
        --panel: #fffaf4;
        --ink: #1c1c1c;
        --muted: #665f57;
        --line: #ddd3c5;
        --accent: #1f6f5f;
        --accent-2: #d98d3c;
        --seeker: #efe2d1;
        --agent: #dceddf;
        --error: #9e2b25;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(217,141,60,0.14), transparent 28%),
          radial-gradient(circle at top right, rgba(31,111,95,0.16), transparent 30%),
          linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
      }
      .shell {
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 16px 40px;
      }
      .hero {
        margin-bottom: 18px;
      }
      h1 {
        margin: 0 0 8px;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1;
      }
      .sub {
        color: var(--muted);
        max-width: 70ch;
        margin: 0;
      }
      .layout {
        display: grid;
        grid-template-columns: 320px 1fr;
        gap: 18px;
      }
      .panel {
        background: rgba(255, 250, 244, 0.92);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.06);
        backdrop-filter: blur(8px);
      }
      .stack { display: grid; gap: 12px; }
      label {
        display: block;
        font-size: 0.92rem;
        margin-bottom: 6px;
        color: var(--muted);
      }
      select, textarea, button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid var(--line);
        padding: 10px 12px;
        font: inherit;
      }
      select, textarea {
        background: #fffdf9;
        color: var(--ink);
      }
      textarea {
        min-height: 110px;
        resize: vertical;
      }
      button {
        background: var(--accent);
        color: #fff;
        border: 0;
        cursor: pointer;
        font-weight: 600;
      }
      button.secondary {
        background: #efe7dc;
        color: var(--ink);
        border: 1px solid var(--line);
      }
      button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .badge {
        background: #f4eadf;
        border: 1px solid var(--line);
        color: var(--muted);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.84rem;
      }
      .chat {
        display: grid;
        gap: 10px;
        min-height: 420px;
        max-height: 68vh;
        overflow: auto;
        padding-right: 4px;
      }
      .msg {
        border-radius: 16px;
        padding: 12px 14px;
        border: 1px solid var(--line);
        line-height: 1.45;
      }
      .msg small {
        display: block;
        margin-bottom: 6px;
        color: var(--muted);
      }
      .msg.seeker { background: var(--seeker); }
      .msg.agent { background: var(--agent); }
      .status {
        color: var(--muted);
        min-height: 1.4em;
      }
      .status.error { color: var(--error); }
      .reward {
        font-variant-numeric: tabular-nums;
        color: var(--accent);
        font-weight: 700;
      }
      .footer {
        margin-top: 8px;
        color: var(--muted);
        font-size: 0.9rem;
      }
      @media (max-width: 860px) {
        .layout { grid-template-columns: 1fr; }
        .chat { min-height: 320px; max-height: none; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="hero">
        <h1>Emotional Support Conversations</h1>
        <p class="sub">
          Interactive browser playground for the deterministic OpenEnv benchmark.
          The API stays unchanged; this page just calls <code>/tasks</code>,
          <code>/reset</code>, <code>/step</code>, and <code>/state</code>.
        </p>
      </div>
      <div class="layout">
        <div class="panel stack">
          <div>
            <label for="taskSelect">Task</label>
            <select id="taskSelect"></select>
          </div>
          <div class="meta" id="taskMeta"></div>
          <button id="resetBtn">Start / Reset Episode</button>
          <div>
            <label for="messageInput">Your reply</label>
            <textarea id="messageInput" placeholder="Type the agent response here..."></textarea>
          </div>
          <button id="stepBtn" disabled>Send Step</button>
          <button id="stateBtn" class="secondary">Refresh Public State</button>
          <div class="status" id="status"></div>
          <div class="footer">
            Tip: keep replies short, warm, and stage-aware. The hard task only
            succeeds if you eventually include real-world safety support.
          </div>
        </div>
        <div class="panel">
          <div class="meta" id="episodeMeta"></div>
          <div class="chat" id="chat"></div>
        </div>
      </div>
    </div>
    <script>
      const taskSelect = document.getElementById("taskSelect");
      const taskMeta = document.getElementById("taskMeta");
      const episodeMeta = document.getElementById("episodeMeta");
      const chat = document.getElementById("chat");
      const statusEl = document.getElementById("status");
      const messageInput = document.getElementById("messageInput");
      const resetBtn = document.getElementById("resetBtn");
      const stepBtn = document.getElementById("stepBtn");
      const stateBtn = document.getElementById("stateBtn");

      let currentDone = true;

      function setStatus(text, isError = false) {
        statusEl.textContent = text || "";
        statusEl.className = isError ? "status error" : "status";
      }

      function renderBadge(label, value) {
        return `<span class="badge"><strong>${label}:</strong> ${value}</span>`;
      }

      function renderEpisodeMeta(obs, reward = null, done = null) {
        const parts = [
          renderBadge("Task", obs.task_id),
          renderBadge("Stage", obs.stage_hint),
          renderBadge("Turn", obs.turn),
          renderBadge("Remaining", obs.remaining_turns),
        ];
        if (reward !== null) parts.push(renderBadge("Reward", `<span class="reward">${reward.toFixed(2)}</span>`));
        if (done !== null) parts.push(renderBadge("Done", done ? "true" : "false"));
        episodeMeta.innerHTML = parts.join("");
      }

      function addMessage(role, text) {
        const node = document.createElement("div");
        node.className = `msg ${role}`;
        node.innerHTML = `<small>${role === "agent" ? "Agent" : "Seeker"}</small>${text}`;
        chat.appendChild(node);
        chat.scrollTop = chat.scrollHeight;
      }

      async function loadTasks() {
        const res = await fetch("/tasks");
        const data = await res.json();
        taskSelect.innerHTML = "";
        data.tasks.forEach((task) => {
          const option = document.createElement("option");
          option.value = task.id;
          option.textContent = `${task.id} (${task.difficulty})`;
          option.dataset.meta = JSON.stringify(task);
          taskSelect.appendChild(option);
        });
        updateTaskMeta();
      }

      function updateTaskMeta() {
        const selected = taskSelect.options[taskSelect.selectedIndex];
        if (!selected) return;
        const task = JSON.parse(selected.dataset.meta);
        taskMeta.innerHTML = [
          renderBadge("Difficulty", task.difficulty),
          renderBadge("Max turns", task.max_turns),
          renderBadge("Success threshold", task.success_threshold),
        ].join("");
      }

      async function resetEpisode() {
        setStatus("Starting episode...");
        chat.innerHTML = "";
        const res = await fetch("/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task_id: taskSelect.value }),
        });
        const data = await res.json();
        currentDone = false;
        stepBtn.disabled = false;
        renderEpisodeMeta(data.observation);
        addMessage("seeker", data.observation.seeker_utterance);
        setStatus(data.observation.scenario_brief);
      }

      async function sendStep() {
        const message = messageInput.value.trim();
        if (!message) {
          setStatus("Write a reply before sending.", true);
          return;
        }
        if (currentDone) {
          setStatus("Episode is finished. Reset before sending another step.", true);
          return;
        }

        stepBtn.disabled = true;
        setStatus("Sending step...");
        addMessage("agent", message);
        const res = await fetch("/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: { message } }),
        });

        if (!res.ok) {
          const err = await res.text();
          setStatus(err, true);
          stepBtn.disabled = false;
          return;
        }

        const data = await res.json();
        messageInput.value = "";
        addMessage("seeker", data.observation.seeker_utterance);
        renderEpisodeMeta(data.observation, data.reward, data.done);
        currentDone = Boolean(data.done);
        stepBtn.disabled = currentDone;
        if (data.done) {
          const final = data.info && data.info.final ? data.info.final : null;
          if (final) {
            setStatus(`Episode finished. score=${final.score.toFixed(3)} success=${final.success >= 1.0}`);
          } else {
            setStatus("Episode finished.");
          }
        } else {
          setStatus(`Step accepted. reward=${data.reward.toFixed(2)}`);
        }
      }

      async function refreshState() {
        const res = await fetch("/state");
        if (!res.ok) {
          setStatus("No active episode yet. Reset first.", true);
          return;
        }
        const data = await res.json();
        setStatus(`Public state refreshed. turn=${data.turn} cumulative_reward=${data.cumulative_reward.toFixed(3)}`);
      }

      taskSelect.addEventListener("change", updateTaskMeta);
      resetBtn.addEventListener("click", resetEpisode);
      stepBtn.addEventListener("click", sendStep);
      stateBtn.addEventListener("click", refreshState);
      loadTasks().catch((err) => setStatus(String(err), true));
    </script>
  </body>
</html>
"""


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

    if not hmac.compare_digest(signature, _sign(payload)):
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


def _root_payload() -> dict:
    return {
        "name": "emotional-support-conversations",
        "version": "0.1.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/ui"],
        "tasks": [t["id"] for t in ESCEnv.list_tasks()],
    }


@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(UI_HTML)
    return _root_payload()


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return UI_HTML


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


def main() -> None:
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
