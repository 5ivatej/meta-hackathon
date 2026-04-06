"""Baseline inference script for the ESC OpenEnv environment.

MANDATORY env vars
------------------
    API_BASE_URL   - LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     - Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       - API key for the inference endpoint
    ESC_ENV_URL    - URL of the running ESC OpenEnv HTTP server (e.g. the HF Space URL)

STDOUT contract (strict)
------------------------
One [START] line per episode, one [STEP] per step, one [END] per episode.
See the hackathon spec for exact format.

Runs all 3 tasks (easy/medium/hard) sequentially and prints a final summary
to stderr. Total wall-clock budget kept well under 20min on 2 vCPU / 8GB.
"""
from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import traceback
from typing import List, Optional

from openai import OpenAI

from src.client import ESCHttpClient
from src.models import Action

# -------------------------- mandated env vars --------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "http://10.11.7.65:11434/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "qwen2.5:7b-instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "ollama"
ESC_ENV_URL = os.getenv("ESC_ENV_URL") or "http://localhost:7860"

BENCHMARK = "emotional-support-conversations"
MAX_STEPS = 14  # upper bound; env imposes per-task limits too
TEMPERATURE = 0.6
MAX_TOKENS = 220

TASK_IDS = ["work_stress_venting", "guarded_relationship", "crisis_fragile_trust"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an emotionally attuned peer supporter chatting with someone who is
    going through a hard time. Your job is NOT to fix their problem. Your job
    is to make them feel heard, safe, and understood first — and only move
    toward gentle exploration or light action once trust is established.

    Principles:
    - Lead with empathy and validation. Reflect what you hear.
    - Do NOT give advice until the person has clearly shared what's really
      going on and feels heard.
    - Ask at most one open-ended question per reply. Never interrogate.
    - Never be dismissive, minimising, or instructive in a judgmental tone.
    - Keep replies warm, brief (1-3 sentences), and human.
    - In high-distress / crisis scenarios, gently reference professional
      support (a therapist, crisis line) only after rapport is built.

    You will receive the current conversation state. Reply with ONLY your
    next message to the person — no role labels, no prefixes, no quotes.
    """
).strip()


# -------------------------- stdout contract ----------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    # collapse any newlines in the action so the stdout contract stays single-line
    flat_action = " ".join((action or "").split())
    print(
        f"[STEP] step={step} action={flat_action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# -------------------------- LLM call -----------------------------------------

def build_user_prompt(
    scenario_brief: str,
    stage_hint: str,
    turn: int,
    remaining: int,
    seeker_utterance: str,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-8:]) if history else "(this is the first turn)"
    return textwrap.dedent(
        f"""
        Scenario: {scenario_brief}
        Conversation stage (public hint): {stage_hint}
        Turn: {turn}   Remaining turns: {remaining}

        Recent exchange:
        {history_block}

        Seeker just said:
        "{seeker_utterance}"

        Write your next reply (1-3 sentences, warm, no advice unless rapport is clearly established):
        """
    ).strip()


def call_llm(client: OpenAI, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "I hear you. That sounds really hard — can you tell me a little more about what's weighing on you?"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return "That sounds really hard. I'm here — do you want to tell me more about what's going on?"


# -------------------------- per-task episode ---------------------------------

async def run_task(openai_client: OpenAI, env_client: ESCHttpClient, task_id: str) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []
    last_error: Optional[str] = None

    try:
        reset = await env_client.reset(task_id=task_id)
        obs = reset.observation
        history.append(f"Seeker: {obs.seeker_utterance!r}")

        for step in range(1, MAX_STEPS + 1):
            user_prompt = build_user_prompt(
                scenario_brief=obs.scenario_brief,
                stage_hint=obs.stage_hint,
                turn=obs.turn,
                remaining=obs.remaining_turns,
                seeker_utterance=obs.seeker_utterance,
                history=history,
            )
            message = call_llm(openai_client, user_prompt)

            try:
                result = await env_client.step(Action(message=message))
            except Exception as e:
                last_error = f"step_failed: {e}"
                log_step(step=step, action=message, reward=0.0, done=True, error=last_error)
                break

            reward = float(result.reward)
            done = bool(result.done)
            rewards.append(reward)
            steps_taken = step
            obs = result.observation

            history.append(f"Agent: {message!r}")
            history.append(f"Seeker: {obs.seeker_utterance!r}")

            log_step(step=step, action=message, reward=reward, done=done, error=None)

            if done:
                final = result.info.get("final", {}) if isinstance(result.info, dict) else {}
                score = float(final.get("score", sum(rewards) / max(1, steps_taken)))
                success = bool(final.get("success", 0.0) >= 1.0)
                break
        else:
            # Ran out of outer loop without env-side done — fall back to state().
            st = await env_client.state()
            score = float(st.get("cumulative_reward", 0.0)) / max(1, steps_taken)
            success = score >= 0.5

    except Exception as exc:
        last_error = f"episode_failed: {exc}"
        traceback.print_exc(file=sys.stderr)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# -------------------------- main ---------------------------------------------

async def main() -> None:
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    env_client = ESCHttpClient.from_url(ESC_ENV_URL)

    results = []
    try:
        for task_id in TASK_IDS:
            res = await run_task(openai_client, env_client, task_id)
            results.append(res)
    finally:
        await env_client.close()

    # Summary to stderr so it doesn't pollute the stdout contract.
    print("\n=== Baseline summary ===", file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:<26} score={r['score']:.3f}  success={r['success']}  steps={r['steps']}",
            file=sys.stderr,
        )
    avg = sum(r["score"] for r in results) / max(1, len(results))
    print(f"  {'AVERAGE':<26} score={avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
