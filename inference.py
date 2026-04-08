"""Baseline inference script for the ESC OpenEnv environment.

MANDATORY env vars
------------------
    API_BASE_URL   - LLM endpoint
    MODEL_NAME     - Model identifier
    HF_TOKEN       - Hugging Face / router token (preferred)
    ESC_ENV_URL    - URL of the running ESC OpenEnv HTTP server (defaults to localhost)

Compatible auth env vars
------------------------
    OPENAI_API_KEY - standard OpenAI-compatible auth key
    API_KEY        - generic OpenAI-compatible auth key

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

from src.agentic import AgentMemory, SkillRouter, build_default_skills
from src.client import ESCHttpClient
from src.models import Action
from src.seeker import extract_features

BENCHMARK = "emotional-support-conversations"
MAX_STEPS = 14  # upper bound; env imposes per-task limits too
TEMPERATURE = 0.6
MAX_TOKENS = 220

TASK_IDS = ["work_stress_venting", "guarded_relationship", "crisis_fragile_trust"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the response generator inside a controlled emotional-support agent.

    A deterministic controller has already selected the correct conversational
    move for this turn and written a draft reply. Your job is only to lightly
    polish that draft while preserving its intent and structure.

    Hard rules:
    - Stay extremely close to the draft.
    - Keep the same stage objective. Do not change exploration into advice or
      advice into exploration.
    - Preserve any explicit safety support mention, validation, and questions
      already present in the draft.
    - Do not add extra questions, extra advice, or new topics.
    - Keep replies warm, brief, and human.
    - If the draft is already strong, repeat it verbatim.

    Reply with ONLY the next message to the seeker.
    """
).strip()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}\n"
            "Set the judging env vars and rerun `python inference.py`."
        )
    return value


def resolve_api_key() -> str:
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing authentication token. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY "
            "before running `python inference.py`."
        )
    return api_key


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
    skill_name: str,
    rationale: str,
    skill_instruction: str,
    draft_reply: str,
) -> str:
    history_block = "\n".join(history[-8:]) if history else "(this is the first turn)"
    return textwrap.dedent(
        f"""
        Scenario: {scenario_brief}
        Conversation stage (public hint): {stage_hint}
        Turn: {turn}   Remaining turns: {remaining}
        Selected skill: {skill_name}
        Why this skill was selected: {rationale}
        Skill directive: {skill_instruction}

        Recent exchange:
        {history_block}

        Seeker just said:
        "{seeker_utterance}"

        Deterministic draft reply:
        "{draft_reply}"

        Lightly polish the draft only if needed. Preserve its goal and
        structure. If unsure, output the draft unchanged.
        """
    ).strip()


def call_llm(client: OpenAI, model_name: str, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=model_name,
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


def _count_questions(text: str) -> int:
    return (text or "").count("?")


def should_accept_rewrite(draft: str, candidate: str) -> bool:
    candidate = (candidate or "").strip()
    if not candidate:
        return False

    draft_features = extract_features(draft)
    candidate_features = extract_features(candidate)

    if candidate_features.dismissive > 0 or candidate_features.bare:
        return False
    if _count_questions(candidate) > 1 or candidate_features.interrogative > 0:
        return False

    # Do not let the rewrite weaken the key stage-driving signals already
    # present in the deterministic draft.
    if draft_features.open_question > 0 and candidate_features.open_question <= 0:
        return False
    if draft_features.validation > 0 and candidate_features.validation <= 0:
        return False
    if draft_features.empathy > 0 and candidate_features.empathy <= 0:
        return False
    if draft_features.advice > 0 and candidate_features.advice <= 0:
        return False
    if draft_features.safety > 0 and candidate_features.safety <= 0:
        return False

    return True


# -------------------------- per-task episode ---------------------------------

async def run_task(
    openai_client: OpenAI,
    env_client: ESCHttpClient,
    model_name: str,
    task_id: str,
) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=model_name)

    router = SkillRouter()
    skills = build_default_skills()
    memory = AgentMemory()
    memory.reset(task_id)

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
            memory.observe(obs)
            decision = router.choose(obs, memory)
            skill = skills[decision.skill_name]
            draft_message = skill.render(obs, memory, decision)
            user_prompt = build_user_prompt(
                scenario_brief=obs.scenario_brief,
                stage_hint=obs.stage_hint,
                turn=obs.turn,
                remaining=obs.remaining_turns,
                seeker_utterance=obs.seeker_utterance,
                history=history,
                skill_name=decision.skill_name,
                rationale=decision.rationale,
                skill_instruction=skill.llm_instruction(obs, memory, decision),
                draft_reply=draft_message,
            )
            candidate_message = call_llm(openai_client, model_name, user_prompt)
            message = candidate_message if should_accept_rewrite(draft_message, candidate_message) else draft_message
            memory.remember(decision.skill_name, message)

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
    api_base_url = require_env("API_BASE_URL")
    model_name = require_env("MODEL_NAME")
    api_key = resolve_api_key()
    env_url = os.getenv("ESC_ENV_URL") or "http://127.0.0.1:7860"

    openai_client = OpenAI(base_url=api_base_url, api_key=api_key)
    env_client = ESCHttpClient.from_url(env_url)

    results = []
    try:
        for task_id in TASK_IDS:
            res = await run_task(openai_client, env_client, model_name, task_id)
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
