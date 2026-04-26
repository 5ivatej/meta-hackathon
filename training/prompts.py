"""Prompt templates for the environment-aligned RLFF-ESC pipeline."""
from __future__ import annotations

from typing import List

from .datasets import SeedExample
from .memory import EpisodeMemory


POLICY_SYSTEM_PROMPT = """
You are the policy model in a long-horizon emotional support training loop.

Return exactly:
<think>brief reasoning about the user's emotional state, the current therapy
session, and the best next move</think>
<response>your user-facing reply</response>

Guidelines:
- Be emotionally attuned and context-sensitive.
- Use the environment state, continuity signals, and memory.
- Do not jump to advice before the user is ready.
- In safety-sensitive situations, stay calm and connect the user to real-world support.
- Keep the visible response warm, concise, and natural.
""".strip()


CRITIC_SYSTEM_PROMPT = """
You are the critic in a future-oriented emotional support RL pipeline.

You will see the simulated trajectory plus the environment's long-horizon
state summary. Score whether the candidate response improved the user's future
trajectory across trust, disclosure, continuity, safety, and movement toward a
stable close.

Return strict JSON with keys:
- score: float in [0, 1]
- goal_achieved: boolean
- rationale: string
""".strip()


def render_transcript(transcript: List[dict[str, str]], max_turns: int | None = None) -> str:
    rows = transcript[-max_turns:] if max_turns else transcript
    lines = [f"{entry['role'].title()}: {entry['content']}" for entry in rows]
    return "\n".join(lines).strip() or "(empty transcript)"


def _render_seed_context(seed: SeedExample) -> str:
    if not seed.context_turns:
        return "(no seed dialogue prefix)"
    return render_transcript(seed.context_turns, max_turns=6)


def _render_env_snapshot(env_state: dict, observation: dict | None = None) -> str:
    seeker = env_state.get("seeker") or {}
    info_lines = [
        f"Task: {env_state.get('task_id', 'unknown')}",
        f"Session: {env_state.get('session_index', 1)}/{env_state.get('sessions_total', 1)}",
        f"Turn: {env_state.get('turn', 0)}",
        f"Current goal hint: {env_state.get('current_goal_hint', '') or '(none)'}",
        f"Last session outcome: {env_state.get('last_session_outcome', '') or '(none)'}",
        f"Memory summary: {env_state.get('memory_summary', '') or '(none)'}",
        f"Unfinished threads: {', '.join(env_state.get('unfinished_threads', []) or []) or '(none)'}",
        f"Budget spent: {float(env_state.get('episode_budget_spent', 0.0)):.2f}",
        f"Time spent: {float(env_state.get('episode_time_spent', 0.0)):.2f}",
        f"Trust: {float(seeker.get('trust', 0.0)):.2f}",
        f"Distress: {float(seeker.get('distress', 0.0)):.2f}",
        f"Openness: {float(seeker.get('openness', 0.0)):.2f}",
        f"Revealed core issue: {bool(seeker.get('revealed', False))}",
        f"Stage: {seeker.get('stage', 'unknown')}",
    ]
    if observation:
        info_lines.extend(
            [
                f"Stage hint shown to policy: {observation.get('stage_hint', 'unknown')}",
                f"Remaining episode turns: {observation.get('remaining_turns', 0)}",
                f"Remaining session turns: {observation.get('remaining_session_turns', 0)}",
            ]
        )
    return "\n".join(info_lines)


def build_policy_messages(
    seed: SeedExample,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
    env_state: dict,
    max_recent_turns: int,
) -> List[dict[str, str]]:
    transcript_block = render_transcript(transcript, max_recent_turns)
    observation = env_state.get("last_observation") or {}
    user_prompt = f"""
Scenario: {seed.scenario_brief}
Desired outcome: {seed.desired_outcome}
Emotion type: {seed.emotion_type or "unknown"}
Problem type: {seed.problem_type or "unknown"}

Reference seed dialogue:
{_render_seed_context(seed)}

Environment state:
{_render_env_snapshot(env_state, observation=observation)}

Durable memory:
{memory.render_for_prompt()}

Recent transcript:
{transcript_block}

Write the next assistant turn now.
""".strip()
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_critic_messages(
    seed: SeedExample,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
    env_state: dict,
    rollout_step_rewards: List[float],
    final_info: dict,
) -> List[dict[str, str]]:
    user_prompt = f"""
Scenario: {seed.scenario_brief}
Desired outcome: {seed.desired_outcome}
Emotion type: {seed.emotion_type or "unknown"}
Problem type: {seed.problem_type or "unknown"}

Reference seed dialogue:
{_render_seed_context(seed)}

Durable memory:
{memory.render_for_prompt()}

Evaluated transcript:
{render_transcript(transcript)}

Environment trajectory summary:
{_render_env_snapshot(env_state, observation=env_state.get("last_observation") or {})}
Rollout step rewards: {', '.join(f'{reward:.3f}' for reward in rollout_step_rewards) or '(none)'}
Environment completion signals: {final_info.get('final') or '(not terminal yet)'}
Other rollout info: had_safety_reference={final_info.get('had_safety_reference', False)}, natural_done={final_info.get('natural_done', False)}

Judge whether the candidate response improved the future emotional trajectory
under the environment's semantics. High scores require emotional attunement,
progress toward disclosure or stabilization when needed, continuity across
sessions, budget-aware pacing, and correct safety follow-through.
""".strip()
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
