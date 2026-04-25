"""Prompt templates for the paper-aligned RLFF-ESC pipeline."""
from __future__ import annotations

from typing import List

from .datasets import SeedExample
from .memory import EpisodeMemory


POLICY_SYSTEM_PROMPT = """
You are the system agent in an emotional support dialogue training pipeline.

Return exactly:
<think>brief internal reasoning about emotional state, trust, and next move</think>
<response>your user-facing reply</response>

Rules:
- Prioritize empathy, validation, and emotional pacing.
- Ask at most one question.
- Avoid advice until the user appears heard and emotionally safe.
- In crisis scenarios, include real-world support when appropriate.
- Keep the visible response warm, concise, and human.
""".strip()


USER_SIM_SYSTEM_PROMPT = """
You are simulating a user in an emotional support conversation.

Respond as the user, not as an evaluator. Stay consistent with the scenario,
emotional state, and memory. Progress may be gradual; do not instantly resolve
the conversation unless the exchange genuinely earns it.

Return strict JSON with keys:
- user_message: string
- completed: boolean
- learned_facts: array of strings
- unresolved_needs: array of strings
""".strip()


CRITIC_SYSTEM_PROMPT = """
You are the critic in an emotional support RL training pipeline.

Evaluate the full dialogue state after a candidate assistant response and any
simulated future rollout. Focus on whether the assistant improved the user's
future emotional trajectory, not only whether the last message sounded nice.

Return strict JSON with keys:
- score: float in [0, 1]
- goal_achieved: boolean
- rationale: string
""".strip()


SUMMARY_SYSTEM_PROMPT = """
You compress emotional-support dialogue state into durable memory.

Return strict JSON with keys:
- summary: string
- seeker_facts: array of strings
- unresolved_needs: array of strings
- commitments: array of strings
- risk_flags: array of strings
""".strip()


def render_transcript(transcript: List[dict[str, str]], max_turns: int | None = None) -> str:
    rows = transcript[-max_turns:] if max_turns else transcript
    lines = [f"{entry['role'].title()}: {entry['content']}" for entry in rows]
    return "\n".join(lines).strip() or "(empty transcript)"


def build_policy_messages(
    seed: SeedExample,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
    max_recent_turns: int,
) -> List[dict[str, str]]:
    transcript_block = render_transcript(transcript, max_recent_turns)
    user_prompt = f"""
Scenario: {seed.scenario_brief}
Desired outcome: {seed.desired_outcome}
Emotion type: {seed.emotion_type or "unknown"}
Problem type: {seed.problem_type or "unknown"}

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


def build_user_sim_messages(
    seed: SeedExample,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
    max_recent_turns: int,
) -> List[dict[str, str]]:
    transcript_block = render_transcript(transcript, max_recent_turns)
    user_prompt = f"""
Scenario: {seed.scenario_brief}
Desired outcome: {seed.desired_outcome}
Emotion type: {seed.emotion_type or "unknown"}
Problem type: {seed.problem_type or "unknown"}

Durable memory:
{memory.render_for_prompt()}

Dialogue so far:
{transcript_block}

Generate the next user turn.
""".strip()
    return [
        {"role": "system", "content": USER_SIM_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_critic_messages(
    seed: SeedExample,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
) -> List[dict[str, str]]:
    user_prompt = f"""
Scenario: {seed.scenario_brief}
Desired outcome: {seed.desired_outcome}
Emotion type: {seed.emotion_type or "unknown"}
Problem type: {seed.problem_type or "unknown"}

Durable memory:
{memory.render_for_prompt()}

Evaluated transcript:
{render_transcript(transcript)}

Judge whether the system response improves the future emotional trajectory of
the dialogue. High scores require emotional attunement, context-sensitive
support, and movement toward the stated goal.
""".strip()
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(
    seed: SeedExample,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
) -> List[dict[str, str]]:
    user_prompt = f"""
Scenario: {seed.scenario_brief}
Current memory:
{memory.render_for_prompt()}

Transcript:
{render_transcript(transcript)}

Update the durable memory state.
""".strip()
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
