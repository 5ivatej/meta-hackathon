"""Shared data collection and prompting utilities for training scripts."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any, Dict, List, Sequence

from .agentic import AgentMemory, SkillDecision, SkillRouter, build_default_skills
from .baselines import GenericTemplateBaseline, ValidationOnlyBaseline
from .env import ESCEnv
from .models import Action, Observation
from .tasks import TASKS


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the response generator inside a therapist-style support agent.

    A deterministic controller has already selected the right conversational
    move. Your job is to produce the next response while preserving the memory,
    pacing, and safety needs of the ongoing therapy arc.

    Rules:
    - Stay warm, brief, and human.
    - Preserve continuity with prior sessions.
    - Ask at most one question.
    - Do not jump to advice before trust is built.
    - Keep safety follow-through explicit when the context requires it.
    """
).strip()


@dataclass
class PolicySample:
    task_id: str
    prompt: str
    completion: str
    session_index: int
    turn: int
    source: str


@dataclass
class RewardSample:
    task_id: str
    text: str
    future_oriented: float
    step_reward: float
    session_index: int
    turn: int
    candidate_source: str


@dataclass
class EvalEpisode:
    task_id: str
    score: float
    success: bool
    steps: int
    completion: float
    final_resolution: float
    transcript: List[str]


def build_policy_prompt(
    observation: Observation,
    memory: AgentMemory,
    decision: SkillDecision,
    skill_instruction: str,
) -> str:
    return textwrap.dedent(
        f"""
        {SYSTEM_PROMPT}

        Selected skill: {decision.skill_name}
        Why this skill was selected: {decision.rationale}
        Skill directive: {skill_instruction}

        Scenario: {observation.scenario_brief}
        Public stage hint: {observation.stage_hint}
        Session: {observation.session_index}/{observation.sessions_total}
        Turn: {observation.turn}
        Remaining turns: {observation.remaining_turns}
        Remaining turns in session: {observation.remaining_session_turns}

        Durable memory and recent exchange:
        {memory.prompt_context(observation)}

        Seeker just said:
        "{observation.seeker_utterance}"

        Write the next reply now.
        """
    ).strip()


def build_reward_text(observation: Observation, memory: AgentMemory, candidate_response: str) -> str:
    return textwrap.dedent(
        f"""
        Scenario: {observation.scenario_brief}
        Stage: {observation.stage_hint}
        Session: {observation.session_index}/{observation.sessions_total}
        Turn: {observation.turn}
        Current goal: {observation.current_goal_hint}

        Durable memory and recent exchange:
        {memory.prompt_context(observation)}

        Seeker:
        "{observation.seeker_utterance}"

        Candidate therapist response:
        "{candidate_response}"

        Predict the future-oriented quality of that response.
        """
    ).strip()


def _candidate_skill_outputs(observation: Observation, memory: AgentMemory) -> List[tuple[str, str]]:
    skills = build_default_skills()
    outputs: List[tuple[str, str]] = []
    for skill_name, skill in skills.items():
        candidate_memory = AgentMemory.from_dict(memory.to_dict())
        decision = SkillDecision(skill_name=skill_name, rationale=f"Candidate generated from {skill_name}")
        message = skill.render(observation, candidate_memory, decision)
        outputs.append((skill_name, message))
    return outputs


def _candidate_baseline_outputs(observation: Observation) -> List[tuple[str, str]]:
    generic = GenericTemplateBaseline()
    validation = ValidationOnlyBaseline()
    generic.reset(observation.task_id)
    validation.reset(observation.task_id)
    return [
        ("generic_template", generic.act(observation)),
        ("validation_only", validation.act(observation)),
    ]


def candidate_responses(observation: Observation, memory: AgentMemory) -> List[tuple[str, str]]:
    seen: set[str] = set()
    candidates: List[tuple[str, str]] = []
    for source, message in _candidate_skill_outputs(observation, memory) + _candidate_baseline_outputs(observation):
        key = " ".join(message.lower().split())
        if key in seen:
            continue
        seen.add(key)
        candidates.append((source, message))
    return candidates


def collect_policy_teacher_dataset(episodes_per_task: int) -> List[PolicySample]:
    env = ESCEnv()
    router = SkillRouter()
    skills = build_default_skills()
    samples: List[PolicySample] = []

    for task_id in TASKS:
        for _ in range(episodes_per_task):
            memory = AgentMemory()
            memory.reset(task_id)
            obs = env.reset(task_id=task_id).observation

            while True:
                memory.observe(obs)
                decision = router.choose(obs, memory)
                skill = skills[decision.skill_name]
                message = skill.render(obs, memory, decision)
                prompt = build_policy_prompt(
                    observation=obs,
                    memory=memory,
                    decision=decision,
                    skill_instruction=skill.llm_instruction(obs, memory, decision),
                )
                samples.append(
                    PolicySample(
                        task_id=task_id,
                        prompt=prompt,
                        completion=message,
                        session_index=obs.session_index,
                        turn=obs.turn,
                        source="teacher",
                    )
                )
                memory.remember(decision.skill_name, message)
                result = env.step(Action(message=message))
                obs = result.observation
                if result.done:
                    break

    return samples


def collect_reward_dataset(episodes_per_task: int) -> List[RewardSample]:
    env = ESCEnv()
    router = SkillRouter()
    skills = build_default_skills()
    samples: List[RewardSample] = []

    for task_id in TASKS:
        for _ in range(episodes_per_task):
            memory = AgentMemory()
            memory.reset(task_id)
            obs = env.reset(task_id=task_id).observation

            while True:
                memory.observe(obs)

                for source, candidate_message in candidate_responses(obs, memory):
                    cloned_env = ESCEnv.from_state(env.export_state())
                    reward_result = cloned_env.step(Action(message=candidate_message))
                    reward_detail = reward_result.reward_detail
                    samples.append(
                        RewardSample(
                            task_id=task_id,
                            text=build_reward_text(obs, memory, candidate_message),
                            future_oriented=float(reward_detail.future_oriented),
                            step_reward=float(reward_result.reward),
                            session_index=obs.session_index,
                            turn=obs.turn,
                            candidate_source=source,
                        )
                    )

                decision = router.choose(obs, memory)
                skill = skills[decision.skill_name]
                teacher_message = skill.render(obs, memory, decision)
                memory.remember(decision.skill_name, teacher_message)
                result = env.step(Action(message=teacher_message))
                obs = result.observation
                if result.done:
                    break

    return samples


def summarize_eval(episodes: Sequence[EvalEpisode]) -> Dict[str, Any]:
    return {
        "avg_score": mean(ep.score for ep in episodes) if episodes else 0.0,
        "success_rate": mean(1.0 if ep.success else 0.0 for ep in episodes) if episodes else 0.0,
        "avg_steps": mean(ep.steps for ep in episodes) if episodes else 0.0,
        "episodes": [asdict(ep) for ep in episodes],
    }
