"""Stage 1: build Dr via environment-aligned future rollouts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .critic import FutureRewardCritic, FutureRewardResult
from .config import EndpointConfig, SimulationConfig
from .datasets import SeedExample, infer_task_id_for_seed, load_seed_examples
from .io import append_jsonl
from .llm import build_client, chat_text, extract_response_text
from .memory import EpisodeMemory
from .prompts import build_policy_messages
from .simulator import EnvironmentSimulator


@dataclass
class CandidateRecord:
    example_id: str
    source: str
    task_id: str | None
    env_task_id: str
    emotion_type: str | None
    problem_type: str | None
    turn_index: int
    candidate_index: int
    prompt_for_policy: str
    policy_think: str
    response: str
    scalar_reward: float
    terminal_score: float
    completed: bool
    rollout_steps_used: int
    rationale: str
    env_final_score: float
    env_final_success: float
    env_cumulative_reward: float
    env_stage: str
    ground_truth_response: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run environment-aligned simulation and build candidate reward data.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-source", choices=["tasks", "jsonl", "esconv_hf", "extes_hf", "extes_jsonl"], default="tasks")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-seed-examples", type=int)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rollout-temperature", type=float, default=0.3)
    parser.add_argument("--max-completion-tokens", type=int, default=220)
    parser.add_argument("--max-recent-turns-in-prompt", type=int, default=8)
    args = parser.parse_args()

    endpoint = EndpointConfig.from_env()
    client = build_client(endpoint)
    config = SimulationConfig(
        policy_model=args.policy_model,
        critic_model=args.critic_model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_seed_examples=args.max_seed_examples,
        episodes_per_seed=args.episodes_per_seed,
        num_candidates=args.num_candidates,
        rollout_steps=args.rollout_steps,
        max_turns=args.max_turns,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        rollout_temperature=args.rollout_temperature,
        max_recent_turns_in_prompt=args.max_recent_turns_in_prompt,
    )

    output_dir = Path(args.output_dir)
    candidate_path = output_dir / "candidate_rewards.jsonl"
    trajectory_path = output_dir / "trajectories.jsonl"
    if candidate_path.exists():
        candidate_path.unlink()
    if trajectory_path.exists():
        trajectory_path.unlink()

    seeds = load_seed_examples(
        source=args.examples_source,
        input_jsonl=args.input_jsonl,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_examples=args.max_seed_examples,
    )
    critic = FutureRewardCritic(client=client, config=config)
    for seed in seeds:
        for episode_index in range(config.episodes_per_seed):
            _run_episode(
                client=client,
                seed=seed,
                config=config,
                critic=critic,
                episode_index=episode_index,
                candidate_path=candidate_path,
                trajectory_path=trajectory_path,
            )


def _run_episode(
    client,
    seed: SeedExample,
    config: SimulationConfig,
    critic: FutureRewardCritic,
    episode_index: int,
    candidate_path: Path,
    trajectory_path: Path,
) -> None:
    env_task_id = infer_task_id_for_seed(seed)
    simulator = EnvironmentSimulator(task_id=env_task_id)
    memory = EpisodeMemory()
    for turn in seed.context_turns:
        if turn["role"] == "user":
            memory.note_user_message(turn["content"])
        else:
            memory.note_assistant_message(turn["content"])
    memory.sync_from_env(simulator.export_state())
    memory.refresh_summary()

    chosen_turns: List[dict[str, object]] = []
    for turn_index in range(config.max_turns):
        env_state = simulator.export_state()
        transcript = simulator.transcript_for_chat()
        policy_messages = build_policy_messages(
            seed=seed,
            transcript=transcript,
            memory=memory,
            env_state=env_state,
            max_recent_turns=config.max_recent_turns_in_prompt,
        )
        prompt_for_policy = policy_messages[-1]["content"]

        best_record: CandidateRecord | None = None
        for candidate_index in range(config.num_candidates):
            raw_candidate = chat_text(
                client=client,
                model=config.policy_model,
                messages=policy_messages,
                temperature=config.temperature,
                max_tokens=config.max_completion_tokens,
            )
            policy_think, response = extract_response_text(raw_candidate)
            reward_result = _evaluate_candidate(
                client=client,
                seed=seed,
                config=config,
                memory=memory,
                env_state=env_state,
                initial_response=response,
                critic=critic,
            )
            record = CandidateRecord(
                example_id=seed.example_id,
                source=seed.source,
                task_id=seed.task_id,
                env_task_id=env_task_id,
                emotion_type=seed.emotion_type,
                problem_type=seed.problem_type,
                turn_index=turn_index,
                candidate_index=candidate_index,
                prompt_for_policy=prompt_for_policy,
                policy_think=policy_think,
                response=response,
                scalar_reward=reward_result.reward,
                terminal_score=reward_result.terminal_score,
                completed=reward_result.goal_achieved,
                rollout_steps_used=reward_result.steps_used,
                rationale=reward_result.rationale,
                env_final_score=reward_result.env_final_score,
                env_final_success=reward_result.env_final_success,
                env_cumulative_reward=reward_result.env_cumulative_reward,
                env_stage=reward_result.env_stage,
                ground_truth_response=seed.ground_truth_response,
            )
            append_jsonl(candidate_path, asdict(record))
            if best_record is None or record.scalar_reward > best_record.scalar_reward:
                best_record = record

        if best_record is None:
            raise RuntimeError("No candidate response was produced.")

        live_step = simulator.step_assistant(best_record.response)
        memory.note_assistant_message(best_record.response)
        if live_step.user_message:
            memory.note_user_message(live_step.user_message)
        memory.sync_from_env(live_step.env_state, live_step.info)
        memory.refresh_summary()

        chosen_turns.append(
            {
                "turn_index": turn_index,
                "response": best_record.response,
                "reward": best_record.scalar_reward,
                "completed_after_rollout": best_record.completed,
                "terminal_score": best_record.terminal_score,
                "env_final_score": best_record.env_final_score,
                "env_final_success": best_record.env_final_success,
                "env_stage": best_record.env_stage,
            }
        )

        if live_step.done:
            break

    append_jsonl(
        trajectory_path,
        {
            "episode_id": f"{seed.example_id}-ep-{episode_index}",
            "example_id": seed.example_id,
            "source": seed.source,
            "task_id": seed.task_id,
            "env_task_id": env_task_id,
            "turns": chosen_turns,
            "final_memory": memory.render_for_prompt(),
            "env_state": simulator.export_state(),
            "transcript": simulator.transcript_for_chat(),
        },
    )


def _evaluate_candidate(
    client,
    seed: SeedExample,
    config: SimulationConfig,
    memory: EpisodeMemory,
    env_state: dict,
    initial_response: str,
    critic: FutureRewardCritic,
) -> FutureRewardResult:
    simulator = EnvironmentSimulator.from_exported_state(env_state)
    sim_memory = memory.clone()
    rollout_rewards: List[float] = []

    final_transition = simulator.step_assistant(initial_response)
    rollout_rewards.append(final_transition.reward)
    sim_memory.note_assistant_message(initial_response)
    if final_transition.user_message:
        sim_memory.note_user_message(final_transition.user_message)
    sim_memory.sync_from_env(final_transition.env_state, final_transition.info)
    sim_memory.refresh_summary()

    for _ in range(config.rollout_steps):
        if final_transition.done:
            break
        rollout_messages = build_policy_messages(
            seed=seed,
            transcript=simulator.transcript_for_chat(),
            memory=sim_memory,
            env_state=simulator.export_state(),
            max_recent_turns=config.max_recent_turns_in_prompt,
        )
        raw_rollout = chat_text(
            client=client,
            model=config.policy_model,
            messages=rollout_messages,
            temperature=config.rollout_temperature,
            max_tokens=config.max_completion_tokens,
        )
        _, rollout_response = extract_response_text(raw_rollout)
        final_transition = simulator.step_assistant(rollout_response)
        rollout_rewards.append(final_transition.reward)
        sim_memory.note_assistant_message(rollout_response)
        if final_transition.user_message:
            sim_memory.note_user_message(final_transition.user_message)
        sim_memory.sync_from_env(final_transition.env_state, final_transition.info)
        sim_memory.refresh_summary()

    terminal_judgment = critic.evaluate(
        seed=seed,
        transcript=simulator.transcript_for_chat(),
        memory=sim_memory,
        env_state=simulator.export_state(),
        rollout_step_rewards=rollout_rewards,
        final_info=final_transition.info,
    )
    final_score = 0.0
    final_success = 0.0
    if isinstance(final_transition.info.get("final"), dict):
        final_score = float(final_transition.info["final"].get("score", 0.0))
        final_success = float(final_transition.info["final"].get("success", 0.0))

    return FutureRewardResult(
        reward=critic.compute_future_reward(terminal_judgment.score),
        terminal_score=terminal_judgment.score,
        goal_achieved=terminal_judgment.goal_achieved,
        steps_used=len(rollout_rewards),
        rationale=terminal_judgment.rationale,
        env_final_score=final_score,
        env_final_success=final_success,
        env_cumulative_reward=float(simulator.export_state().get("cumulative_reward", 0.0)),
        env_stage=str((simulator.export_state().get("seeker") or {}).get("stage", "unknown")),
    )


if __name__ == "__main__":
    main()
