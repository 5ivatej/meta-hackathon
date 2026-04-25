"""Stage 1: build Dr via 3-role multi-agent dialogue simulation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .critic import FutureRewardCritic, FutureRewardResult
from .config import EndpointConfig, SimulationConfig
from .datasets import SeedExample, load_seed_examples
from .io import append_jsonl
from .llm import build_client, chat_text, extract_response_text
from .memory import EpisodeMemory
from .prompts import build_policy_messages
from .simulator import UserSimulator


@dataclass
class CandidateRecord:
    example_id: str
    source: str
    task_id: str | None
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
    ground_truth_response: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-role simulation and build candidate reward data.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-source", choices=["tasks", "jsonl", "esconv_hf", "extes_hf", "extes_jsonl"], default="tasks")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-seed-examples", type=int)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--summary-model")
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=220)
    parser.add_argument("--summary-every-n-turns", type=int, default=4)
    parser.add_argument("--max-recent-turns-in-prompt", type=int, default=8)
    parser.add_argument("--critic-completion-threshold", type=float, default=0.8)
    parser.add_argument("--success-turn-bonus", type=float, default=1.0)
    args = parser.parse_args()

    endpoint = EndpointConfig.from_env()
    client = build_client(endpoint)
    config = SimulationConfig(
        policy_model=args.policy_model,
        user_model=args.user_model,
        critic_model=args.critic_model,
        summary_model=args.summary_model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_seed_examples=args.max_seed_examples,
        episodes_per_seed=args.episodes_per_seed,
        num_candidates=args.num_candidates,
        rollout_steps=args.rollout_steps,
        max_turns=args.max_turns,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        summary_every_n_turns=args.summary_every_n_turns,
        max_recent_turns_in_prompt=args.max_recent_turns_in_prompt,
        critic_completion_threshold=args.critic_completion_threshold,
        success_turn_bonus=args.success_turn_bonus,
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
    for seed in seeds:
        for episode_index in range(config.episodes_per_seed):
            _run_episode(
                client=client,
                seed=seed,
                config=config,
                episode_index=episode_index,
                candidate_path=candidate_path,
                trajectory_path=trajectory_path,
            )


def _run_episode(
    client,
    seed: SeedExample,
    config: SimulationConfig,
    episode_index: int,
    candidate_path: Path,
    trajectory_path: Path,
) -> None:
    transcript: List[dict[str, str]] = [dict(turn) for turn in seed.context_turns]
    memory = EpisodeMemory()
    for turn in transcript:
        if turn["role"] == "user":
            memory.note_user_message(turn["content"])
        else:
            memory.note_assistant_message(turn["content"])
    chosen_turns: List[dict[str, object]] = []
    simulator = UserSimulator(client=client, config=config)
    critic = FutureRewardCritic(client=client, config=config)

    for turn_index in range(config.max_turns):
        policy_messages = build_policy_messages(
            seed=seed,
            transcript=transcript,
            memory=memory,
            max_recent_turns=config.max_recent_turns_in_prompt,
        )
        prompt_for_policy = policy_messages[-1]["content"]

        candidate_records: List[CandidateRecord] = []
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
                transcript=transcript,
                memory=memory,
                initial_response=response,
                simulator=simulator,
                critic=critic,
            )
            record = CandidateRecord(
                example_id=seed.example_id,
                source=seed.source,
                task_id=seed.task_id,
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
                ground_truth_response=seed.ground_truth_response,
            )
            candidate_records.append(record)
            append_jsonl(candidate_path, asdict(record))
            if best_record is None or record.scalar_reward > best_record.scalar_reward:
                best_record = record

        if best_record is None:
            raise RuntimeError("No candidate response was produced.")

        transcript.append({"role": "assistant", "content": best_record.response})
        memory.note_assistant_message(best_record.response)

        user_turn = simulator.generate_turn(seed=seed, transcript=transcript, memory=memory)
        transcript.append({"role": "user", "content": user_turn.user_message})
        simulator.apply_turn_to_memory(memory=memory, turn=user_turn)

        if (turn_index + 1) % max(1, config.summary_every_n_turns) == 0:
            simulator.refresh_summary(seed=seed, transcript=transcript, memory=memory)

        chosen_turns.append(
            {
                "turn_index": turn_index,
                "response": best_record.response,
                "reward": best_record.scalar_reward,
                "completed_after_rollout": best_record.completed,
                "terminal_score": best_record.terminal_score,
            }
        )

        if user_turn.completed or best_record.completed:
            break

    append_jsonl(
        trajectory_path,
        {
            "episode_id": f"{seed.example_id}-ep-{episode_index}",
            "example_id": seed.example_id,
            "source": seed.source,
            "task_id": seed.task_id,
            "turns": chosen_turns,
            "final_memory": memory.render_for_prompt(),
            "transcript": transcript,
        },
    )


def _evaluate_candidate(
    client,
    seed: SeedExample,
    config: SimulationConfig,
    transcript: List[dict[str, str]],
    memory: EpisodeMemory,
    initial_response: str,
    simulator: UserSimulator,
    critic: FutureRewardCritic,
) -> FutureRewardResult:
    sim_transcript = [dict(entry) for entry in transcript]
    sim_memory = memory.clone()

    sim_transcript.append({"role": "assistant", "content": initial_response})
    sim_memory.note_assistant_message(initial_response)

    steps_used = 0
    terminal_judgment = critic.evaluate(seed=seed, transcript=sim_transcript, memory=sim_memory)
    if terminal_judgment.goal_achieved:
        return FutureRewardResult(
            reward=critic.compute_future_reward(terminal_judgment.score, steps_used=1),
            terminal_score=terminal_judgment.score,
            goal_achieved=True,
            steps_used=1,
            rationale=terminal_judgment.rationale,
        )

    for rollout_step in range(1, config.rollout_steps + 1):
        user_turn = simulator.generate_turn(seed=seed, transcript=sim_transcript, memory=sim_memory)
        sim_transcript.append({"role": "user", "content": user_turn.user_message})
        simulator.apply_turn_to_memory(memory=sim_memory, turn=user_turn)
        steps_used = rollout_step

        terminal_judgment = critic.evaluate(seed=seed, transcript=sim_transcript, memory=sim_memory)
        if user_turn.completed or terminal_judgment.goal_achieved or rollout_step == config.rollout_steps:
            reward = critic.compute_future_reward(terminal_judgment.score, steps_used=rollout_step + 1)
            return FutureRewardResult(
                reward=reward,
                terminal_score=terminal_judgment.score,
                goal_achieved=user_turn.completed or terminal_judgment.goal_achieved,
                steps_used=rollout_step + 1,
                rationale=terminal_judgment.rationale,
            )

        rollout_response = chat_text(
            client=client,
            model=config.policy_model,
            messages=build_policy_messages(
                seed=seed,
                transcript=sim_transcript,
                memory=sim_memory,
                max_recent_turns=config.max_recent_turns_in_prompt,
            ),
            temperature=config.temperature,
            max_tokens=config.max_completion_tokens,
        )
        _, rollout_text = extract_response_text(rollout_response)
        sim_transcript.append({"role": "assistant", "content": rollout_text})
        sim_memory.note_assistant_message(rollout_text)

    reward = critic.compute_future_reward(terminal_judgment.score, steps_used=max(1, steps_used))
    return FutureRewardResult(
        reward=reward,
        terminal_score=terminal_judgment.score,
        goal_achieved=terminal_judgment.goal_achieved,
        steps_used=max(1, steps_used),
        rationale=terminal_judgment.rationale,
    )
