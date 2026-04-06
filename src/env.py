"""Core ESC environment: OpenEnv-style step() / reset() / state()."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .grader import GradeBreakdown, final_task_score, grade_step
from .models import (
    Action,
    EnvState,
    Observation,
    ResetResult,
    Reward,
    StepResult,
)
from .seeker import (
    SeekerState,
    Stage,
    extract_features,
    resolution_score,
    step_seeker,
)
from .tasks import TASKS, TaskSpec, get_task


class ESCEnv:
    """Emotional Support Conversations environment.

    Usage (in-process):
        env = ESCEnv()
        obs = env.reset(task_id="work_stress_venting")
        result = env.step(Action(message="That sounds really hard. What's weighing on you most right now?"))
    """

    def __init__(self) -> None:
        self._task: Optional[TaskSpec] = None
        self._seeker: Optional[SeekerState] = None
        self._turn: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._transcript: List[Dict[str, str]] = []
        self._agent_messages: List[str] = []
        self._had_safety_reference: bool = False
        self._last_obs: Optional[Observation] = None

    # ------------------------------------------------------------------ reset

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> ResetResult:
        """Reset to a clean initial state for the given task (default: easy)."""
        task_id = task_id or "work_stress_venting"
        self._task = get_task(task_id)
        self._seeker = SeekerState.from_persona(self._task.persona)
        self._turn = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._transcript = [
            {"role": "seeker", "text": self._task.persona.surface_concern}
        ]
        self._agent_messages = []
        self._had_safety_reference = False

        obs = Observation(
            seeker_utterance=self._task.persona.surface_concern,
            turn=0,
            remaining_turns=self._task.max_turns,
            stage_hint=self._seeker.stage.value,
            task_id=self._task.id,
            scenario_brief=self._task.persona.scenario_brief,
        )
        self._last_obs = obs
        return ResetResult(
            observation=obs,
            info={
                "difficulty": self._task.difficulty,
                "max_turns": self._task.max_turns,
                "success_threshold": self._task.success_threshold,
            },
        )

    # ------------------------------------------------------------------- step

    def step(self, action: Action) -> StepResult:
        if self._task is None or self._seeker is None:
            raise RuntimeError("env.step() called before reset()")
        if self._done:
            raise RuntimeError("env.step() called on a finished episode — call reset()")

        # 1. Record the agent's turn.
        normalized_message = " ".join(action.message.lower().split())
        repetitive = normalized_message in self._agent_messages
        self._transcript.append({"role": "agent", "text": action.message})
        self._agent_messages.append(normalized_message)

        # 2. Snapshot pre-action state (for reward deltas and future-oriented lookahead).
        pre_state = self._seeker.snapshot()

        # 3. Extract features and advance seeker dynamics.
        features = extract_features(action.message)
        if features.safety > 0:
            self._had_safety_reference = True
        transition = step_seeker(self._seeker, features)
        post_state = transition.new_state  # same object, mutated
        self._seeker = post_state
        self._turn += 1
        transition.flags["repetitive"] = repetitive

        # 4. Grade the step.
        breakdown: GradeBreakdown = grade_step(
            pre_state=pre_state,
            post_state=post_state,
            features=features,
            flags=transition.flags,
        )
        self._cumulative_reward += breakdown.value

        # 5. Record seeker's reply.
        self._transcript.append({"role": "seeker", "text": transition.seeker_utterance})

        # 6. Termination check.
        reached_required_stage = post_state.stage.value == self._task.required_final_stage
        met_trust_target = post_state.trust >= self._task.min_final_trust
        met_distress_target = post_state.distress <= self._task.max_final_distress
        revealed_if_required = (not self._task.require_reveal) or post_state.revealed
        safety_if_required = (not self._task.require_safety_reference) or self._had_safety_reference
        natural_done = bool(
            reached_required_stage
            and met_trust_target
            and met_distress_target
            and revealed_if_required
            and safety_if_required
        )
        trust_collapse = post_state.trust <= 0.05
        budget_exhausted = self._turn >= self._task.max_turns
        done = bool(natural_done or trust_collapse or budget_exhausted)
        self._done = done

        # 7. Build the next observation.
        obs = Observation(
            seeker_utterance=transition.seeker_utterance,
            turn=self._turn,
            remaining_turns=max(0, self._task.max_turns - self._turn),
            stage_hint=post_state.stage.value,
            task_id=self._task.id,
            scenario_brief=self._task.persona.scenario_brief,
        )
        self._last_obs = obs

        info: Dict[str, Any] = {
            "features": features.__dict__,
            "flags": transition.flags,
            "stage": post_state.stage.value,
            "resolution_score": resolution_score(post_state),
            "natural_done": natural_done,
            "repetitive": repetitive,
            "had_safety_reference": self._had_safety_reference,
            "meets_trust_target": met_trust_target,
            "meets_distress_target": met_distress_target,
            "revealed_if_required": revealed_if_required,
            "safety_if_required": safety_if_required,
            "trust_collapse": trust_collapse,
            "budget_exhausted": budget_exhausted,
            "reward_components": breakdown.components,
        }

        if done:
            info["final"] = final_task_score(
                cumulative_reward=self._cumulative_reward,
                steps_taken=self._turn,
                max_turns=self._task.max_turns,
                final_state=post_state,
                success_threshold=self._task.success_threshold,
                completed=natural_done,
            )

        reward_detail = Reward(
            value=breakdown.value,
            immediate=breakdown.immediate,
            future_oriented=breakdown.future_oriented,
            penalties=breakdown.penalties,
            components={k: float(v) for k, v in breakdown.components.items()},
        )

        return StepResult(
            observation=obs,
            reward=breakdown.value,
            reward_detail=reward_detail,
            done=done,
            info=info,
        )

    # ------------------------------------------------------------------ state

    def state(self) -> EnvState:
        if self._task is None:
            raise RuntimeError("env.state() called before reset()")
        return EnvState(
            task_id=self._task.id,
            turn=self._turn,
            max_turns=self._task.max_turns,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            transcript=list(self._transcript),
        )

    # ---------------------------------------------------------------- listing

    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        return [
            {
                "id": t.id,
                "difficulty": t.difficulty,
                "max_turns": t.max_turns,
                "success_threshold": t.success_threshold,
                "scenario_brief": t.persona.scenario_brief,
            }
            for t in TASKS.values()
        ]
