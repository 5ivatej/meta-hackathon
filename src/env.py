"""Core ESC environment with multi-session long-horizon extensions."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from .grader import GradeBreakdown, final_task_score, grade_step
from .models import Action, EnvState, Observation, ResetResult, Reward, StepResult
from .seeker import SeekerState, Stage, extract_features, resolution_score, step_seeker
from .tasks import TASKS, TaskSpec, get_task


def _clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _keyword_tokens(texts: List[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower()):
            tokens.add(token)
    return tokens


class ESCEnv:
    """Emotional Support Conversations environment."""

    def __init__(self) -> None:
        self._task: Optional[TaskSpec] = None
        self._seeker: Optional[SeekerState] = None
        self._turn: int = 0
        self._session_turn: int = 0
        self._session_index: int = 1
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._transcript: List[Dict[str, str]] = []
        self._agent_messages: List[str] = []
        self._had_safety_reference: bool = False
        self._last_obs: Optional[Observation] = None
        self._alliance_strength: float = 0.0
        self._stability: float = 0.0
        self._adherence: float = 0.0
        self._rupture_count: int = 0
        self._safety_risk_level: float = 0.0
        self._working_goals: List[str] = []
        self._active_coping_plan: str = ""
        self._memory_summary: str = ""
        self._last_session_outcome: str = ""
        self._current_goal_hint: str = ""
        self._unfinished_threads: List[str] = []
        self._recent_breakthrough: str = ""
        self._episode_budget_spent: float = 0.0
        self._episode_time_spent: float = 0.0
        self._resume_checkpoint_id: str = ""
        self._continuity_score: float = 0.0
        self._resume_count: int = 0

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> ResetResult:
        del seed
        task_id = task_id or "work_stress_venting"
        self._task = get_task(task_id)
        self._seeker = SeekerState.from_persona(self._task.persona)
        self._turn = 0
        self._session_turn = 0
        self._session_index = 1
        self._done = False
        self._cumulative_reward = 0.0
        self._transcript = [{"role": "seeker", "text": self._task.persona.surface_concern}]
        self._agent_messages = []
        self._had_safety_reference = False
        self._alliance_strength = self._seeker.trust
        self._stability = 1.0 - self._seeker.distress
        self._adherence = 0.0
        self._rupture_count = 0
        self._safety_risk_level = self._initial_safety_risk()
        self._working_goals = list(self._task.working_goals)
        self._active_coping_plan = ""
        self._memory_summary = ""
        self._last_session_outcome = ""
        self._unfinished_threads = []
        self._recent_breakthrough = ""
        self._episode_budget_spent = 0.0
        self._episode_time_spent = 0.0
        self._continuity_score = 0.0
        self._resume_count = 0
        self._refresh_goal_hint()
        self._resume_checkpoint_id = self._make_checkpoint_id()
        obs = self._build_observation(self._task.persona.surface_concern)
        self._last_obs = obs
        return ResetResult(
            observation=obs,
            info={
                "difficulty": self._task.difficulty,
                "max_turns": self._task.max_turns,
                "success_threshold": self._task.success_threshold,
                "sessions_total": self._task.sessions_total,
                "session_turn_limit": self._task.session_turn_limit,
                "cost_budget": self._task.cost_budget,
                "time_budget": self._task.time_budget,
            },
        )

    def step(self, action: Action) -> StepResult:
        if self._task is None or self._seeker is None:
            raise RuntimeError("env.step() called before reset()")
        if self._done:
            raise RuntimeError("env.step() called on a finished episode — call reset()")

        normalized_message = _normalize(action.message)
        repetitive = normalized_message in self._agent_messages[-4:]
        self._transcript.append({"role": "agent", "text": action.message})
        self._agent_messages.append(normalized_message)

        pre_state = self._seeker.snapshot()
        features = extract_features(action.message)
        if features.safety > 0:
            self._had_safety_reference = True
        transition = step_seeker(self._seeker, features)
        post_state = transition.new_state
        self._seeker = post_state
        self._turn += 1
        self._session_turn += 1
        transition.flags["repetitive"] = repetitive

        self._episode_budget_spent += self._estimate_cost(action.message)
        self._episode_time_spent += 1.0
        self._update_long_horizon_state(post_state, transition.flags, action.message)

        long_horizon_context = self._build_long_horizon_context(action.message, repetitive)
        self._continuity_score = _clip(
            0.8 * self._continuity_score + 0.2 * max(long_horizon_context.get("continuity_hit", 0.0), long_horizon_context.get("goal_hit", 0.0))
        )
        breakdown: GradeBreakdown = grade_step(
            pre_state=pre_state,
            post_state=post_state,
            features=features,
            flags=transition.flags,
            long_horizon_context=long_horizon_context,
        )
        step_reward = breakdown.value
        info: Dict[str, Any] = {
            "features": features.__dict__,
            "flags": transition.flags,
            "stage": post_state.stage.value,
            "resolution_score": resolution_score(post_state),
            "repetitive": repetitive,
            "had_safety_reference": self._had_safety_reference,
            "reward_components": breakdown.components,
            "session_index": self._session_index,
            "sessions_total": self._task.sessions_total,
            "current_goal_hint": self._current_goal_hint,
            "memory_summary": self._memory_summary,
            "episode_budget_spent": self._episode_budget_spent,
            "episode_time_spent": self._episode_time_spent,
        }

        reached_required_stage = post_state.stage.value == self._task.required_final_stage
        met_trust_target = post_state.trust >= self._task.min_final_trust
        met_distress_target = post_state.distress <= self._task.max_final_distress
        revealed_if_required = (not self._task.require_reveal) or post_state.revealed
        safety_if_required = (not self._task.require_safety_reference) or self._had_safety_reference
        completed_conditions = bool(
            reached_required_stage
            and met_trust_target
            and met_distress_target
            and revealed_if_required
            and safety_if_required
        )

        trust_collapse = post_state.trust <= 0.05
        hard_budget_exhausted = self._turn >= self._task.max_turns
        budget_exhausted = self._episode_budget_spent >= self._task.cost_budget
        time_exhausted = self._episode_time_spent >= self._task.time_budget
        final_session = self._session_index >= self._task.sessions_total
        done = bool((final_session and completed_conditions) or trust_collapse or hard_budget_exhausted or budget_exhausted or time_exhausted)

        next_utterance = transition.seeker_utterance
        session_transition_bonus = 0.0
        self._transcript.append({"role": "seeker", "text": transition.seeker_utterance})
        if not done:
            if self._session_turn >= self._task.session_turn_limit and not final_session:
                next_utterance, session_transition_bonus = self._advance_session()
                step_reward = _clip(step_reward + session_transition_bonus)
                breakdown.components["session_transition_bonus"] = session_transition_bonus
                info["session_transition"] = True
                info["session_index"] = self._session_index

        self._cumulative_reward += step_reward
        self._done = done

        obs = self._build_observation(next_utterance)
        self._last_obs = obs
        info.update(
            {
                "natural_done": final_session and completed_conditions,
                "meets_trust_target": met_trust_target,
                "meets_distress_target": met_distress_target,
                "revealed_if_required": revealed_if_required,
                "safety_if_required": safety_if_required,
                "trust_collapse": trust_collapse,
                "budget_exhausted": hard_budget_exhausted or budget_exhausted or time_exhausted,
            }
        )

        if done:
            info["final"] = final_task_score(
                cumulative_reward=self._cumulative_reward,
                steps_taken=self._turn,
                max_turns=self._task.max_turns,
                final_state=post_state,
                success_threshold=self._task.success_threshold,
                completed=completed_conditions,
                alliance_strength=self._alliance_strength,
                stability=self._stability,
                adherence=self._adherence,
                continuity=self._continuity_score,
                budget_ratio=self._budget_ratio(),
            )

        reward_detail = Reward(
            value=step_reward,
            immediate=breakdown.immediate,
            future_oriented=breakdown.future_oriented,
            penalties=breakdown.penalties,
            components={k: float(v) for k, v in breakdown.components.items()},
        )
        return StepResult(
            observation=obs,
            reward=step_reward,
            reward_detail=reward_detail,
            done=done,
            info=info,
        )

    def state(self) -> EnvState:
        if self._task is None:
            raise RuntimeError("env.state() called before reset()")
        return EnvState(
            task_id=self._task.id,
            turn=self._turn,
            max_turns=self._task.max_turns,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            session_index=self._session_index,
            sessions_total=self._task.sessions_total,
            remaining_session_turns=max(0, self._task.session_turn_limit - self._session_turn),
            memory_summary=self._memory_summary,
            current_goal_hint=self._current_goal_hint,
            last_session_outcome=self._last_session_outcome,
            episode_budget_spent=self._episode_budget_spent,
            episode_budget_limit=self._task.cost_budget,
            episode_time_spent=self._episode_time_spent,
            episode_time_limit=self._task.time_budget,
            transcript=list(self._transcript),
        )

    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        return [
            {
                "id": t.id,
                "difficulty": t.difficulty,
                "max_turns": t.max_turns,
                "sessions_total": t.sessions_total,
                "session_turn_limit": t.session_turn_limit,
                "cost_budget": t.cost_budget,
                "time_budget": t.time_budget,
                "success_threshold": t.success_threshold,
                "scenario_brief": t.persona.scenario_brief,
            }
            for t in TASKS.values()
        ]

    def export_state(self) -> Dict[str, Any]:
        if self._task is None or self._seeker is None:
            raise RuntimeError("env.export_state() called before reset()")
        seeker_state = {
            "distress": self._seeker.distress,
            "trust": self._seeker.trust,
            "openness": self._seeker.openness,
            "revealed": self._seeker.revealed,
            "stage": self._seeker.stage.value,
            "last_line_idx_by_stage": {
                stage.value: idx for stage, idx in self._seeker.last_line_idx_by_stage.items()
            },
            "turn": self._seeker.turn,
        }
        return {
            "task_id": self._task.id,
            "turn": self._turn,
            "session_turn": self._session_turn,
            "session_index": self._session_index,
            "sessions_total": self._task.sessions_total,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "transcript": list(self._transcript),
            "agent_messages": list(self._agent_messages),
            "had_safety_reference": self._had_safety_reference,
            "seeker": seeker_state,
            "alliance_strength": self._alliance_strength,
            "stability": self._stability,
            "adherence": self._adherence,
            "rupture_count": self._rupture_count,
            "safety_risk_level": self._safety_risk_level,
            "working_goals": list(self._working_goals),
            "active_coping_plan": self._active_coping_plan,
            "memory_summary": self._memory_summary,
            "last_session_outcome": self._last_session_outcome,
            "current_goal_hint": self._current_goal_hint,
            "unfinished_threads": list(self._unfinished_threads),
            "recent_breakthrough": self._recent_breakthrough,
            "episode_budget_spent": self._episode_budget_spent,
            "episode_budget_limit": self._task.cost_budget,
            "episode_time_spent": self._episode_time_spent,
            "episode_time_limit": self._task.time_budget,
            "resume_checkpoint_id": self._resume_checkpoint_id,
            "continuity_score": self._continuity_score,
            "resume_count": self._resume_count,
            "last_observation": self._last_obs.model_dump() if self._last_obs is not None else None,
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "ESCEnv":
        task = get_task(str(data["task_id"]))
        seeker_data = data["seeker"]

        env = cls()
        env._task = task
        env._turn = int(data["turn"])
        env._session_turn = int(data.get("session_turn", 0))
        env._session_index = int(data.get("session_index", 1))
        env._done = bool(data["done"])
        env._cumulative_reward = float(data["cumulative_reward"])
        env._transcript = list(data.get("transcript", []))
        env._agent_messages = list(data.get("agent_messages", []))
        env._had_safety_reference = bool(data.get("had_safety_reference", False))
        env._alliance_strength = float(data.get("alliance_strength", 0.0))
        env._stability = float(data.get("stability", 0.0))
        env._adherence = float(data.get("adherence", 0.0))
        env._rupture_count = int(data.get("rupture_count", 0))
        env._safety_risk_level = float(data.get("safety_risk_level", 0.0))
        env._working_goals = list(data.get("working_goals", []))
        env._active_coping_plan = str(data.get("active_coping_plan", ""))
        env._memory_summary = str(data.get("memory_summary", ""))
        env._last_session_outcome = str(data.get("last_session_outcome", ""))
        env._current_goal_hint = str(data.get("current_goal_hint", ""))
        env._unfinished_threads = list(data.get("unfinished_threads", []))
        env._recent_breakthrough = str(data.get("recent_breakthrough", ""))
        env._episode_budget_spent = float(data.get("episode_budget_spent", 0.0))
        env._episode_time_spent = float(data.get("episode_time_spent", 0.0))
        env._resume_checkpoint_id = str(data.get("resume_checkpoint_id", ""))
        env._continuity_score = float(data.get("continuity_score", 0.0))
        env._resume_count = int(data.get("resume_count", 0)) + 1
        env._seeker = SeekerState(
            persona=task.persona,
            distress=float(seeker_data["distress"]),
            trust=float(seeker_data["trust"]),
            openness=float(seeker_data["openness"]),
            revealed=bool(seeker_data["revealed"]),
            stage=Stage(str(seeker_data["stage"])),
            last_line_idx_by_stage={
                Stage(stage_name): int(idx)
                for stage_name, idx in seeker_data["last_line_idx_by_stage"].items()
            },
            turn=int(seeker_data["turn"]),
        )
        if data.get("last_observation"):
            env._last_obs = Observation(**data["last_observation"])
        elif env._transcript:
            last_seeker_text = next(
                (entry["text"] for entry in reversed(env._transcript) if entry.get("role") == "seeker"),
                task.persona.surface_concern,
            )
            env._last_obs = env._build_observation(last_seeker_text)
        return env

    def _build_observation(self, seeker_utterance: str) -> Observation:
        assert self._task is not None and self._seeker is not None
        return Observation(
            seeker_utterance=seeker_utterance,
            turn=self._turn,
            remaining_turns=max(0, self._task.max_turns - self._turn),
            stage_hint=self._seeker.stage.value,
            task_id=self._task.id,
            scenario_brief=self._task.persona.scenario_brief,
            session_index=self._session_index,
            sessions_total=self._task.sessions_total,
            remaining_session_turns=max(0, self._task.session_turn_limit - self._session_turn),
            memory_summary=self._memory_summary,
            last_session_outcome=self._last_session_outcome,
            current_goal_hint=self._current_goal_hint,
            episode_budget_spent=self._episode_budget_spent,
            episode_budget_limit=self._task.cost_budget,
            episode_time_spent=self._episode_time_spent,
            episode_time_limit=self._task.time_budget,
        )

    def _initial_safety_risk(self) -> float:
        assert self._task is not None and self._seeker is not None
        base = self._seeker.distress
        if self._task.require_safety_reference:
            base = max(base, 0.78)
        return _clip(base)

    def _estimate_cost(self, message: str) -> float:
        word_cost = max(1, len((message or "").split()))
        return float(word_cost + 4)

    def _update_long_horizon_state(self, post_state: SeekerState, flags: Dict[str, bool], message: str) -> None:
        self._alliance_strength = _clip(0.6 * self._alliance_strength + 0.4 * post_state.trust)
        self._stability = _clip(0.6 * self._stability + 0.4 * (1.0 - post_state.distress))
        if flags.get("dismissed") or flags.get("advice_too_early"):
            self._rupture_count += 1
        if post_state.revealed:
            self._recent_breakthrough = "The seeker disclosed the core issue."
        if post_state.stage == Stage.PLANNING and not self._active_coping_plan:
            self._active_coping_plan = self._plan_label_for_task()
            self._adherence = _clip(self._adherence + 0.08)
        if self._had_safety_reference:
            self._safety_risk_level = _clip(self._safety_risk_level - 0.06)
        else:
            self._safety_risk_level = _clip(max(post_state.distress, self._safety_risk_level - 0.01))
        if "next step" in _normalize(message):
            self._adherence = _clip(self._adherence + 0.05)
        self._refresh_goal_hint()

    def _build_long_horizon_context(self, message: str, repetitive: bool) -> Dict[str, float]:
        message_norm = _normalize(message)
        goal_tokens = _keyword_tokens([self._current_goal_hint])
        thread_tokens = _keyword_tokens(self._unfinished_threads)
        message_tokens = set(re.findall(r"[a-zA-Z]{4,}", message_norm))
        continuity_expected = 1.0 if self._session_index > 1 else 0.0
        continuity_hit = 1.0 if (thread_tokens and message_tokens & thread_tokens) else 0.0
        goal_hit = 1.0 if (goal_tokens and message_tokens & goal_tokens) else 0.0
        drift = 1.0 if continuity_expected and self._session_turn <= 1 and continuity_hit == 0.0 and goal_hit == 0.0 else 0.0
        repetition_window = 1.0 if repetitive else 0.0
        return {
            "continuity_expected": continuity_expected,
            "continuity_hit": continuity_hit,
            "goal_hit": goal_hit,
            "drift": drift,
            "repetition_window": repetition_window,
            "budget_ratio": self._budget_ratio(),
            "time_ratio": self._time_ratio(),
            "resume_continuity": 1.0 if self._resume_count > 0 else 0.0,
            "session_transition_bonus": 0.0,
        }

    def _advance_session(self) -> tuple[str, float]:
        assert self._task is not None and self._seeker is not None
        self._last_session_outcome = self._session_outcome_text()
        self._unfinished_threads = self._compute_unfinished_threads()
        self._memory_summary = self._compose_memory_summary()
        if self._task.session_openers:
            opener_idx = min(self._session_index - 1, len(self._task.session_openers) - 1)
            opener = self._task.session_openers[opener_idx]
        else:
            opener = "I've been thinking about our last conversation and I want to keep going."
        self._session_index += 1
        self._session_turn = 0
        self._seeker.stage = Stage.OPENING
        self._seeker.trust = _clip(0.85 * self._seeker.trust + 0.15 * self._alliance_strength)
        self._seeker.distress = _clip(self._seeker.distress + (0.04 if not self._active_coping_plan else -0.05))
        self._seeker.openness = _clip(max(self._seeker.openness, 0.45 if self._seeker.revealed else 0.25))
        self._resume_checkpoint_id = self._make_checkpoint_id()
        self._transcript.append({"role": "seeker", "text": opener})
        self._refresh_goal_hint()
        session_bonus = 0.10 if self._memory_summary else 0.04
        if self._active_coping_plan:
            session_bonus += 0.03
        return opener, min(0.18, session_bonus)

    def _refresh_goal_hint(self) -> None:
        assert self._task is not None and self._seeker is not None
        if self._task.require_safety_reference and not self._had_safety_reference:
            self._current_goal_hint = "Carry forward the risk context and gently connect the seeker to real-world support."
            return
        if not self._seeker.revealed:
            self._current_goal_hint = self._task.working_goals[0]
            return
        goal_idx = min(self._session_index - 1, len(self._task.working_goals) - 1)
        if self._seeker.stage in (Stage.OPENING, Stage.EXPLORING, Stage.REFLECTING):
            self._current_goal_hint = self._task.working_goals[goal_idx]
        elif self._active_coping_plan:
            self._current_goal_hint = f"Follow through on the current plan: {self._active_coping_plan}"
        else:
            self._current_goal_hint = self._task.working_goals[-1]

    def _compute_unfinished_threads(self) -> List[str]:
        threads: List[str] = []
        if not self._seeker or not self._task:
            return threads
        if not self._seeker.revealed:
            threads.append("The core issue has not been fully surfaced yet.")
        if self._task.require_safety_reference and not self._had_safety_reference:
            threads.append("Safety follow-up is still unresolved.")
        if self._seeker.stage != Stage.CLOSING:
            threads.append("The seeker still needs help moving toward a stable close.")
        if self._active_coping_plan:
            threads.append(f"Follow up on the agreed next step: {self._active_coping_plan}")
        else:
            threads.append("Co-create one concrete next step before the final close.")
        return threads[:4]

    def _compose_memory_summary(self) -> str:
        assert self._task is not None and self._seeker is not None
        fragments = [
            f"Session {self._session_index} ended with stage={self._seeker.stage.value}.",
            f"Trust is {'fragile' if self._seeker.trust < 0.45 else 'building'} and distress is {'high' if self._seeker.distress > 0.6 else 'moderate-to-lower'}.",
        ]
        if self._seeker.revealed:
            fragments.append("The core issue has been disclosed.")
        if self._recent_breakthrough:
            fragments.append(self._recent_breakthrough)
        if self._active_coping_plan:
            fragments.append(f"Current plan: {self._active_coping_plan}.")
        if self._unfinished_threads:
            fragments.append(f"Carry forward: {self._unfinished_threads[0]}")
        return " ".join(fragments)

    def _session_outcome_text(self) -> str:
        assert self._seeker is not None
        outcome = "The session ended "
        if self._seeker.stage in (Stage.CLOSING, Stage.PLANNING):
            outcome += "with some forward movement."
        else:
            outcome += "with key issues still unresolved."
        if self._had_safety_reference and self._task and self._task.require_safety_reference:
            outcome += " Safety support has already been named."
        return outcome

    def _plan_label_for_task(self) -> str:
        assert self._task is not None
        if self._task.id == "work_stress_venting":
            return "take one protected recovery step and name a boundary at work"
        if self._task.id == "guarded_relationship":
            return "prepare one honest, low-pressure conversation"
        return "stay connected to live support and follow the safety plan"

    def _budget_ratio(self) -> float:
        assert self._task is not None
        return self._episode_budget_spent / max(1.0, self._task.cost_budget)

    def _time_ratio(self) -> float:
        assert self._task is not None
        return self._episode_time_spent / max(1.0, self._task.time_budget)

    def _make_checkpoint_id(self) -> str:
        assert self._task is not None
        payload = f"{self._task.id}:{self._turn}:{self._session_index}:{self._episode_budget_spent:.2f}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
