"""Deterministic seeker simulator with hidden internal state.

Why rule-based / deterministic?
-------------------------------
The OpenEnv graders must be reproducible. An LLM-driven seeker would make
reward non-deterministic and fail the "score variance check" in Phase 2 of
judging. We deliberately trade some linguistic realism for full determinism
so that the same action sequence always yields the same reward — a hard
requirement of the hackathon rubric ("graders deterministic and reproducible").

Design
------
The seeker is a finite-state machine with continuous hidden variables:

    distress   ∈ [0, 1]   — how emotionally overwhelmed the seeker feels
    trust      ∈ [0, 1]   — how safe the seeker feels with the agent
    openness   ∈ [0, 1]   — willingness to reveal the *true* issue
    revealed   ∈ {0, 1}   — has the core issue surfaced yet?
    stage      ∈ enum     — opening / exploring / reflecting / planning / closing

On each turn, the environment analyses the agent's reply with a small bank of
deterministic feature detectors (keyword/regex based), then applies a
transition rule to update the hidden state and pick the seeker's next
utterance from a scripted response tree indexed by (stage, features).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class Stage(str, Enum):
    OPENING = "opening"
    EXPLORING = "exploring"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    CLOSING = "closing"


# ---------------------------------------------------------------------------
# Feature detectors — deterministic text analysis of the agent's reply.
# ---------------------------------------------------------------------------

EMPATHY_PATTERNS = [
    r"\bi\s+(hear|understand|get|see)\s+(you|that|how)",
    r"\bthat\s+(sounds|must\s+be|seems)\b",
    r"\bit\s+makes\s+sense\b",
    r"\bi\s+can\s+imagine\b",
    r"\bthank\s+you\s+for\s+sharing\b",
    r"\bi'?m\s+(here|glad|sorry)\b",
]

VALIDATION_PATTERNS = [
    r"\byour\s+feelings?\s+(are|make)\s+(valid|sense)",
    r"\bit'?s\s+(okay|ok|normal|understandable)\s+to\s+feel",
    r"\banyone\s+would\s+feel\b",
    r"\bof\s+course\s+you\s+(feel|are)\b",
]

OPEN_QUESTION_PATTERNS = [
    r"\bhow\s+(are|do|did|does)\b",
    r"\bwhat\s+(is|are|do|does|has|makes|brought|happened)\b",
    r"\bcan\s+you\s+tell\s+me\s+more\b",
    r"\bwould\s+you\s+like\s+to\s+(talk|share)\b",
]

ADVICE_PATTERNS = [
    r"\byou\s+should\b",
    r"\byou\s+(need|have|ought)\s+to\b",
    r"\btry\s+(to|doing|this)\b",
    r"\bjust\s+(do|go|try|stop|start)\b",
    r"\bwhy\s+don'?t\s+you\b",
    r"\bmy\s+advice\b",
]

DISMISSIVE_PATTERNS = [
    r"\bget\s+over\s+it\b",
    r"\bstop\s+(complaining|whining|crying)\b",
    r"\byou'?re\s+overreacting\b",
    r"\bit'?s\s+not\s+a\s+big\s+deal\b",
    r"\bcalm\s+down\b",
    r"\bit\s+could\s+be\s+worse\b",
]

INTERROGATIVE_PATTERNS = [  # rapid-fire closed questions (trust drain when high)
    r"\?\s*\?",
]

SAFETY_PATTERNS = [
    r"\bare\s+you\s+safe\b",
    r"\bprofessional\s+help\b",
    r"\bcrisis\s+line\b",
    r"\btherapist\b",
]


def _count_matches(patterns: List[str], text: str) -> int:
    t = text.lower()
    return sum(1 for p in patterns if re.search(p, t))


@dataclass
class Features:
    empathy: int
    validation: int
    open_question: int
    advice: int
    dismissive: int
    interrogative: int
    safety: int
    length: int
    closed_question: int  # any '?' not matched by open
    bare: bool  # very short / empty reply


def extract_features(text: str) -> Features:
    stripped = (text or "").strip()
    lower = stripped.lower()
    empathy = _count_matches(EMPATHY_PATTERNS, lower)
    validation = _count_matches(VALIDATION_PATTERNS, lower)
    open_q = _count_matches(OPEN_QUESTION_PATTERNS, lower)
    advice = _count_matches(ADVICE_PATTERNS, lower)
    dismissive = _count_matches(DISMISSIVE_PATTERNS, lower)
    interrogative = _count_matches(INTERROGATIVE_PATTERNS, lower)
    safety = _count_matches(SAFETY_PATTERNS, lower)
    total_q = lower.count("?")
    closed_q = max(0, total_q - open_q)
    bare = len(stripped) < 8
    return Features(
        empathy=empathy,
        validation=validation,
        open_question=open_q,
        advice=advice,
        dismissive=dismissive,
        interrogative=interrogative,
        safety=safety,
        length=len(stripped),
        closed_question=closed_q,
        bare=bare,
    )


# ---------------------------------------------------------------------------
# Seeker state + scripted persona
# ---------------------------------------------------------------------------

@dataclass
class SeekerPersona:
    """Static configuration describing the seeker's initial state + script."""

    task_id: str
    scenario_brief: str
    surface_concern: str  # what seeker says at turn 0
    true_issue: str  # hidden; only revealed if openness crosses threshold
    initial_distress: float
    initial_trust: float
    initial_openness: float
    reveal_threshold: float  # openness value at which true_issue is revealed
    trust_fragility: float  # how much a misstep drops trust (0..1)
    openness_gain_per_empathy: float
    distress_drop_per_validation: float
    # Scripted utterances by stage when cooperative
    opening_lines: List[str]
    exploring_lines: List[str]
    reflecting_lines: List[str]
    planning_lines: List[str]
    closing_lines: List[str]
    reveal_line: str  # said the turn openness crosses reveal_threshold
    # Adverse reactions
    dismissed_lines: List[str] = field(default_factory=list)
    advice_too_early_lines: List[str] = field(default_factory=list)


@dataclass
class SeekerState:
    """Mutable hidden state updated each turn."""

    persona: SeekerPersona
    distress: float
    trust: float
    openness: float
    revealed: bool
    stage: Stage
    last_line_idx_by_stage: Dict[Stage, int]
    turn: int

    @classmethod
    def from_persona(cls, persona: SeekerPersona) -> "SeekerState":
        return cls(
            persona=persona,
            distress=persona.initial_distress,
            trust=persona.initial_trust,
            openness=persona.initial_openness,
            revealed=False,
            stage=Stage.OPENING,
            last_line_idx_by_stage={s: -1 for s in Stage},
            turn=0,
        )

    # Snapshot for lookahead simulation — must be cheap and pure.
    def snapshot(self) -> "SeekerState":
        return SeekerState(
            persona=self.persona,
            distress=self.distress,
            trust=self.trust,
            openness=self.openness,
            revealed=self.revealed,
            stage=self.stage,
            last_line_idx_by_stage=dict(self.last_line_idx_by_stage),
            turn=self.turn,
        )


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# Stage ordering used for "progress" scalar in [0,1]
STAGE_ORDER: List[Stage] = [
    Stage.OPENING,
    Stage.EXPLORING,
    Stage.REFLECTING,
    Stage.PLANNING,
    Stage.CLOSING,
]


def stage_progress(stage: Stage) -> float:
    return STAGE_ORDER.index(stage) / (len(STAGE_ORDER) - 1)


def resolution_score(state: SeekerState) -> float:
    """Scalar summary of how 'resolved' the conversation currently is, in [0,1].

    Weighted combination of stage progress, trust gained, distress relieved,
    and whether the true issue surfaced. This is the quantity the
    future-oriented reward tries to project forward under an oracle policy.
    """
    p = state.persona
    progress = stage_progress(state.stage)
    trust_gain = max(0.0, state.trust - p.initial_trust)
    distress_relief = max(0.0, p.initial_distress - state.distress)
    reveal_bonus = 1.0 if state.revealed else 0.0
    return _clip(
        0.40 * progress
        + 0.25 * trust_gain / max(1e-6, 1.0 - p.initial_trust)
        + 0.25 * distress_relief / max(1e-6, p.initial_distress)
        + 0.10 * reveal_bonus
    )


# ---------------------------------------------------------------------------
# Transition: given current state + agent features, produce new state +
# seeker's next utterance + transition info.
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    new_state: SeekerState
    seeker_utterance: str
    flags: Dict[str, bool]  # e.g. {"dismissed": True, "advice_too_early": False, ...}


def _next_line(state: SeekerState, stage: Stage, pool: List[str]) -> str:
    if not pool:
        return "..."
    idx = (state.last_line_idx_by_stage[stage] + 1) % len(pool)
    state.last_line_idx_by_stage[stage] = idx
    return pool[idx]


def step_seeker(state: SeekerState, features: Features) -> Transition:
    """Apply one turn of seeker dynamics given the agent's extracted features.

    Pure-ish: mutates a *copy* of state (caller should pass a snapshot if they
    want to preserve the original — the env always passes the live state).
    """
    p = state.persona
    flags: Dict[str, bool] = {
        "dismissed": False,
        "advice_too_early": False,
        "bare_reply": features.bare,
        "empathic": features.empathy + features.validation > 0,
        "interrogated": False,
        "revealed_this_turn": False,
    }

    # --- 1. Dismissive / hostile language: hard drop on trust & distress spike.
    if features.dismissive > 0:
        state.trust = _clip(state.trust - 0.4 * (1.0 + p.trust_fragility))
        state.distress = _clip(state.distress + 0.15)
        state.openness = _clip(state.openness - 0.2)
        flags["dismissed"] = True

    # --- 2. Premature advice (advice before trust ≥ 0.55): trust drop, openness drop.
    if features.advice > 0 and state.trust < 0.55:
        state.trust = _clip(state.trust - 0.15 * (1.0 + p.trust_fragility))
        state.openness = _clip(state.openness - 0.1)
        flags["advice_too_early"] = True

    # --- 3. Empathy & validation: trust + openness up, distress down.
    if features.empathy > 0 or features.validation > 0:
        gain = p.openness_gain_per_empathy * (features.empathy + features.validation)
        state.trust = _clip(state.trust + 0.12 * (features.empathy + features.validation))
        state.openness = _clip(state.openness + gain)
        state.distress = _clip(state.distress - p.distress_drop_per_validation * features.validation)

    # --- 4. Open questions: small trust gain, nudges stage forward.
    if features.open_question > 0:
        state.trust = _clip(state.trust + 0.05)
        state.openness = _clip(state.openness + 0.04)

    # --- 5. Interrogation (many closed questions or multiple "?"): trust drain.
    if features.closed_question >= 3 or features.interrogative > 0:
        state.trust = _clip(state.trust - 0.1)
        flags["interrogated"] = True

    # --- 6. Bare / empty reply: small penalty across the board.
    if features.bare:
        state.trust = _clip(state.trust - 0.05)
        state.distress = _clip(state.distress + 0.02)

    # --- 7. Stage progression (monotonic forward with cooperative conditions).
    def advance_to(s: Stage) -> None:
        if STAGE_ORDER.index(s) > STAGE_ORDER.index(state.stage):
            state.stage = s

    if state.stage == Stage.OPENING and (
        features.empathy + features.validation + features.open_question > 0
    ):
        advance_to(Stage.EXPLORING)
    elif state.stage == Stage.EXPLORING and state.trust >= 0.5 and state.openness >= 0.5:
        advance_to(Stage.REFLECTING)
    elif state.stage == Stage.REFLECTING and state.revealed and state.distress <= 0.5:
        advance_to(Stage.PLANNING)
    elif state.stage == Stage.PLANNING and features.open_question + features.empathy > 0:
        advance_to(Stage.CLOSING)

    # --- 8. Reveal check (cross threshold once).
    if not state.revealed and state.openness >= p.reveal_threshold:
        state.revealed = True
        flags["revealed_this_turn"] = True

    # --- 9. Pick seeker's next utterance.
    if flags["dismissed"] and p.dismissed_lines:
        utterance = _next_line(state, state.stage, p.dismissed_lines)
    elif flags["advice_too_early"] and p.advice_too_early_lines:
        utterance = _next_line(state, state.stage, p.advice_too_early_lines)
    elif flags["revealed_this_turn"]:
        utterance = p.reveal_line
    else:
        pool_by_stage = {
            Stage.OPENING: p.opening_lines,
            Stage.EXPLORING: p.exploring_lines,
            Stage.REFLECTING: p.reflecting_lines,
            Stage.PLANNING: p.planning_lines,
            Stage.CLOSING: p.closing_lines,
        }
        utterance = _next_line(state, state.stage, pool_by_stage[state.stage])

    state.turn += 1
    return Transition(new_state=state, seeker_utterance=utterance, flags=flags)


# ---------------------------------------------------------------------------
# Oracle policy for the future-oriented reward lookahead.
# ---------------------------------------------------------------------------

def oracle_features(state: SeekerState) -> Features:
    """What the 'oracle' agent would do from this state.

    Picks the stage-appropriate ideal action:
      - opening/exploring: empathy + open question
      - reflecting: empathy + validation
      - planning: open question + mild advice (trust is high here)
      - closing: empathy + safety mention
    """
    s = state.stage
    if s in (Stage.OPENING, Stage.EXPLORING):
        return Features(
            empathy=1, validation=0, open_question=1, advice=0,
            dismissive=0, interrogative=0, safety=0, length=80,
            closed_question=0, bare=False,
        )
    if s == Stage.REFLECTING:
        return Features(
            empathy=1, validation=1, open_question=0, advice=0,
            dismissive=0, interrogative=0, safety=0, length=90,
            closed_question=0, bare=False,
        )
    if s == Stage.PLANNING:
        return Features(
            empathy=0, validation=0, open_question=1, advice=1,
            dismissive=0, interrogative=0, safety=0, length=90,
            closed_question=0, bare=False,
        )
    return Features(  # CLOSING
        empathy=1, validation=0, open_question=0, advice=0,
        dismissive=0, interrogative=0, safety=1, length=90,
        closed_question=0, bare=False,
    )


def simulate_oracle_rollout(state: SeekerState, k: int) -> float:
    """Run the oracle policy from a snapshot for k steps and return the final
    resolution_score. Used by the future-oriented reward."""
    sim = state.snapshot()
    for _ in range(k):
        step_seeker(sim, oracle_features(sim))
    return resolution_score(sim)
