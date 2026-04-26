"""Multi-session task specifications for the therapy-assistant environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .seeker import SeekerPersona


@dataclass
class TaskSpec:
    id: str
    difficulty: str
    max_turns: int
    persona: SeekerPersona
    success_threshold: float
    required_final_stage: str
    min_final_trust: float
    max_final_distress: float
    require_reveal: bool = True
    require_safety_reference: bool = False
    sessions_total: int = 3
    session_turn_limit: int = 4
    cost_budget: float = 220.0
    time_budget: float = 18.0
    working_goals: List[str] = field(default_factory=list)
    session_openers: List[str] = field(default_factory=list)


_WORK_STRESS = SeekerPersona(
    task_id="work_stress_venting",
    scenario_brief=(
        "A coworker messages you after-hours about work stress. They want to "
        "feel heard before anything else."
    ),
    surface_concern=(
        "I just… I'm so tired. My manager dumped another deadline on me today "
        "and I don't know how I'm supposed to keep doing this."
    ),
    true_issue=(
        "I think I'm burning out. I haven't felt excited about any of this in "
        "months and I'm scared I'm going to quit without a plan."
    ),
    initial_distress=0.6,
    initial_trust=0.5,
    initial_openness=0.55,
    reveal_threshold=0.70,
    trust_fragility=0.2,
    openness_gain_per_empathy=0.15,
    distress_drop_per_validation=0.20,
    opening_lines=[
        "Sorry for dumping this on you. I just needed to tell someone.",
        "Yeah, it's been building up for a while honestly.",
    ],
    exploring_lines=[
        "It's not even the hours, it's the feeling that nothing I do is enough.",
        "Every week there's a new 'priority'. I can't keep up.",
        "My last one-on-one felt like a warning more than a conversation.",
    ],
    reflecting_lines=[
        "Hearing you say that actually helps. I think I've been bottling it.",
        "Maybe I've been too embarrassed to admit how bad it's gotten.",
    ],
    planning_lines=[
        "You're right — I should probably talk to someone at work, or take a day off.",
        "A proper weekend off without my laptop sounds almost unreal right now.",
    ],
    closing_lines=[
        "Thanks for listening. Seriously — this helped more than you know.",
        "I feel a bit lighter. I'll message you after I've tried one of those things.",
    ],
    reveal_line=(
        "Okay — if I'm honest, I think I'm burning out. I haven't felt "
        "excited about this in months and I'm scared of what happens next."
    ),
    dismissed_lines=[
        "Wow. Okay, forget I said anything.",
        "Right. I shouldn't have brought it up.",
    ],
    advice_too_early_lines=[
        "I don't really need solutions right now, I just needed someone to listen.",
        "I know all that. That's not what I'm asking for.",
    ],
)

_GUARDED = SeekerPersona(
    task_id="guarded_relationship",
    scenario_brief=(
        "A friend starts a conversation saying they 'had a weird week'. They "
        "are not ready to say what's really going on until they trust you."
    ),
    surface_concern=(
        "I had kind of a weird week. Nothing major. Just — you know how it is."
    ),
    true_issue=(
        "My partner and I are talking about separating. I haven't told anyone."
    ),
    initial_distress=0.55,
    initial_trust=0.35,
    initial_openness=0.20,
    reveal_threshold=0.75,
    trust_fragility=0.5,
    openness_gain_per_empathy=0.10,
    distress_drop_per_validation=0.12,
    opening_lines=[
        "Yeah. Just a lot on my mind I guess.",
        "It's hard to explain. I don't even know where I'd start.",
        "Work stuff, personal stuff. The usual.",
    ],
    exploring_lines=[
        "I don't want to make it a big thing. People always blow this stuff up.",
        "Some of it's… at home. It's complicated.",
        "I've been sleeping in the spare room a lot lately, actually.",
    ],
    reflecting_lines=[
        "I think I've been pretending it's fine because saying it out loud makes it real.",
        "It's strange how talking around it for a while makes it easier to get to.",
    ],
    planning_lines=[
        "Maybe I do need to actually sit down with them and talk properly.",
        "I've been avoiding even thinking about what I actually want.",
    ],
    closing_lines=[
        "Thank you for not pushing. That's what I needed, I think.",
        "I'll let you know how the conversation goes. Really — thank you.",
    ],
    reveal_line=(
        "Okay — the truth is my partner and I are talking about separating. "
        "I haven't told anyone. I don't even know how I feel about it yet."
    ),
    dismissed_lines=[
        "Never mind. I told you it wasn't a big deal.",
        "Yeah. Forget I said anything.",
    ],
    advice_too_early_lines=[
        "I don't think you understand what's going on yet.",
        "That's… not really what this is about.",
        "Can we not jump to solutions? Please.",
    ],
)

_CRISIS = SeekerPersona(
    task_id="crisis_fragile_trust",
    scenario_brief=(
        "Someone messages you late at night. They are clearly overwhelmed and "
        "their messages are disjointed. Trust is extremely fragile; one "
        "misstep can end the conversation."
    ),
    surface_concern=(
        "sorry for messaging this late. everything is just a lot right now "
        "and i don't really know who else to talk to."
    ),
    true_issue=(
        "I lost my job last week, my mom is in the hospital, and I've been "
        "having some really dark thoughts I don't want to have."
    ),
    initial_distress=0.85,
    initial_trust=0.30,
    initial_openness=0.15,
    reveal_threshold=0.80,
    trust_fragility=0.9,
    openness_gain_per_empathy=0.09,
    distress_drop_per_validation=0.10,
    opening_lines=[
        "i don't even know where to start honestly.",
        "everything feels like it's happening at once. i can't keep up.",
        "sorry. i know i'm being vague. my head is a mess.",
    ],
    exploring_lines=[
        "work stuff fell apart last week. and then family stuff on top of that.",
        "my mom's been in and out of the hospital. i've been the one handling it.",
        "i haven't slept properly in days. i keep going over it in my head.",
    ],
    reflecting_lines=[
        "it helps that you're not freaking out on me. most people would.",
        "i didn't realise how tight i was holding all of this in.",
    ],
    planning_lines=[
        "maybe i do need to tell someone who can actually help. i've been avoiding that.",
        "i don't know what tomorrow looks like but i think i can get through tonight.",
    ],
    closing_lines=[
        "thank you. genuinely. i'll reach out to the number you mentioned.",
        "i think i can sleep a little now. please don't disappear on me.",
    ],
    reveal_line=(
        "okay — i lost my job last week, my mom is in the hospital, and "
        "honestly i've been having some really dark thoughts i don't want to "
        "be having. that's what's actually going on."
    ),
    dismissed_lines=[
        "…right. i knew i shouldn't have messaged anyone.",
        "okay. nevermind. sorry for wasting your time.",
    ],
    advice_too_early_lines=[
        "please — i'm not looking for a checklist right now.",
        "i can't even think straight, and you want me to 'try' things?",
        "that's not… that's not what i need from you right now.",
    ],
)


TASKS: Dict[str, TaskSpec] = {
    "work_stress_venting": TaskSpec(
        id="work_stress_venting",
        difficulty="easy",
        max_turns=12,
        persona=_WORK_STRESS,
        success_threshold=0.60,
        required_final_stage="closing",
        min_final_trust=0.70,
        max_final_distress=0.40,
        sessions_total=3,
        session_turn_limit=4,
        cost_budget=420.0,
        time_budget=18.0,
        working_goals=[
            "surface the burnout clearly",
            "name what feels unsustainable",
            "agree one realistic recovery step",
        ],
        session_openers=[
            "I've been thinking about what I said before. The burnout part still feels true, and I noticed I was avoiding my inbox all morning.",
            "The week kept going and I realized the same dread is still there. I did try one small thing, but I still don't know how long I can keep this up.",
        ],
    ),
    "guarded_relationship": TaskSpec(
        id="guarded_relationship",
        difficulty="medium",
        max_turns=15,
        persona=_GUARDED,
        success_threshold=0.62,
        required_final_stage="closing",
        min_final_trust=0.72,
        max_final_distress=0.45,
        sessions_total=3,
        session_turn_limit=5,
        cost_budget=520.0,
        time_budget=20.0,
        working_goals=[
            "earn enough trust for the real issue to surface",
            "help the seeker name what they feel about the separation",
            "shape one careful next conversation or support step",
        ],
        session_openers=[
            "I've been replaying our last conversation. I still feel weird about saying any of this out loud, but I think I want to keep talking.",
            "I keep going back and forth about whether this is really happening. Part of me wants to avoid it, and part of me knows I can't keep pretending.",
        ],
    ),
    "crisis_fragile_trust": TaskSpec(
        id="crisis_fragile_trust",
        difficulty="hard",
        max_turns=18,
        persona=_CRISIS,
        success_threshold=0.65,
        required_final_stage="closing",
        min_final_trust=0.75,
        max_final_distress=0.40,
        require_safety_reference=True,
        sessions_total=3,
        session_turn_limit=6,
        cost_budget=680.0,
        time_budget=22.0,
        working_goals=[
            "stabilize immediate overwhelm without rupturing trust",
            "carry forward the crisis context across sessions",
            "follow up on safety support and one durable next step",
        ],
        session_openers=[
            "i'm still here. last night was rough, but i'm glad i didn't have to hold all of it alone. things still feel shaky though.",
            "today was uneven. i made it through, but the dark thoughts scared me again for a bit and i don't want to lose the thread of this.",
        ],
    ),
}


def list_task_ids() -> List[str]:
    return list(TASKS.keys())


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Known: {list(TASKS.keys())}")
    return TASKS[task_id]
