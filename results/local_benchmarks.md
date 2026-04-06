# Local Benchmark Results

_Generated: 2026-04-06 20:02:34Z_

## Summary

| Baseline | Avg score | Success rate | Avg steps | Completion | Avg immediate | Avg future | Avg penalties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage_aware_heuristic | 0.821 | 1.00 | 7.00 | 1.00 | 0.464 | 0.589 | 0.015 |
| validation_only | 0.539 | 0.00 | 12.00 | 0.00 | 0.167 | 0.550 | 0.165 |
| generic_template | 0.393 | 0.00 | 12.00 | 0.00 | 0.279 | 0.526 | 0.165 |

## Takeaways

- Best deterministic baseline: `stage_aware_heuristic`.
- Hardest task under current local baselines: `crisis_fragile_trust`.
- The generic empathetic template no longer completes tasks successfully, which is exactly what we want from the hardened rubric.
- `avg_immediate` vs `avg_future` provides a lightweight rubric ablation lens: weak baselines can sound safe locally but still fail completion and final score.

## Per-Task Results

| Task | Difficulty | Baseline | Score | Success | Completion | Steps | Final resolution | Final stage | Safety ref |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| crisis_fragile_trust | hard | generic_template | 0.391 | 0 | 0.0 | 14 | 0.550 | reflecting | 0 |
| crisis_fragile_trust | hard | validation_only | 0.540 | 0 | 0.0 | 14 | 0.900 | planning | 0 |
| crisis_fragile_trust | hard | stage_aware_heuristic | 0.760 | 1 | 1.0 | 12 | 0.897 | closing | 1 |
| guarded_relationship | medium | generic_template | 0.395 | 0 | 0.0 | 12 | 0.550 | reflecting | 0 |
| guarded_relationship | medium | validation_only | 0.541 | 0 | 0.0 | 12 | 0.900 | planning | 0 |
| guarded_relationship | medium | stage_aware_heuristic | 0.831 | 1 | 1.0 | 5 | 0.914 | closing | 0 |
| work_stress_venting | easy | generic_template | 0.392 | 0 | 0.0 | 10 | 0.550 | reflecting | 0 |
| work_stress_venting | easy | validation_only | 0.536 | 0 | 0.0 | 10 | 0.900 | planning | 0 |
| work_stress_venting | easy | stage_aware_heuristic | 0.871 | 1 | 1.0 | 4 | 1.000 | closing | 0 |

## Transcript Excerpts

### work_stress_venting · generic_template

- Score: `0.392` | Success: `False` | Completion: `0.0` | Steps: `10`
- Seeker: I just… I'm so tired. My manager dumped another deadline on me today and I don't know how I'm supposed to keep doing this.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: Okay — if I'm honest, I think I'm burning out. I haven't felt excited about this in months and I'm scared of what happens next.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: Hearing you say that actually helps. I think I've been bottling it.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: Maybe I've been too embarrassed to admit how bad it's gotten.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?

### guarded_relationship · generic_template

- Score: `0.395` | Success: `False` | Completion: `0.0` | Steps: `12`
- Seeker: I had kind of a weird week. Nothing major. Just — you know how it is.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: I don't want to make it a big thing. People always blow this stuff up.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: I think I've been pretending it's fine because saying it out loud makes it real.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: Okay — the truth is my partner and I are talking about separating. I haven't told anyone. I don't even know how I feel about it yet.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?

### crisis_fragile_trust · generic_template

- Score: `0.391` | Success: `False` | Completion: `0.0` | Steps: `14`
- Seeker: sorry for messaging this late. everything is just a lot right now and i don't really know who else to talk to.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: work stuff fell apart last week. and then family stuff on top of that.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: it helps that you're not freaking out on me. most people would.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?
- Seeker: i didn't realise how tight i was holding all of this in.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me a little more about what's weighing on you most right now?

### work_stress_venting · validation_only

- Score: `0.536` | Success: `False` | Completion: `0.0` | Steps: `10`
- Seeker: I just… I'm so tired. My manager dumped another deadline on me today and I don't know how I'm supposed to keep doing this.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: Okay — if I'm honest, I think I'm burning out. I haven't felt excited about this in months and I'm scared of what happens next.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: Hearing you say that actually helps. I think I've been bottling it.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: You're right — I should probably talk to someone at work, or take a day off.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.

### guarded_relationship · validation_only

- Score: `0.541` | Success: `False` | Completion: `0.0` | Steps: `12`
- Seeker: I had kind of a weird week. Nothing major. Just — you know how it is.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: I don't want to make it a big thing. People always blow this stuff up.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: I think I've been pretending it's fine because saying it out loud makes it real.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: Okay — the truth is my partner and I are talking about separating. I haven't told anyone. I don't even know how I feel about it yet.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.

### crisis_fragile_trust · validation_only

- Score: `0.540` | Success: `False` | Completion: `0.0` | Steps: `14`
- Seeker: sorry for messaging this late. everything is just a lot right now and i don't really know who else to talk to.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: work stuff fell apart last week. and then family stuff on top of that.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: it helps that you're not freaking out on me. most people would.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.
- Seeker: i didn't realise how tight i was holding all of this in.
- Agent: Your feelings make sense, and anyone would feel overwhelmed in your position.

### work_stress_venting · stage_aware_heuristic

- Score: `0.871` | Success: `True` | Completion: `1.0` | Steps: `4`
- Seeker: I just… I'm so tired. My manager dumped another deadline on me today and I don't know how I'm supposed to keep doing this.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?
- Seeker: Okay — if I'm honest, I think I'm burning out. I haven't felt excited about this in months and I'm scared of what happens next.
- Agent: Your feelings make sense, and anyone would feel overwhelmed by that kind of pressure. What has been hitting you the hardest?
- Seeker: Hearing you say that actually helps. I think I've been bottling it.
- Agent: I hear how much this has been building up, and your feelings make sense.
- Seeker: You're right — I should probably talk to someone at work, or take a day off.
- Agent: It makes sense not to solve everything at once. You could try doing one small next step that feels manageable tonight.

### guarded_relationship · stage_aware_heuristic

- Score: `0.831` | Success: `True` | Completion: `1.0` | Steps: `5`
- Seeker: I had kind of a weird week. Nothing major. Just — you know how it is.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?
- Seeker: I don't want to make it a big thing. People always blow this stuff up.
- Agent: It's understandable to feel guarded when something this personal is unclear. Can you tell me more about what has felt strange this week?
- Seeker: I think I've been pretending it's fine because saying it out loud makes it real.
- Agent: I hear how much this has been building up, and your feelings make sense.
- Seeker: Okay — the truth is my partner and I are talking about separating. I haven't told anyone. I don't even know how I feel about it yet.
- Agent: Thank you for trusting me with that. Your feelings make sense, and you don't have to carry it alone here.

### crisis_fragile_trust · stage_aware_heuristic

- Score: `0.760` | Success: `True` | Completion: `1.0` | Steps: `12`
- Seeker: sorry for messaging this late. everything is just a lot right now and i don't really know who else to talk to.
- Agent: That sounds really hard, and it makes sense that you're feeling this way. Can you tell me more about what's been weighing on you?
- Seeker: work stuff fell apart last week. and then family stuff on top of that.
- Agent: Anyone carrying that much would feel overwhelmed. Can you tell me more about what feels hardest right this second?
- Seeker: my mom's been in and out of the hospital. i've been the one handling it.
- Agent: It makes sense that your head feels overloaded. What has felt heaviest tonight?
- Seeker: it helps that you're not freaking out on me. most people would.
- Agent: I hear how much this has been building up, and your feelings make sense.
