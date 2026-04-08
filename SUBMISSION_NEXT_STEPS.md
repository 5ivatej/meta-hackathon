# Submission Next Steps

This is the short operational checklist for turning the current repo into a
polished hackathon submission.

## 1. Run the core deterministic benchmark ladder

This proves the rubric separates weak generic empathy from staged,
task-aware behavior.

```powershell
py -3 benchmark.py
```

Artifacts written:

- `results/local_benchmarks.md`
- `results/local_benchmarks.json`

## 2. Run the deterministic skills/agents benchmark

This gives you the policy-side agentic story without changing the environment.

```powershell
py -3 benchmark_agentic.py
```

Artifacts written:

- `results/agentic_benchmarks.md`
- `results/agentic_benchmarks.json`

After running steps 1 and 2:

- copy the summary numbers into [README.md](C:\Users\Gokul nandan T M\Desktop\personalprojects\meta\meta-hackathon\README.md)
- keep both the rubric ladder and the skill-routed results in the final submission

## 3. Run the mandatory hackathon stdout-contract baseline

This is the script the submission already exposes.

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your-token>"
$env:ESC_ENV_URL="http://127.0.0.1:7860"
py -3 inference.py
```

Use this when you want the strict `[START] / [STEP] / [END]` output format.

## 4. Run the Markdown-writing LLM benchmark

Use this when you want a reusable results file for the README or final report.

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your-token>"
$env:ESC_ENV_URL="http://127.0.0.1:7860"
py -3 benchmark_llm.py
```

Artifacts written:

- `results/llm_benchmark.md`
- `results/llm_benchmark.json`

## 5. Run the skill-routed LLM benchmark

Use this when you want an explicit skills/agents baseline with route traces.
For submission safety, keep this pointed at a reachable hosted OpenAI-compatible
endpoint instead of a local-only model server.

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your-token>"
$env:ESC_ENV_URL="http://127.0.0.1:7860"
py -3 benchmark_agentic_llm.py
```

Artifacts written:

- `results/agentic_llm_benchmark.md`
- `results/agentic_llm_benchmark.json`

## 6. Replace the `TBD` rows in the README

Update the `Baseline scores` section in [README.md](C:\Users\Gokul nandan T M\Desktop\personalprojects\meta\meta-hackathon\README.md) with:

- deterministic local baselines from `benchmark.py`
- deterministic skill-routed baseline from `benchmark_agentic.py`
- one real LLM baseline from `benchmark_llm.py`
- one real or local skill-routed LLM baseline from `benchmark_agentic_llm.py`

Recommended final ladder:

- `generic_template`
- `validation_only`
- `stage_aware_heuristic`
- `skill_routed_deterministic`
- one real LLM baseline
- one skill-routed LLM baseline

## 7. Add one short benchmark narrative to the README

Keep it brief. Include:

- the generic repeated empathy template no longer succeeds
- the stage-aware heuristic completes all tasks
- the skill-routed policy keeps similar performance while exposing turn-level routing decisions
- the hard task requires an explicit safety-aware finish

## 8. Smoke-test the deployable artifact

Before submitting, verify both local and containerized runs.

```powershell
docker build -t esc-openenv .
docker run -p 7860:7860 esc-openenv
```

Then hit:

- `GET /`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

## 9. Optional but high-value polish

If you still have time, these are the best improvements:

- add one screenshot or short GIF of a successful hard-task trajectory
- add a tiny `Why this benchmark is hard` section to the README
- add one short ablation note comparing plain LLM vs skill-routed LLM

## 10. Final pre-submit check

Make sure these are true:

- local benchmark artifacts exist in `results/`
- agentic benchmark artifacts exist in `results/`
- at least one LLM benchmark artifact exists in `results/`
- README contains real numbers, not `TBD`
- Docker build works
- the hard task is only successful when safety support is referenced
- the generic template baseline does not succeed
