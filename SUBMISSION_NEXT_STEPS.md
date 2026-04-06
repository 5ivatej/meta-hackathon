# Submission Next Steps

This file is the short operational checklist for turning the current repo into
a polished hackathon submission.

## 1. Run the local benchmark ladder

This gives you deterministic numbers for the baseline section and proves the
rubric separates weak generic empathy from staged, task-aware behavior.

```powershell
py -3 benchmark.py
```

Artifacts written:

- `results/local_benchmarks.md`
- `results/local_benchmarks.json`

After running:

- copy the summary table from `results/local_benchmarks.md`
- paste the final baseline numbers into [README.md](C:\Users\Gokul nandan T M\Desktop\personalprojects\meta\meta-hackathon\README.md)

## 2. Run the mandatory hackathon stdout-contract baseline

This is the script the submission already exposes.

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your-token>"
$env:ESC_ENV_URL="http://localhost:7860"
py -3 inference.py
```

Use this when you want the strict `[START] / [STEP] / [END]` output format.

## 3. Run the Markdown-writing LLM benchmark

Use this when you want a reusable results file for the README or final report.

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your-token>"
$env:ESC_ENV_URL="http://localhost:7860"
py -3 benchmark_llm.py
```

Artifacts written:

- `results/llm_benchmark.md`
- `results/llm_benchmark.json`

## 4. Replace the `TBD` results table in the README

Update the `Baseline scores` section in [README.md](C:\Users\Gokul nandan T M\Desktop\personalprojects\meta\meta-hackathon\README.md) with:

- deterministic local baselines from `benchmark.py`
- one real LLM baseline from `benchmark_llm.py`

Recommended final table:

- `generic_template`
- `validation_only`
- `stage_aware_heuristic`
- one real LLM baseline

## 5. Add one short benchmark narrative to the README

Keep it brief. Include:

- the generic repeated empathy template no longer succeeds
- the stage-aware heuristic completes all tasks
- the crisis task requires an explicit safety-aware finish

## 6. Smoke-test the deployable artifact

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

## 7. Optional but high-value polish

If you still have time, these are the best improvements:

- add one screenshot or short GIF of a successful hard-task trajectory
- add a tiny `Why this benchmark is hard` section to the README
- add one extra adversarial baseline later, only if the current docs/results are already polished

## 8. Final pre-submit check

Make sure these are true:

- local benchmark artifacts exist in `results/`
- one LLM benchmark artifact exists in `results/`
- README contains real numbers, not `TBD`
- Docker build works
- the hard task is only successful when safety support is referenced
- the generic template baseline does not succeed
