# TikTok TechJam Submission Checklist

This file separates artifacts that are already available from values that must come from the final live LLM-enabled run.

## 1. Devpost project description

- [x] Project overview and problem solution: `DEVPOST.md`
- [x] Development tools
- [x] APIs
- [x] Libraries/frameworks
- [x] Dataset/assets
- [ ] Replace team contribution section if there are additional team members

## 2. Public repository

- [x] Public GitHub repository
- [x] README with overview/setup/reproduction/limitations
- [x] Agent code and LLM planner
- [x] Generated-code sandbox and contract validation
- [x] Run-log format
- [x] Resource telemetry generator

## 3. Live LLM run — REQUIRED BEFORE FINAL CLAIMS

Do not fabricate these values. Run with the local `.env` present on the machine executing the agent:

```bash
python3 agent.py --data_dir ./KuaiRand-Pure/data --out_dir run_logs --max_iters 50
python3 summarize_submission.py --run_dir run_logs
```

Then verify:

- [ ] `run_logs/llm_calls.jsonl` exists and contains provider-reported usage
- [ ] At least one `iterations.jsonl` record has `family: "llm_code"`
- [ ] At least one LLM-authored candidate reached real training rather than only dry-run validation
- [ ] Every iteration has hypothesis, code diff, metrics/status, and errors where applicable
- [ ] Failed candidates have recovery evidence in subsequent planning context
- [ ] `run_logs/run_metadata.json` has wall-clock duration and iteration count
- [ ] Manual interventions count is accurate

## 4. Final benchmark output

- [ ] Validation-best configuration identified
- [ ] `submission.csv` generated from the validation-best configuration
- [ ] `submit.py --check` passes
- [ ] Final test scoring performed only once for the selected candidate
- [ ] KuaiRand-Pure GAUC recorded
- [ ] KuaiRand-Pure nDCG@5 recorded
- [ ] Absolute primary delta vs official FM baseline recorded
- [ ] Bonus benchmark outputs included only if actually run

## 5. Resource usage

Populate only from `run_logs/run_metadata.json` and `summarize_submission.py` output:

| Resource | Final value |
|---|---:|
| LLM input tokens | pending live run |
| LLM output tokens | pending live run |
| LLM total tokens | pending live run |
| Agent wall-clock | pending live run |
| Iterations | pending live run |
| GPU-hours | pending live run / 0 if CPU-only |
| Manual interventions | pending live run |

## Current honest status

The repository contains the LLM-authored-code implementation, but the historical logs inspected before the current submission-preparation changes contain no `llm_generated` / `llm_code` iterations. Therefore the final Devpost submission must not claim that the LLM generated or trained a candidate until the live run above produces that evidence.
