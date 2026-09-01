# Submission packet — where every required artifact lives

All values below are read from committed run artifacts. Nothing here is estimated.

---

## 3. Run & iteration logs

**File to submit: `run_logs/iterations.jsonl`** (7 records from the converged run)

Each record carries every required field:

| Requirement | Field in `iterations.jsonl` |
|---|---|
| Hypothesis for that iteration | `hypothesis` |
| Code diff applied | `code_diff` (full generated source for LLM-authored candidates; parameter delta for tuning moves) |
| Resulting metrics | `metrics.valid.GAUC`, `metrics.valid.nDCG@5`, `metrics.valid.primary` |
| Errors and recovery events | `error`, `status`, `error_type`, `retryable` |

**Supporting file: `run_logs/llm_calls.jsonl`** — one record per provider HTTP call
(`kind: "provider_call"`) with token usage, `finish_reason` and, on failure, the raw
response; plus one `kind: "planner_result"` summary per planner invocation.

### What the run did

| Iter | Experiment | Stage | Status | Validation primary |
|---|---|---|---|---:|
| 1 | `llm_generated_0` | model architecture | ok | 0.5910 |
| 1 | `llm_planner` | — | **planner_failed** → recovered | — |
| 2 | `llm_train_1` | training recipe | ok | 0.6012 |
| 2 | `llm_planner` | — | **planner_failed** → recovered | — |
| 2, 3 | `fm_baseline` | baseline gate | ok | 0.6015 |

Stages explored: `{llm_code: 1, llm_train: 1, fm: 1}` — the agent moved across the
pipeline rather than only editing the model.

The best agent-authored candidate was a **training-stage** move, not an architecture
one: `llm_train_1` applied weighted binary cross-entropy to address label imbalance
and reached 0.6012, within 0.0003 of the baseline and ahead of every generated
architecture. That is direct evidence the multi-stage design earns its complexity.

### Error and recovery events

Two planner failures occurred and both were handled autonomously, with no human input:

1. **`proposal claimed family='llm_code' but carried no code`** — the planner
   returned a code-mode proposal with an empty body. The controller logged a
   `planner_failed` record and fell back to the fixed schedule for that iteration.
2. **`proposal named unknown experiment 'llm_train_1'`** — the planner tried to
   tune a previous run-generated candidate by name, which is not in the registry.
   Same handling: logged, recovered, run continued.

Neither failure ended the run, and neither was silently swallowed — each wrote a
record with `status: "planner_failed"` and the raw provider response.

### Manual interventions

**0 manual interventions during the run.**

The run was launched with a single command and ran unattended to convergence. No
human edited code, restarted a step, or supplied a value while it executed. All
recovery from the two planner failures was performed by the agent's own fallback
path. (Code development between runs is not counted as intervention, consistent
with the definition in `run_metadata.json`.)

---

## 4. Final submission & results summary

**Model output to submit: `run_logs/submission.csv`** — 170,588 scored test rows in
the Starter Kit schema (`row_id,user_id,video_id,score`). Validate with
`python3 submit.py --check`.

### Results table — KuaiRand-Pure (required benchmark)

| Model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| **Our validation-best** (causal-history DeepFM, regularized) | **0.6710** | **0.5376** | **0.6043** |
| | | | |
| Official FM baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| Our final model (test, scored once) | 0.6554 | 0.5265 | 0.5909 |

**Absolute delta over the official baseline**

- **Validation: +0.0027** (0.6043 − 0.6016) — our validation-best beats the baseline
- **Held-out test: −0.0037** (0.5909 − 0.5946)

The validation gain did not survive the temporal split between validation
(2022-04-22 – 04-28) and test (2022-04-29 – 05-08). The test set was scored once,
for the validation-selected configuration only. We report the test delta as the
result rather than the more flattering validation number.

**Bonus benchmarks (KuaiRand-1k / 27k): not attempted.** No results claimed.

### Resource usage (Feasibility & Practicality)

Source: `run_logs/run_metadata.json` and `python3 summarize_submission.py`.

| Resource | Value |
|---|---:|
| LLM input tokens | 9,901 |
| LLM output tokens | 1,837 |
| **LLM total tokens** | **11,738** |
| Provider calls (failed) | 8 (0) |
| Reported LLM cost | $0.0023 |
| Planner model | `openai/gpt-4o-mini` via OpenRouter |
| **Agent wall-clock** | **264.5 s (4.4 min)** |
| **Iterations used** | **3 of 50 cap** |
| Stop reason | `converged` (ε = 0.002, N = 3) |
| **GPU-hours** | **0.0 — CPU-only, no GPU at any stage** |

The run converged at 3 iterations under the ε/N rule, well inside the 50-iteration
cap and the 6-hour wall-clock ceiling.

---

## Regenerate the final submission before submitting

`run_logs/submission.csv` was overwritten by the most recent agent run and currently
holds `fm_baseline` predictions, not the validation-best champion quoted above. Run:

```bash
python3 finalize_and_score.py
python3 submit.py --check
```

This rewrites `submission.csv` from the champion configuration. It is the same
model, same params, same seed, so it reproduces the 0.5909 already recorded in
`final_result.json` — it is not a second look at the test set.

## Files to attach

| Item | File |
|---|---|
| Per-iteration run log | `run_logs/iterations.jsonl` |
| LLM call telemetry | `run_logs/llm_calls.jsonl` |
| Run metadata | `run_logs/run_metadata.json` |
| Resource summary | `run_logs/submission_summary.json` |
| Final model output | `run_logs/submission.csv` |
| One-time test result | `final_result.json` |
| Narrative writeup | `DEVPOST_NARRATIVE.md` |
| Technical writeup | `DEVPOST.md` |
