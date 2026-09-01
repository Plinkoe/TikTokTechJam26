# Da AI AI tuner — 3-minute video script

**Format:** screen recording + voiceover. No talking head needed.
**Pace:** ~145 words/minute. Narration below is ~420 words = 2:55. Do not add more.

> **BEFORE YOU HIT RECORD**
> 1. **Close your `.env` file and any editor tab showing it.** It contains a live
>    OpenRouter API key. If it appears on screen you must revoke the key.
> 2. Do not try to run a full 8-iteration run live — it takes far too long.
>    Record the terminal output you already have, or start a run and cut away.
> 3. Have these open in tabs, ready to switch to:
>    - terminal in `kuairand-starter-kit`
>    - `run_logs/iterations.jsonl`
>    - `llm_planner.py` (at `compile_feature_transform` in `llm_model_experiment.py`)
>    - `final_result.json`

---

## 0:00 – 0:20 — Hook

**ON SCREEN:** Title card, or just the repo open in your editor.

> Machine learning research is mostly not the modelling. It's the loop around
> it — form a hypothesis, run it, read the metric, decide what it means, pick
> the next move. That loop is where the days go.
>
> Da AI AI tuner is an autonomous research agent that runs that loop on the
> KuaiRand-Pure recommendation benchmark. Not autocomplete for a model class —
> it proposes experiments, trains them, reads its own validation feedback, and
> decides what to try next.

## 0:20 – 0:50 — Show it running

**ON SCREEN:** Terminal. `python3 agent.py --data_dir ./KuaiRand-Pure/data --max_iters 8`
Let the baseline gate print, then cut to a finished run's output.

> Every run starts by reproducing the official FM baseline and refuses to
> continue unless it matches the published score. If the pipeline is wrong, the
> run stops before it can generate confident nonsense on a broken foundation.
>
> Then it iterates. Each iteration it picks one of four moves — rewrite the
> model architecture, write a new feature transform, change the training
> recipe, or retune an existing experiment — based on its own history.

## 0:50 – 1:20 — The sandbox and the leakage guard

**ON SCREEN:** Scroll `compile_feature_transform` in `llm_model_experiment.py`.
Highlight the two invariant checks.

> The agent writes real code, so that code runs in a sandbox: restricted
> builtins, no file, network or process access, and a dry-run forward pass
> before any expensive training.
>
> The part I'd point at is this. A feature transform that normalises by a batch
> mean would inflate validation and then collapse on the hidden test set — and
> no metric would warn you. So before a transform touches data, we prove it's
> row-wise, using two invariants: subset consistency and permutation
> equivariance. Our first version only had the permutation check, and
> mean-centering passed it. A unit test caught that, not a run.

## 1:20 – 2:00 — The failure story

**ON SCREEN:** `run_logs/iterations.jsonl`, then `llm_calls.jsonl` showing
`finish_reason` and a `planner_failed` row.

> The hardest bug wasn't a crash. For several runs the agent printed
> "best validation model: fm_baseline" and exited cleanly. Nothing looked
> broken. In fact the planner was failing on every call, and the controller was
> quietly falling through to a hardcoded schedule. A run that did zero
> autonomous research was indistinguishable from a successful one.
>
> That changed the design. Every provider call now logs its raw response and
> finish reason. Every planner failure writes a row you can find. Deterministic
> errors get blocklisted after one attempt instead of reproducing the same
> traceback three times. A silent fallback is worse than a crash.

## 2:00 – 2:35 — Results, honestly

**ON SCREEN:** `final_result.json`, then the results table from `DEVPOST_NARRATIVE.md`.

> Results. Our strongest candidate — a regularised causal-history DeepFM — beat
> the baseline on validation, 0.6043 against 0.6015. On the held-out test set it
> scored 0.5909 against the baseline's 0.5946. A delta of minus 0.0037.
>
> The validation gain was real and it did not survive the temporal split. We
> scored the test set once and we're reporting that number rather than shopping
> for a better one. It's the most useful thing the project taught us: an agent
> that optimises validation alone will keep walking into this.

## 2:35 – 3:00 — Close

**ON SCREEN:** Back to the repo, or a simple end card.

> What's next is straightforward. A stronger planner model — planner tokens
> cost under a dollar per full run, so quality is the only constraint that
> matters. Autonomous ensembling. And scheduling by expected information gain
> instead of running experiments in sequence.
>
> The loop works, it's instrumented, and it tells you the truth about its own
> results. That was the hard part.

---

## If you are short on time

Cut section 0:50–1:20 (the sandbox) and stretch the failure story. The silent
fallback is your strongest differentiator on the "robust operation" criterion —
it shows judgement, not just implementation.

## What NOT to do

- Don't claim you beat the baseline. Your logs are in the repo and judges can check.
- Don't narrate code line by line. Show the file, say what it defends against.
- Don't spend 40 seconds on a title card. You have 180 seconds total.
