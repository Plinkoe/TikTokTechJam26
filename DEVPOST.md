# TikTok TechJam 2026 — Devpost Submission Draft

## Project overview

This project is an autonomous ML research agent for the KuaiRand-Pure recommendation benchmark. Instead of requiring a human to manually decide every model experiment, the agent maintains an experiment history, asks an LLM to propose the next research move, executes the proposal against a fixed validation protocol, records the hypothesis/code/parameters/metrics/errors, and keeps the best validation candidate.

The design deliberately separates **research decisions** from **benchmark correctness**. Data loading, causal feature construction, the training/evaluation loop, and GAUC/nDCG@5 definitions remain fixed. In LLM-code mode, the LLM is allowed to author only a `CandidateModel` architecture. Generated source is checked against a restricted execution environment and a dummy forward pass before expensive training. Failed candidates are logged and their error is fed back into the next planning prompt.

## How the solution addresses the problem

The agent implements an iterative research loop:

1. Establish the official FM baseline as a validation gate.
2. Maintain a structured history of previous experiments, validation metrics, hypotheses, code changes, and failures.
3. Ask the LLM to either tune a registered experiment or write a new model architecture.
4. Validate generated model code before training.
5. Train only on the development data and evaluate on the fixed validation split.
6. Record every iteration in `run_logs/iterations.jsonl`.
7. Save the validation-best checkpoint/configuration.
8. Generate the final submission only from the validation-selected configuration.
9. Score the held-out test set only as a final, separate evaluation step.

The repository also includes pre-LLM research experiments covering causal user-history modeling, multitask learning, tabular models, ranking losses, and blending. These experiments informed the search space and provide reproducible baselines for the autonomous loop.

## Development tools

- Python 3.9+
- VS Code / local terminal workflow
- Git and GitHub
- NumPy
- PyTorch
- Local CSV-based KuaiRand-Pure dataset

A GPU is not required by the implementation; CPU execution is supported.

## APIs used

The autonomous planner uses an OpenAI-compatible Chat Completions API. The provider/model are configurable through environment variables:

- `KUAI_LLM_PROVIDER` — `openai`, `openrouter`, or `azure`
- `KUAI_LLM_MODEL`
- `KUAI_LLM_API_KEY` or `OPENAI_API_KEY`
- `KUAI_LLM_BASE_URL` when required by the provider

API credentials are never committed to the repository. A local `.env` may be used and is ignored by Git.

## Libraries and frameworks

- **PyTorch** — neural model training and generated `CandidateModel` architectures.
- **NumPy** — data processing, FM implementation, feature arrays, ranking utilities and reproducible random seeds.
- **Python standard library** — CSV/JSON handling, HTTP API calls, timing, logging and sandbox checks.

## Dataset and assets

The primary benchmark is **KuaiRand-Pure**, using its supplied user/video features and interaction logs. The official task is within-user ranking with `long_view` as the relevance label and GAUC / nDCG@5 as the required metrics; the primary score is their average.

The repository contains the KuaiRand-Pure data layout required by the starter kit. The randomly exposed KuaiRand log is also used as an additional validation-only screen in the research analysis; it does not use the held-out test labels.

## Autonomy and safety of generated code

LLM-authored models must define exactly one `CandidateModel(torch.nn.Module)` with a fixed constructor and forward interface. The generated source is executed with a restricted namespace and rejected if it contains file/network/process/environment access patterns. A dummy forward pass catches common shape/interface errors before real training.

The agent logs recovery events rather than silently treating failures as successes. Provider token usage is recorded from the API response in `run_logs/llm_calls.jsonl`, while run wall-clock time and iteration count are recorded in `run_logs/run_metadata.json`.

## Results

The currently committed research results do **not** show an improvement over the official FM test baseline. The latest one-time checked candidate was causal-history DeepFM with regularization:

| Model | GAUC | nDCG@5 | Primary | Delta vs FM |
|---|---:|---:|---:|---:|
| Official FM baseline | 0.6610 | 0.5282 | **0.5946** | — |
| Causal-history DeepFM, regularized | 0.6554 | 0.5265 | 0.5909 | -0.0037 |

Therefore the honest recommendation from the current completed research is to submit the official FM baseline rather than claim that the experimental model improves the benchmark.

**Important:** these numbers pre-date a verified live LLM-authored-code run. They must not be presented as results produced by the LLM-code mode.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the official baseline:

```bash
python3 baseline.py --model fm
```

Enable the LLM planner locally by creating an uncommitted `.env` containing at least:

```text
KUAI_LLM_ENABLED=true
KUAI_LLM_PROVIDER=openai
KUAI_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=YOUR_KEY
```

Then run a short smoke test first:

```bash
python3 agent.py --data_dir ./KuaiRand-Pure/data --out_dir run_logs --max_iters 1 --epochs 1
```

For a real research run, remove `--epochs 1` and increase `--max_iters` up to the challenge's 50-iteration cap.

After the run, generate resource telemetry:

```bash
python3 summarize_submission.py --run_dir run_logs
```

Only after the validation run has converged should the held-out test result be generated using the repository's finalization/scoring workflow. Do not repeatedly evaluate candidates on the test set.

## Run logs

Each iteration should contain:

- hypothesis
- experiment family/name
- code diff or generated architecture
- parameters
- validation GAUC
- validation nDCG@5
- primary validation score
- error/recovery information, if any
- status

`run_logs/llm_calls.jsonl` contains provider-reported token usage without API credentials. `run_logs/run_metadata.json` records wall-clock duration, iteration count and manual-intervention count.

## Limitations and future improvements

1. The LLM planner currently uses a JSON API interface and a compact history window; a stronger research-memory representation could improve long-horizon experiment selection.
2. Generated code is intentionally constrained to the model architecture, so the LLM cannot autonomously rewrite the full data pipeline or training algorithm. This trades research freedom for reproducibility and benchmark safety.
3. The current completed experiments suggest temporal distribution shift between validation and test. A future agent should explicitly optimize validation splits for temporal robustness rather than repeatedly querying the held-out test set.
4. More efficient experiment scheduling could allocate compute according to uncertainty and expected information gain instead of simple sequential iteration.
5. The project needs a completed live LLM-enabled run before claiming LLM token consumption, LLM-authored iterations, or final autonomy statistics for the submission.

## Team contributions

- **Sean Hoe / repository owner:** agent architecture, benchmark integration, experiment implementation, LLM-code generation path, validation/reproducibility tooling and submission packaging.

If the project has additional team members, replace this section with the agreed contribution breakdown before submitting to Devpost.
