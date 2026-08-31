# LLM-generated experiment mode

The repository now supports a stronger autonomy mode in which the research LLM writes the next experiment rather than selecting only from the pre-written experiment registry.

## Run

Set the provider credentials in `.env` (or the shell), for example:

```text
KUAI_LLM_ENABLED=true
KUAI_LLM_PROVIDER=openai
KUAI_LLM_MODEL=gpt-4o-mini
KUAI_LLM_API_KEY=...
KUAI_LLM_CODE_MODE=true
```

Then run the normal entry point:

```bash
python agent.py --data_dir ./KuaiRand-Pure/data --out_dir run_logs --max_iters 10
```

When `KUAI_LLM_CODE_MODE=true`, `agent.py` uses `AutonomousCodeAgent`.

## What happens each iteration

1. The official FM baseline is reproduced first.
2. The LLM receives the current validation trajectory, including failed attempts.
3. The LLM proposes a **new experiment and complete Python source code**.
4. `code_experiment.py` parses and syntax-checks the source.
5. The candidate is executed in a separate Python subprocess.
6. The subprocess receives a temporary data directory containing only:
   - train: 2022-04-08 through 2022-04-21
   - public validation: 2022-04-22 through 2022-04-28
7. Validation metrics are returned to the agent.
8. The result is logged and fed back to the LLM on the next iteration.
9. A failed candidate does not terminate the research loop; its traceback becomes part of the next research context.

## Autonomy boundary

The generated module must define:

```python
def run(train_csv: str, valid_csv: str, data_dir: str) -> dict:
    ...
```

and return a `valid` metrics object containing `primary`, `GAUC`, and `nDCG@5`.

Generated code is statically checked for obvious hidden-test access and dangerous process/network imports. It is then executed with a sandbox data directory. This is an engineering boundary, **not a security sandbox**; never run arbitrary generated code on a machine containing secrets.

## Research log

`run_logs/iterations.jsonl` contains the hypothesis, generated change description, validation metrics, status, traceback, and whether the candidate improved the current validation champion. `run_logs/generated_research_result.json` contains the final generated champion and its source code.

## Why this is different from the original agent

The original planner could only choose a registered experiment such as `history_deepfm` or `multitask_aux`. Code mode can instead invent a new feature transformation, loss, architecture, regularizer, sampler, or ensemble implementation on every iteration. The generated source is not pre-registered in the repository.
