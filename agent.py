"""CLI entry point for the autonomous KuaiRand research agent.

The runner loads a local .env (never committed), records LLM API usage in a
separate telemetry file, and leaves the benchmark controller responsible for
experiment execution and validation logging.
"""

import json
import os
import time
from functools import wraps

from agent_architecture import BenchmarkController, build_parser
from llm_planner import load_dotenv


def _install_llm_telemetry(controller: BenchmarkController) -> None:
    """Record provider-reported token usage without recording API credentials."""
    planner = controller.llm_planner
    original = planner.suggest_next_experiment
    telemetry_path = os.path.join(controller.out_dir, "llm_calls.jsonl")
    os.makedirs(controller.out_dir, exist_ok=True)

    @wraps(original)
    def wrapped(*args, **kwargs):
        started = time.time()
        proposal = original(*args, **kwargs)
        usage = dict(getattr(planner, "last_usage", {}) or {})
        record = {
            "timestamp": started,
            "duration_sec": time.time() - started,
            "provider": planner.config.provider,
            "model": planner.config.model,
            "success": proposal is not None,
            "error": planner.last_error,
            "usage": usage,
        }
        with open(telemetry_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return proposal

    planner.suggest_next_experiment = wrapped


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    args = build_parser().parse_args()
    controller = BenchmarkController(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        epsilon=args.epsilon,
        patience=args.patience,
        seed=args.seed,
    )
    controller.apply_epochs_override(getattr(args, "epochs", None))
    _install_llm_telemetry(controller)

    run_started = time.time()
    result = controller.run()
    submission_path = controller.finalize_submission(args.submission_path)

    metadata = {
        "started_at": run_started,
        "finished_at": time.time(),
        "wall_clock_sec": time.time() - run_started,
        "iterations": len(result["iterations"]),
        "max_iterations": args.max_iters,
        "manual_interventions": 0,
        "manual_interventions_definition": "No human changes or recovery actions during this unattended agent run.",
        "best_model": result["best_model"],
        "best_validation": result["best"],
        "llm_enabled": controller.llm_planner.enabled,
    }
    with open(os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n=== autonomous research agent ===")
    print(f"best validation model: {result['best_model']}")
    print(f"best validation metrics: {result['best']}")
    print(f"iterations: {len(result['iterations'])}")
    print(f"wall clock: {metadata['wall_clock_sec']:.2f}s")
    print(f"submission: {submission_path}")


if __name__ == "__main__":
    main()
