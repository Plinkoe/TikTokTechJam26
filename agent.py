"""CLI entry point for the autonomous KuaiRand research agent.

The runner loads a local .env (never committed), records LLM API usage in a
separate telemetry file, and leaves the benchmark controller responsible for
experiment execution and validation logging.
"""

import json
import os
import time
from functools import wraps

import numpy as np

from agent_architecture import BenchmarkController, build_parser
from llm_planner import load_dotenv


def _json_safe(value):
    """Convert common NumPy scalar/container values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _install_llm_telemetry(controller: BenchmarkController) -> None:
    """Record provider-reported usage without recording API credentials.

    Logs one record per HTTP call to the provider (via the planner's
    call_log_hook) plus one summary record per planner invocation. The old
    version only logged the latter, which hid retries and threw away the raw
    response text -- the only evidence that explains a parse failure.
    """
    planner = controller.llm_planner
    original = planner.suggest_next_experiment
    telemetry_path = os.path.join(controller.out_dir, "llm_calls.jsonl")
    os.makedirs(controller.out_dir, exist_ok=True)

    def _append(record):
        with open(telemetry_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def _log_provider_call(record):
        record["kind"] = "provider_call"
        _append(record)

    planner.call_log_hook = _log_provider_call

    @wraps(original)
    def wrapped(*args, **kwargs):
        started = time.time()
        proposal = original(*args, **kwargs)
        record = {
            "kind": "planner_result",
            "timestamp": started,
            "duration_sec": time.time() - started,
            "provider": planner.config.provider,
            "model": planner.config.model,
            "force_code": planner.config.force_code,
            "success": proposal is not None,
            "mode": getattr(proposal, "mode", None),
            "family": getattr(proposal, "family", None),
            "code_chars": len(getattr(proposal, "code", "") or ""),
            "error": planner.last_error,
            "finish_reason": planner.last_finish_reason,
            "usage": dict(getattr(planner, "last_usage", {}) or {}),
        }
        _append(record)
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
        wall_clock_limit=args.wall_clock_hours * 3600.0,
    )
    controller.apply_epochs_override(getattr(args, "epochs", None))
    _install_llm_telemetry(controller)

    run_started = time.time()
    result = controller.run()
    submission_path = controller.finalize_submission(args.submission_path)

    # Spec 2.3 / judging: report the delta against the OFFICIAL baseline, not
    # against whatever this run happened to reproduce.
    with open(os.path.join(os.path.dirname(__file__), "baseline_scores.json"), "r", encoding="utf-8") as fh:
        official = json.load(fh)["scores"]["fm_official"]["valid"]
    best_primary = float((result["best"] or {}).get("primary", float("nan")))
    stages = {}
    for record in result["iterations"]:
        family = record.get("family") or "unknown"
        stages[family] = stages.get(family, 0) + 1

    metadata = {
        "started_at": run_started,
        "finished_at": time.time(),
        "wall_clock_sec": time.time() - run_started,
        "iterations": len(result["iterations"]),
        "max_iterations": args.max_iters,
        "stop_reason": result.get("stop_reason"),
        "convergence_rule": {"epsilon": args.epsilon, "N": args.patience},
        "manual_interventions": 0,
        "manual_interventions_definition": "No human changes or recovery actions during this unattended agent run.",
        "best_model": result["best_model"],
        "best_validation": result["best"],
        "official_baseline_valid": official,
        "primary_delta_vs_official_baseline": best_primary - float(official["primary"]),
        "beat_official_baseline": best_primary > float(official["primary"]),
        "stages_explored": stages,
        "blocked_experiments": result.get("blocked", {}),
        "llm_enabled": controller.llm_planner.enabled,
        "llm_model": controller.llm_planner.config.model,
    }
    with open(os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metadata), fh, indent=2)

    delta = metadata["primary_delta_vs_official_baseline"]
    print("\n=== autonomous research agent ===")
    print(f"best validation model: {result['best_model']}")
    print(f"best validation metrics: {result['best']}")
    print(f"official baseline primary: {official['primary']:.4f}   "
          f"delta: {delta:+.4f}   "
          f"{'BEATS baseline' if delta > 0 else 'below baseline'}")
    print(f"iterations: {len(result['iterations'])}  stop reason: {result.get('stop_reason')}")
    print(f"stages explored: {stages or '{}'}")
    if result.get("blocked"):
        print(f"routed around {len(result['blocked'])} failing experiment(s): "
              f"{', '.join(result['blocked'])}")
    print(f"wall clock: {metadata['wall_clock_sec']:.2f}s")
    print(f"submission: {submission_path}")


if __name__ == "__main__":
    main()
