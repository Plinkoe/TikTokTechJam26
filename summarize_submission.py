"""Generate the submission-facing run/resource summary from local run artifacts.

This script never invents missing telemetry. If the LLM was not actually used,
LLM token totals remain null/zero and the report explicitly says so.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="run_logs")
    ap.add_argument("--out", default="run_logs/submission_summary.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    metadata = load_json(run_dir / "run_metadata.json", {}) or {}
    final_result = load_json(Path("final_result.json"), None)
    if final_result is None:
        final_result = load_json(run_dir / "final_result.json", None)

    calls = []
    calls_path = run_dir / "llm_calls.jsonl"
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                calls.append(json.loads(line))

    # llm_calls.jsonl holds two record kinds: one "provider_call" per HTTP
    # request, and one "planner_result" summary per planner invocation that
    # echoes the last call's usage. Counting both double-reports the tokens.
    # Records written before that split have no "kind" and are counted.
    billable = [c for c in calls if c.get("kind", "provider_call") == "provider_call"]
    planner_invocations = sum(1 for c in calls if c.get("kind") == "planner_result")

    def usage_sum(key, alt):
        return sum(int((c.get("usage") or {}).get(key, (c.get("usage") or {}).get(alt, 0)) or 0)
                   for c in billable)

    input_tokens = usage_sum("prompt_tokens", "input_tokens")
    output_tokens = usage_sum("completion_tokens", "output_tokens")
    total_tokens = sum(int((c.get("usage") or {}).get("total_tokens", 0) or 0) for c in billable)
    reported_cost = sum(float((c.get("usage") or {}).get("cost", 0) or 0) for c in billable)
    failed_calls = sum(1 for c in billable if c.get("success") is False)

    result = {
        "iterations": metadata.get("iterations"),
        "iteration_cap": metadata.get("max_iterations"),
        "manual_interventions": metadata.get("manual_interventions"),
        "wall_clock_sec": metadata.get("wall_clock_sec"),
        "llm_enabled": metadata.get("llm_enabled", False),
        "stop_reason": metadata.get("stop_reason"),
        "llm_provider_calls": len(billable),
        "llm_failed_calls": failed_calls,
        "llm_planner_invocations": planner_invocations,
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_total_tokens": total_tokens,
        "llm_reported_cost_usd": round(reported_cost, 6),
        "llm_model": metadata.get("llm_model"),
        "gpu_hours": 0.0,
        "gpu_note": "CPU-only run; no GPU was used at any stage.",
        "primary_delta_vs_official_baseline": metadata.get("primary_delta_vs_official_baseline"),
        "final_result": final_result,
        "note": "Token values are provider-reported usage from llm_calls.jsonl, counting one record per HTTP call; missing telemetry is not inferred.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
