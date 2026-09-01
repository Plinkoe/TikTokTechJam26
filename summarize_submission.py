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

    input_tokens = sum(int(c.get("usage", {}).get("prompt_tokens", c.get("usage", {}).get("input_tokens", 0)) or 0) for c in calls)
    output_tokens = sum(int(c.get("usage", {}).get("completion_tokens", c.get("usage", {}).get("output_tokens", 0)) or 0) for c in calls)
    total_tokens = sum(int(c.get("usage", {}).get("total_tokens", 0) or 0) for c in calls)

    result = {
        "iterations": metadata.get("iterations"),
        "iteration_cap": metadata.get("max_iterations"),
        "manual_interventions": metadata.get("manual_interventions"),
        "wall_clock_sec": metadata.get("wall_clock_sec"),
        "llm_enabled": metadata.get("llm_enabled", False),
        "llm_calls": len(calls),
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_total_tokens": total_tokens,
        "final_result": final_result,
        "note": "Token values are provider-reported usage from llm_calls.jsonl; missing telemetry is not inferred.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
