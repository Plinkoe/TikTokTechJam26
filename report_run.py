"""Print the numbers the Devpost narrative needs, straight from run_logs/.

Usage:  python3 report_run.py [--run_dir run_logs]

Reads run_metadata.json and iterations.jsonl and prints paste-ready lines, so
the writeup is filled in from the logs rather than from memory.
"""

import argparse
import json
import os


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="run_logs")
    args = ap.parse_args()

    meta_path = os.path.join(args.run_dir, "run_metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} not found -- run agent.py first.")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    records = load_jsonl(os.path.join(args.run_dir, "iterations.jsonl"))
    # One row per experiment; a failed experiment can appear more than once.
    by_experiment = {}
    for record in records:
        name = record.get("experiment")
        if name and name not in by_experiment:
            by_experiment[name] = record

    scored = []
    for name, record in by_experiment.items():
        valid = (record.get("metrics") or {}).get("valid") or {}
        if "primary" in valid:
            scored.append((float(valid["primary"]), name, valid, record))
    scored.sort(reverse=True)

    official = meta.get("official_baseline_valid", {})
    delta = meta.get("primary_delta_vs_official_baseline")
    if not official or meta.get("stop_reason") is None:
        print("WARNING: this run_logs/ predates the multi-stage agent -- it has no\n"
              "         baseline delta, stop reason or stage breakdown. Delete it and\n"
              "         re-run agent.py before filling in the writeup.\n")

    print("=" * 72)
    print("PASTE INTO 'What we learned'  (replaces the [RUN] placeholder)")
    print("=" * 72)
    if scored:
        span = (f"{scored[-1][0]:.4f}-{scored[0][0]:.4f}" if len(scored) > 1
                else f"{scored[0][0]:.4f}")
        print(f"  our candidates land at {span} against a "
              f"{float(official.get('primary', 0)):.4f} baseline")
    else:
        print("  (no scored candidates in this run)")

    print()
    print("=" * 72)
    print("PASTE INTO 'Current status'")
    print("=" * 72)
    print(f"  best model            : {meta.get('best_model')}")
    best = meta.get("best_validation") or {}
    print(f"  best validation       : primary {float(best.get('primary', 0)):.4f} "
          f"(GAUC {float(best.get('GAUC', 0)):.4f}, nDCG@5 {float(best.get('nDCG@5', 0)):.4f})")
    print(f"  official baseline     : primary {float(official.get('primary', 0)):.4f}")
    if delta is not None:
        print(f"  delta vs baseline     : {float(delta):+.4f}  "
              f"({'BEATS baseline' if float(delta) > 0 else 'below baseline'})")
    print(f"  iterations            : {meta.get('iterations')} "
          f"(cap {meta.get('max_iterations')})")
    print(f"  stop reason           : {meta.get('stop_reason')}")
    rule = meta.get("convergence_rule") or {}
    print(f"  convergence rule      : eps={rule.get('epsilon')}, N={rule.get('N')}")
    print(f"  wall clock            : {float(meta.get('wall_clock_sec', 0)) / 60:.1f} min")
    print(f"  stages explored       : {meta.get('stages_explored')}")
    print(f"  planner model         : {meta.get('llm_model')}")
    if meta.get("blocked_experiments"):
        print(f"  routed around         : {list(meta['blocked_experiments'])}")

    print()
    print("=" * 72)
    print("PER-CANDIDATE TABLE (highest first)")
    print("=" * 72)
    print(f"  {'experiment':<22} {'family':<14} {'primary':>8} {'GAUC':>8} {'nDCG@5':>8}")
    for primary, name, valid, record in scored:
        print(f"  {name:<22} {str(record.get('family')):<14} "
              f"{primary:>8.4f} {float(valid.get('GAUC', 0)):>8.4f} "
              f"{float(valid.get('nDCG@5', 0)):>8.4f}")
    failed = [n for n, r in by_experiment.items() if r.get("status") != "ok"]
    if failed:
        print(f"\n  failed/blocked: {', '.join(failed)}")

    calls = load_jsonl(os.path.join(args.run_dir, "llm_calls.jsonl"))
    provider_calls = [c for c in calls if c.get("kind") == "provider_call"]
    if provider_calls:
        cost = sum(float((c.get("usage") or {}).get("cost", 0) or 0) for c in provider_calls)
        tokens = sum(int((c.get("usage") or {}).get("total_tokens", 0) or 0) for c in provider_calls)
        ok = sum(1 for c in provider_calls if c.get("success"))
        print()
        print("=" * 72)
        print("PLANNER TELEMETRY (for the Feasibility section)")
        print("=" * 72)
        print(f"  provider calls        : {len(provider_calls)} ({ok} succeeded)")
        print(f"  total tokens          : {tokens:,}")
        print(f"  reported cost         : ${cost:.4f}")


if __name__ == "__main__":
    main()
