"""Protocol-compliant final scoring: run this ONCE per candidate, after
rescreen_champion.py's stricter dual-validation, multi-seed gate has already
confirmed the candidate beats FM on both the official and unbiased validation
slices. This script is the point where test labels get touched -- treat every
call as spending your one shot for that specific candidate.

Writes run_logs/final_result.json (moving any existing one to
run_logs/final_result.previous.json first, so a prior candidate's real test
score is never silently lost) and run_logs/submission.csv.

Do not run this repeatedly while iterating on the model. If you change the
model again after this, that is a NEW candidate and gets its own one-time
check -- it does not un-spend this one.
"""
import argparse
import json
import os
import shutil
import time

from agent_architecture import BenchmarkContext, HistoryDeepFMExperiment, FMExperiment, _sanitize
from data import load
from evaluate import evaluate
from rescreen_champion import CONFIGS, EPSILON
from submit import write_submission

DATA_DIR = "./KuaiRand-Pure/data"
RESCREEN_PATH = "run_logs/rescreen_results.json"


def check_dual_validation_gate(config_name):
    """Refuse to spend the one-time test check on a candidate that hasn't
    already cleared rescreen_champion.py's stricter gate (5 seeds, both the
    official and unbiased validation slices)."""
    if not os.path.exists(RESCREEN_PATH):
        raise RuntimeError(
            f"{RESCREEN_PATH} not found -- run rescreen_champion.py first so this "
            f"candidate has actually cleared the dual-validation, multi-seed gate."
        )
    with open(RESCREEN_PATH) as fh:
        results = json.load(fh)
    for key in ("fm_baseline", config_name):
        if key not in results or len(results[key]) < 5:
            raise RuntimeError(
                f"'{key}' has fewer than 5 seeds in {RESCREEN_PATH} -- "
                f"finish the rescreen sweep before finalizing."
            )
    import numpy as np
    fm_valid = np.mean([r["valid"]["primary"] for r in results["fm_baseline"]])
    fm_unbiased = np.mean([r["unbiased_valid"]["primary"] for r in results["fm_baseline"]])
    cand_valid = np.mean([r["valid"]["primary"] for r in results[config_name]])
    cand_unbiased = np.mean([r["unbiased_valid"]["primary"] for r in results[config_name]])
    passes = (cand_valid - fm_valid > EPSILON) and (cand_unbiased - fm_unbiased > EPSILON)
    print(f"dual-validation gate for '{config_name}': valid delta={cand_valid-fm_valid:+.4f}, "
          f"unbiased delta={cand_unbiased-fm_unbiased:+.4f}, PASSES={passes}")
    if not passes:
        raise RuntimeError(f"'{config_name}' does not pass the dual-validation gate; refusing to finalize.")


def main(config_name, seed=0):
    check_dual_validation_gate(config_name)
    params = dict(CONFIGS[config_name])
    params["seed"] = seed

    splits = load(DATA_DIR)
    context = BenchmarkContext(data_dir=DATA_DIR, out_dir="run_logs", splits=splits)

    print(f"\n=== validation gate (re-confirm '{config_name}' beats FM on valid, seed={seed}) ===")
    fm = FMExperiment()
    fm_valid = fm.run(context, fm.default_params())["valid"]
    print("FM valid:", fm_valid)

    champ = HistoryDeepFMExperiment()
    champ_valid = champ.run(context, params)["valid"]
    print(f"{config_name} valid:", champ_valid)
    assert champ_valid["primary"] > fm_valid["primary"], (
        f"'{config_name}' did not beat FM on valid at seed={seed}; aborting finalize"
    )

    print(f"\n=== finalizing '{config_name}': train on train+valid, score test ONCE ===")
    t0 = time.time()
    test_scores = champ.finalize(context, params)
    test_rows = splits["test"]
    test_users = [r[1] for r in test_rows]
    test_labels = [r[6] for r in test_rows]
    test_metrics = evaluate(test_users, test_labels, test_scores)
    print(f"{config_name} TEST (one-time):", test_metrics, f"({time.time()-t0:.1f}s)")

    write_submission("run_logs/submission.csv", test_rows, test_scores)

    with open("baseline_scores.json") as fh:
        baseline = json.load(fh)["scores"]["fm_official"]["test"]

    result = {
        "model": f"history_deepfm ({config_name})",
        "params": params,
        "valid": champ_valid,
        "test": test_metrics,
        "official_fm_baseline_test": baseline,
        "delta_test_primary": test_metrics["primary"] - baseline["primary"],
    }
    result = _sanitize(result)

    out_path = "run_logs/final_result.json"
    if os.path.exists(out_path):
        shutil.move(out_path, "run_logs/final_result.previous.json")
        print(f"(moved prior result to run_logs/final_result.previous.json)")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print("\n=== FINAL ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="champion_regularized", choices=list(CONFIGS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.config, args.seed)
