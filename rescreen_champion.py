"""Stricter re-screening for the causal-history DeepFM champion, per the
overfitting concern in results_summary.md: the original champion was picked
by comparing many variants against one fixed, biased validation split at a
single seed, which is exactly the setup that inflates apparent wins.

This script requires a candidate to beat the FM baseline on BOTH:
  (a) the official (biased) validation split, and
  (b) an unbiased random-exposure slice from the same date window
across 5 seeds (0-4), not just seed 0, before calling it a champion.

Test labels are never read here -- only train + the two validation slices.
Run this BEFORE finalize_and_score.py, and only run finalize_and_score.py
once, for whichever config (if any) survives this gate.

Resumable: re-running with the same out_path skips (seed, config) pairs
already recorded, so a long sweep can be split across several invocations.
"""
import json
import os
import time

import numpy as np

from agent_architecture import _sanitize, read_unbiased_valid_rows
from baseline import FM
from data import load, load_unbiased_valid, encode
from evaluate import evaluate as ev
from history_model import run_history_deepfm

DATA_DIR = "./KuaiRand-Pure/data"
SEEDS = [0, 1, 2, 3, 4]
EPSILON = 0.002  # same convergence threshold used elsewhere in this project

CONFIGS = {
    # The original, unregularized champion -- re-screened under the stricter gate.
    "champion_original": dict(emb_dim=12, hidden=96, dropout=0.1, weight_decay=1e-6),
    # Smaller capacity: fewer embedding dims / narrower hidden layer.
    "champion_small_capacity": dict(emb_dim=6, hidden=48, dropout=0.1, weight_decay=1e-6),
    # Same capacity, heavier regularization.
    "champion_regularized": dict(emb_dim=12, hidden=96, dropout=0.3, weight_decay=1e-4),
}


def load_results(out_path):
    if os.path.exists(out_path):
        with open(out_path) as fh:
            return json.load(fh)
    return {}


def save_results(results, out_path):
    with open(out_path, "w") as fh:
        json.dump(_sanitize(results), fh, indent=2)


def run_fm_once(splits, unbiased_valid_tuples, seed):
    fm_splits = dict(splits)
    fm_splits["unbiased_valid"] = unbiased_valid_tuples
    enc, dim = encode(fm_splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xub, yub, uub = enc["unbiased_valid"]
    m = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        va = ev(uva, yva, m.predict(Xva))
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4:
                break
    m.V, m.W, m.b = best_state
    return ev(uva, yva, m.predict(Xva)), ev(uub, yub, m.predict(Xub))


def main(config_names, out_path, seeds=None):
    seeds = seeds or SEEDS
    splits = load(DATA_DIR)
    unbiased_valid_tuples = load_unbiased_valid(DATA_DIR)
    unbiased_valid_rows = read_unbiased_valid_rows(DATA_DIR)
    print(f"unbiased valid slice: {len(unbiased_valid_rows)} rows "
          f"({len(set(r.user for r in unbiased_valid_rows))} users)")

    results = load_results(out_path)
    if results:
        print(f"resuming from {out_path}: {{k: len(v) for k, v in results.items()}} ="
              f" {[(k, len(v)) for k, v in results.items()]}")

    done_fm_seeds = {r["seed"] for r in results.get("fm_baseline", [])}
    for seed in seeds:
        if seed in done_fm_seeds:
            continue
        t0 = time.time()
        fm_valid, fm_unbiased = run_fm_once(splits, unbiased_valid_tuples, seed)
        results.setdefault("fm_baseline", []).append({
            "seed": seed, "valid": fm_valid, "unbiased_valid": fm_unbiased,
        })
        print(f"[seed {seed}] fm_baseline: valid={fm_valid['primary']:.4f} "
              f"unbiased={fm_unbiased['primary']:.4f} ({time.time()-t0:.1f}s)")
        save_results(results, out_path)

    for name in config_names:
        params = CONFIGS[name]
        done_seeds = {r["seed"] for r in results.get(name, [])}
        for seed in seeds:
            if seed in done_seeds:
                continue
            t0 = time.time()
            out = run_history_deepfm(
                data_dir=DATA_DIR, epochs=8, seed=seed, verbose=False,
                extra_eval_rows={"unbiased_valid": unbiased_valid_rows},
                **params,
            )
            results.setdefault(name, []).append({
                "seed": seed, "valid": out["valid"], "unbiased_valid": out["extra"]["unbiased_valid"],
            })
            print(f"[seed {seed}] {name}: valid={out['valid']['primary']:.4f} "
                  f"unbiased={out['extra']['unbiased_valid']['primary']:.4f} ({time.time()-t0:.1f}s)")
            save_results(results, out_path)

    if len(results.get("fm_baseline", [])) == len(SEEDS) and all(
        len(results.get(n, [])) == len(SEEDS) for n in CONFIGS
    ):
        summarize(results)
    else:
        print("\n(not all seeds/configs complete yet -- re-run to continue)")


def summarize(results):
    fm_valid = np.mean([r["valid"]["primary"] for r in results["fm_baseline"]])
    fm_unbiased = np.mean([r["unbiased_valid"]["primary"] for r in results["fm_baseline"]])
    print(f"\n=== FM baseline mean over {len(results['fm_baseline'])} seeds ===")
    print(f"  valid={fm_valid:.4f}  unbiased_valid={fm_unbiased:.4f}")
    print("\n=== candidates ===")
    for name, records in results.items():
        if name == "fm_baseline":
            continue
        valid_scores = [r["valid"]["primary"] for r in records]
        unbiased_scores = [r["unbiased_valid"]["primary"] for r in records]
        valid_mean, valid_std = np.mean(valid_scores), np.std(valid_scores)
        unbiased_mean, unbiased_std = np.mean(unbiased_scores), np.std(unbiased_scores)
        valid_wins = sum(1 for s in valid_scores if s > fm_valid)
        unbiased_wins = sum(1 for s in unbiased_scores if s > fm_unbiased)
        passes = (valid_mean - fm_valid > EPSILON) and (unbiased_mean - fm_unbiased > EPSILON)
        print(f"{name}:")
        print(f"  valid    mean={valid_mean:.4f} std={valid_std:.4f} delta={valid_mean-fm_valid:+.4f} "
              f"wins={valid_wins}/{len(valid_scores)}")
        print(f"  unbiased mean={unbiased_mean:.4f} std={unbiased_std:.4f} delta={unbiased_mean-fm_unbiased:+.4f} "
              f"wins={unbiased_wins}/{len(unbiased_scores)}")
        print(f"  PASSES dual-validation gate (delta > {EPSILON} on both): {passes}")


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
    main(names, "run_logs/rescreen_results.json")
