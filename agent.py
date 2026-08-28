import json
import os
import time
import traceback
import argparse
import numpy as np

from baseline import run_fm
from data import load
from features import build_tabular_features
from models import LightGBMModel, NumpyNNModel
from bpr import run_bpr
from listwise import run_listwise
from history_model import run_history_deepfm

try:
    import lightgbm as _lgb  # type: ignore
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False


class RunLogger:
    def __init__(self, out_dir="run_logs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "iterations.jsonl")

    def write(self, record):
        def sanitize(o):
            if isinstance(o, dict):
                return {k: sanitize(v) for k, v in o.items()}
            if isinstance(o, list):
                return [sanitize(v) for v in o]
            if isinstance(o, tuple):
                return tuple(sanitize(v) for v in o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o

        safe = sanitize(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")


def single_iteration(splits, params, logger):
    start = time.time()
    record = {
        "timestamp": time.time(),
        "hypothesis": params.get("hypothesis", "run model"),
        "code_diff": params.get("code_diff", ""),
        "params": {k: v for k, v in params.items() if k not in ("hypothesis", "code_diff")},
        "metrics": None,
        "error": None,
    }
    try:
        model_type = params.get("model", "fm")
        if model_type == "fm":
            res = run_fm(splits, k=params.get("k", 16), lr=params.get("lr", 0.001),
                         epochs=params.get("epochs", 40), seed=params.get("seed", 0), verbose=True,
                         evaluate_test=False)
            record["metrics"] = res
        elif model_type == "history_deepfm":
            record["metrics"] = run_history_deepfm(
                data_dir=params.get("data_dir", "./KuaiRand-Pure/data"),
                epochs=params.get("epochs", 8), lr=params.get("lr", 1e-3),
                emb_dim=params.get("emb_dim", 12), hidden=params.get("hidden", 96),
                batch_size=params.get("batch_size", 8192), patience=params.get("patience", 3),
                seed=params.get("seed", 0), verbose=True)
        elif model_type == "history_attention":
            record["metrics"] = run_history_deepfm(
                data_dir=params.get("data_dir", "./KuaiRand-Pure/data"),
                epochs=params.get("epochs", 8), lr=params.get("lr", 1e-3),
                emb_dim=params.get("emb_dim", 12), hidden=params.get("hidden", 96),
                batch_size=params.get("batch_size", 8192), patience=params.get("patience", 3),
                history_len=params.get("history_len", 20), sequence_attention=True,
                seed=params.get("seed", 0), verbose=True)
        elif model_type == "history_multitask":
            record["metrics"] = run_history_deepfm(
                data_dir=params.get("data_dir", "./KuaiRand-Pure/data"),
                epochs=params.get("epochs", 8), lr=params.get("lr", 1e-3),
                emb_dim=params.get("emb_dim", 12), hidden=params.get("hidden", 96),
                batch_size=params.get("batch_size", 8192), patience=params.get("patience", 3),
                multitask_click=True, click_weight=params.get("click_weight", 0.25),
                seed=params.get("seed", 0), verbose=True)
        elif model_type == "bpr":
            res = run_bpr(splits, k=params.get("k", 16), lr=params.get("lr", 0.001),
                           epochs=params.get("epochs", 40), l2=params.get("l2", 1e-6),
                           patience=params.get("patience", 4), seed=params.get("seed", 0), verbose=True)
            record["metrics"] = res
        elif model_type == "listwise":
            res = run_listwise(splits, k=params.get("k", 16), lr=params.get("lr", 0.001),
                                epochs=params.get("epochs", 40), users_per_step=params.get("users_per_step", 2048),
                                l2=params.get("l2", 1e-6), patience=params.get("patience", 4),
                                seed=params.get("seed", 0), verbose=True)
            record["metrics"] = res
        elif model_type == "lgb":
            tab = build_tabular_features(splits)
            Xtr, ytr, _ = tab['train']
            Xva, yva, uva = tab['valid']
            try:
                lgbm = LightGBMModel(params=params.get("lgb_params", None))
                lgbm.fit(Xtr, ytr, valid=(Xva, yva), num_round=params.get("num_round", 100))
                scores_va = lgbm.predict(Xva)
                from evaluate import evaluate
                res = {'valid': evaluate(uva, yva, scores_va)}
                record["metrics"] = res
            except Exception:
                record["error"] = traceback.format_exc()
        elif model_type == "nn":
            tab = build_tabular_features(splits)
            Xtr, ytr, _ = tab['train']
            Xva, yva, uva = tab['valid']
            try:
                nn = NumpyNNModel(input_dim=Xtr.shape[1], hidden=params.get("hidden", 64), lr=params.get("lr", 1e-3))
                nn.fit(Xtr, ytr, epochs=params.get("epochs", 10), bs=params.get("bs", 4096))
                scores_va = nn.predict(Xva)
                from evaluate import evaluate
                res = {'valid': evaluate(uva, yva, scores_va)}
                record["metrics"] = res
            except Exception:
                record["error"] = traceback.format_exc()
        else:
            record["error"] = f"unknown model {model_type}"
    except Exception:
        record["error"] = traceback.format_exc()
    record["duration_sec"] = time.time() - start
    logger.write(record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--out_dir", default="run_logs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    logger = RunLogger(a.out_dir)

    epsilon = 0.002
    N = 3
    stagnation = 0
    best = -1.0
    max_iters = 10
    iter_no = 0

    # NOTE: BPR and listwise-softmax were already tried manually (run_logs/iterations.jsonl
    # iter 9-15) and both underperformed the pointwise baseline across 7 configs -- see
    # README.md's "已实测：这三条没有收益" table. They are NOT re-added here as live proposals;
    # re-running a documented negative result wastes iteration budget, which is exactly the
    # mistake this list made with the embedding-dim proposal before. run_bpr/run_listwise
    # remain importable (and wired into single_iteration above) for one-off checks, e.g. if a
    # later feature change makes it worth re-testing.
    proposals = [
        {"model": "fm", "k": 16, "lr": 0.001, "epochs": a.epochs,
         "hypothesis": "Reproduce official FM baseline", "code_diff": "baseline hyperparams (k=16, lr=0.001)"},
        {"model": "history_deepfm", "epochs": 8, "lr": 0.001, "emb_dim": 12,
         "hidden": 96, "batch_size": 8192, "patience": 3,
         "hypothesis": "Causal user-history DeepFM with author/tag affinity and recency",
         "code_diff": "add causal train-history features + metadata DeepFM; validation only"},
        {"model": "history_multitask", "epochs": 8, "lr": 0.001, "emb_dim": 12,
         "hidden": 96, "batch_size": 8192, "patience": 3, "click_weight": 0.25,
         "hypothesis": "Use dense click feedback as an auxiliary representation-learning task",
         "code_diff": "add shared-trunk click head with 0.25 auxiliary BCE weight"},
 
        # {"model": "history_attention", "epochs": 8, "lr": 0.001, "emb_dim": 12,
        #  "hidden": 96, "batch_size": 8192, "patience": 3, "history_len": 20,
        #  "hypothesis": "Attend from candidate video to the user's recent positive history",
        #  "code_diff": "add DIN-style candidate-to-recent-long-view attention"},
         
    ]

    if LGB_AVAILABLE:
        proposals.append({"model": "lgb", "num_round": 50,
                           "hypothesis": "Tabular LightGBM with simple counts",
                           "code_diff": "add tabular features, train lgb"})
    else:
        print("LightGBM not available; skipping LGB proposal.")

    for prop in proposals:
        prop = {**prop, "data_dir": a.data_dir}
        iter_no += 1
        print(f"\n=== Iteration {iter_no} | model={prop.get('model')} ===")
        rec = single_iteration(splits, prop, logger)
        val = None
        if rec.get("metrics") and rec["metrics"].get("valid"):
            val = rec["metrics"]["valid"]["primary"]
        if val is None:
            print("  iteration produced no valid metrics")
            continue
        if val > best + epsilon:
            best = val
            stagnation = 0
            print(f"  new best valid primary {best:.6f}")
        else:
            stagnation += 1
            print(f"  no significant improvement (stagnation={stagnation})")
        if stagnation >= N:
            print(f"converged by stagnation (N={N}, eps={epsilon})")
            break
        if iter_no >= max_iters:
            print("reached max iters")
            break

    print("\nDone. Logs written to", logger.path)


if __name__ == '__main__':
    main()
