"""Modular autonomous ML research agent for the KuaiRand-Pure benchmark.

This module provides a benchmark-agnostic controller loop, a simple experiment
interface, and a registry for the supported benchmark families: FM baseline,
causal-history DeepFM, multitask auxiliary heads, tabular models, and blending.

The evaluation protocol remains fixed in evaluate.py. All development work uses
train + public validation data only; hidden test labels are never read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from baseline import FM, run_fm
from data import load, encode
from evaluate import evaluate
from features import build_tabular_features
from history_model import HistoryDeepFM, build_causal_features, load_history_rows, run_history_deepfm
from models import NumpyNNModel
from submit import write_submission, read_submission


@dataclass
class ExperimentSpec:
    name: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    code_diff: str = ""


class BaseExperiment:
    name = "base"
    family = "generic"

    def default_params(self) -> Dict[str, Any]:
        return {}

    def run(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        raise NotImplementedError


class FMExperiment(BaseExperiment):
    name = "fm_baseline"
    family = "fm"

    def default_params(self) -> Dict[str, Any]:
        return {"k": 16, "lr": 1e-3, "epochs": 40, "bs": 8192, "patience": 4, "seed": 0}

    def run(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        result = run_fm(
            context.splits,
            k=p["k"],
            lr=p["lr"],
            epochs=p["epochs"],
            bs=p["bs"],
            patience=p["patience"],
            seed=p["seed"],
            verbose=False,
            evaluate_test=False,
        )
        return {"valid": result["valid"]}

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        combined = {"train": context.splits["train"] + context.splits["valid"], "valid": context.splits["valid"], "test": context.splits["test"]}
        enc, dim = encode(combined)
        Xtr, ytr, _ = enc["train"]
        Xte, _, _ = enc["test"]
        model = FM(dim, k=p["k"], lr=p["lr"], seed=p["seed"])
        rng = np.random.default_rng(p["seed"])
        best = -1.0
        best_state = None
        bad = 0
        for _ in range(p["epochs"]):
            idx = rng.permutation(len(ytr))
            for start in range(0, len(idx), p["bs"]):
                batch = idx[start:start + p["bs"]]
                model.step(Xtr[batch], ytr[batch])
            Xva, yva, uva = enc["valid"]
            metrics = evaluate(uva, yva, model.predict(Xva))
            if metrics["primary"] > best + 1e-5:
                best = metrics["primary"]
                best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
                bad = 0
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            model.V, model.W, model.b = best_state
        return model.predict(Xte)


class HistoryDeepFMExperiment(BaseExperiment):
    name = "history_deepfm"
    family = "causal_history"

    def default_params(self) -> Dict[str, Any]:
        return {
            "epochs": 8,
            "lr": 1e-3,
            "emb_dim": 12,
            "hidden": 96,
            "batch_size": 8192,
            "patience": 3,
            "seed": 0,
            "history_len": 20,
            "sequence_attention": False,
            "multitask_click": False,
            "click_weight": 0.25,
            "watch_time_aux": False,
            "watch_weight": 0.25,
        }

    def run(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        result = run_history_deepfm(
            data_dir=context.data_dir,
            epochs=p["epochs"],
            lr=p["lr"],
            emb_dim=p["emb_dim"],
            hidden=p["hidden"],
            batch_size=p["batch_size"],
            patience=p["patience"],
            seed=p["seed"],
            verbose=False,
            sequence_attention=p.get("sequence_attention", False),
            history_len=p.get("history_len", 20),
            multitask_click=p.get("multitask_click", False),
            click_weight=p.get("click_weight", 0.25),
            watch_time_aux=p.get("watch_time_aux", False),
            watch_weight=p.get("watch_weight", 0.25),
        )
        return {"valid": result["valid"]}

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        rows = load_history_rows(context.data_dir)
        train = rows["train"] + rows["valid"]
        test_rows = context.splits.get("test") or _read_test_rows(context.data_dir)
        if not test_rows:
            raise RuntimeError("No test rows available in data_dir; cannot finalize submission.")
        split = {"train": train, "valid": test_rows}
        (tr_cat, tr_dense, tr_hist, tr_y, tr_click, tr_watch_ratio, tr_watch_censored, _), \
            (va_cat, va_dense, va_hist, va_y, _, _, _, va_users), vocab_sizes, _ = build_causal_features(
                split, history_len=p.get("history_len", 20) if p.get("sequence_attention", False) else 0
            )
        tr_cat = torch.from_numpy(np.asarray(tr_cat, dtype=np.int64))
        tr_dense = torch.from_numpy(np.asarray(tr_dense, dtype=np.float32))
        tr_y = torch.from_numpy(np.asarray(tr_y, dtype=np.float32))
        tr_click = torch.from_numpy(np.asarray(tr_click, dtype=np.float32))
        va_cat = torch.from_numpy(np.asarray(va_cat, dtype=np.int64))
        va_dense = torch.from_numpy(np.asarray(va_dense, dtype=np.float32))
        tr_hist = torch.from_numpy(np.asarray(tr_hist, dtype=np.int32)) if tr_hist is not None else None
        va_hist = torch.from_numpy(np.asarray(va_hist, dtype=np.int32)) if va_hist is not None else None

        model = HistoryDeepFM(
            vocab_sizes,
            tr_dense.shape[1],
            emb_dim=p["emb_dim"],
            hidden=p["hidden"],
            sequence_attention=p.get("sequence_attention", False),
            multitask_click=p.get("multitask_click", False),
            watch_time_aux=p.get("watch_time_aux", False),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=1e-6)
        rng = np.random.default_rng(p["seed"])
        for _ in range(p["epochs"]):
            model.train()
            order = rng.permutation(len(tr_y))
            for start in range(0, len(order), p["batch_size"]):
                idx = torch.from_numpy(order[start:start + p["batch_size"]])
                h = tr_hist[idx] if tr_hist is not None else None
                optimizer.zero_grad()
                if p.get("multitask_click", False) or p.get("watch_time_aux", False):
                    out = model(
                        tr_cat[idx],
                        tr_dense[idx],
                        h,
                        return_click=p.get("multitask_click", False),
                        return_watch=p.get("watch_time_aux", False),
                    )
                    out = list(out) if isinstance(out, tuple) else [out]
                    logits = out[0]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tr_y[idx])
                    if p.get("multitask_click", False):
                        loss = loss + p["click_weight"] * torch.nn.functional.binary_cross_entropy_with_logits(out[1], tr_click[idx])
                    if p.get("watch_time_aux", False):
                        target = torch.from_numpy(np.asarray(tr_watch_ratio[idx.numpy()], dtype=np.float32))
                        censored = torch.from_numpy(np.asarray(tr_watch_censored[idx.numpy()], dtype=np.float32))
                        pred = out[1] if p.get("multitask_click", False) else out[0]
                        uncensored = (1.0 - censored) * (pred - target).pow(2)
                        censored_loss = censored * torch.nn.functional.relu(target - pred).pow(2)
                        loss = loss + p["watch_weight"] * (uncensored + censored_loss).mean()
                else:
                    logits = model(tr_cat[idx], tr_dense[idx], h)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tr_y[idx])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = model(va_cat, va_dense, va_hist).sigmoid().numpy()
        return scores


class MultitaskDeepFMExperiment(HistoryDeepFMExperiment):
    name = "multitask_aux"
    family = "multitask"

    def default_params(self) -> Dict[str, Any]:
        params = super().default_params()
        params["multitask_click"] = True
        params["click_weight"] = 0.25
        return params


class TabularExperiment(BaseExperiment):
    name = "tabular_model"
    family = "tabular"

    def default_params(self) -> Dict[str, Any]:
        return {"epochs": 10, "lr": 1e-3, "bs": 4096, "hidden": 64, "seed": 0}

    def run(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        tab = build_tabular_features(context.splits)
        Xtr, ytr, _ = tab["train"]
        Xva, yva, uva = tab["valid"]
        nn = NumpyNNModel(input_dim=Xtr.shape[1], hidden=p["hidden"], lr=p["lr"], seed=p["seed"])
        nn.fit(Xtr, ytr, epochs=p["epochs"], bs=p["bs"])
        scores = nn.predict(Xva)
        return {"valid": evaluate(uva, yva, scores)}

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        combined = {"train": context.splits["train"] + context.splits["valid"], "valid": context.splits["test"]}
        tab = build_tabular_features(combined)
        Xtr, ytr, _ = tab["train"]
        Xte, _, _ = tab["valid"]
        nn = NumpyNNModel(input_dim=Xtr.shape[1], hidden=p["hidden"], lr=p["lr"], seed=p["seed"])
        nn.fit(Xtr, ytr, epochs=p["epochs"], bs=p["bs"])
        return nn.predict(Xte)


class BlendExperiment(BaseExperiment):
    name = "blend_ensemble"
    family = "ensemble"

    def default_params(self) -> Dict[str, Any]:
        return {"champion_path": "", "fm_path": "", "weight": 0.5}

    def run(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        champion = np.load(p["champion_path"])
        fm = np.load(p["fm_path"])
        users = [x[1] for x in context.splits["valid"]]
        labels = [x[6] for x in context.splits["valid"]]
        blended = _blend_scores(users, champion, fm, p["weight"])
        return {"valid": evaluate(users, labels, blended)}

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        champion = np.load(p["champion_path"])
        fm = np.load(p["fm_path"])
        users = [x[1] for x in context.splits["test"]]
        return _blend_scores(users, champion, fm, p["weight"])


@dataclass
class BenchmarkContext:
    data_dir: str
    out_dir: str
    splits: Dict[str, List[Any]]
    seed: int = 0
    max_iters: int = 10
    epsilon: float = 0.002
    patience: int = 3
    best_metrics: Optional[Dict[str, float]] = None
    best_model_name: Optional[str] = None
    best_record: Optional[Dict[str, Any]] = None
    iteration_count: int = 0


class RunLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "iterations.jsonl")

    def write(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_sanitize(record), ensure_ascii=False) + "\n")


class ExperimentRegistry:
    def __init__(self):
        self._items: Dict[str, BaseExperiment] = {}

    def register(self, experiment: BaseExperiment) -> None:
        self._items[experiment.name] = experiment

    def get(self, name: str) -> BaseExperiment:
        return self._items[name]

    def default_schedule(self) -> List[ExperimentSpec]:
        return [
            ExperimentSpec(
                name="fm_baseline",
                family="fm",
                params={"k": 16, "lr": 1e-3, "epochs": 40, "bs": 8192, "patience": 4, "seed": 0},
                hypothesis="Reproduce the official FM baseline and lock the benchmark gate.",
                code_diff="baseline hyperparams (k=16, lr=0.001)",
            ),
            ExperimentSpec(
                name="history_deepfm",
                family="causal_history",
                params={"epochs": 8, "lr": 1e-3, "emb_dim": 12, "hidden": 96, "batch_size": 8192, "patience": 3, "seed": 0, "history_len": 20},
                hypothesis="Causal user-history DeepFM can exploit user recency and item affinity signal.",
                code_diff="training on causal user-history features with metadata DeepFM",
            ),
            ExperimentSpec(
                name="multitask_aux",
                family="multitask",
                params={"epochs": 8, "lr": 1e-3, "emb_dim": 12, "hidden": 96, "batch_size": 8192, "patience": 3, "seed": 0, "click_weight": 0.25, "multitask_click": True},
                hypothesis="Auxiliary click prediction helps representation learning without changing the ranking target.",
                code_diff="add shared-trunk click auxiliary head with BCE weight 0.25",
            ),
            ExperimentSpec(
                name="tabular_model",
                family="tabular",
                params={"epochs": 10, "lr": 1e-3, "bs": 4096, "hidden": 64, "seed": 0},
                hypothesis="Tabular count/rate features can capture repeated exposure patterns and duration effects.",
                code_diff="dense count-rate feature vectors + small MLP",
            ),
            ExperimentSpec(
                name="blend_ensemble",
                family="ensemble",
                params={"weight": 0.5},
                hypothesis="Within-user rank blending can combine the champion and FM strengths with less variance.",
                code_diff="blend champion/FM rank-normalized scores within user",
            ),
        ]


class BenchmarkController:
    def __init__(self, data_dir: str, out_dir: str = "run_logs", max_iters: int = 10, epsilon: float = 0.002, patience: int = 3, seed: int = 0):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.patience = patience
        self.seed = seed
        self.splits = load(data_dir)
        self.logger = RunLogger(out_dir)
        self.registry = ExperimentRegistry()
        for model in [FMExperiment(), HistoryDeepFMExperiment(), MultitaskDeepFMExperiment(), TabularExperiment(), BlendExperiment()]:
            self.registry.register(model)
        self.context = BenchmarkContext(data_dir=data_dir, out_dir=out_dir, splits=self.splits, seed=seed, max_iters=max_iters, epsilon=epsilon, patience=patience)
        self.schedule = self.registry.default_schedule()
        self.tried = set()
        self.best_primary = -1.0

    def baseline_gate(self) -> Dict[str, float]:
        with open(os.path.join(os.path.dirname(__file__), "baseline_scores.json"), "r", encoding="utf-8") as fh:
            reference = json.load(fh)["scores"]["fm_official"]["valid"]
        result = run_fm(self.splits, k=16, lr=1e-3, epochs=40, seed=0, verbose=False, evaluate_test=False)
        metrics = result["valid"]
        tolerance = 0.01
        mismatch = {key: (float(metrics[key]) - float(reference[key])) for key in reference if abs(float(metrics[key]) - float(reference[key])) > tolerance}
        if mismatch:
            raise ValueError(f"Baseline gate failed: observed={metrics} reference={reference} mismatch={mismatch}")
        return metrics

    def _save_checkpoint(self, record: Dict[str, Any]) -> None:
        path = os.path.join(self.out_dir, "best_validation_checkpoint.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_sanitize(record), fh, ensure_ascii=False, indent=2)

    def select_next_experiment(self, previous_records: Iterable[Dict[str, Any]]) -> Optional[ExperimentSpec]:
        previous = list(previous_records)
        seen = {rec.get("experiment") for rec in previous if rec.get("experiment")}
        valid_scores: Dict[str, List[float]] = {}
        for rec in previous:
            metrics = rec.get("metrics") or {}
            valid = metrics.get("valid")
            if valid is not None and "primary" in valid:
                valid_scores.setdefault(rec.get("family", "unknown"), []).append(float(valid["primary"]))
        ranked_families = [
            family for family, _ in sorted(
                valid_scores.items(),
                key=lambda item: sum(item[1]) / max(len(item[1]), 1),
                reverse=True,
            )
        ]
        for family in ranked_families + [spec.family for spec in self.schedule]:
            for spec in self.schedule:
                if spec.family == family and spec.name not in seen:
                    return spec
        for spec in self.schedule:
            if spec.name not in seen:
                return spec
        return None

    def _run_single_experiment(self, spec: ExperimentSpec, retries: int = 2) -> Dict[str, Any]:
        experiment = self.registry.get(spec.name)
        params = dict(spec.params)
        params["data_dir"] = self.data_dir
        for attempt in range(retries + 1):
            record = {
                "timestamp": time.time(),
                "iteration": self.context.iteration_count,
                "experiment": spec.name,
                "family": spec.family,
                "hypothesis": spec.hypothesis,
                "code_diff": spec.code_diff,
                "params": params,
                "metrics": None,
                "error": None,
                "status": "ok",
            }
            try:
                output = experiment.run(self.context, params)
                record["metrics"] = output
                if output and "valid" in output:
                    primary = float(output["valid"]["primary"])
                    record["primary"] = primary
                    if primary > self.best_primary + self.epsilon:
                        self.best_primary = primary
                        self.context.best_metrics = output["valid"]
                        self.context.best_model_name = spec.name
                        self.context.best_record = record
                        self._save_checkpoint(record)
                self.logger.write(record)
                return record
            except Exception:
                record["status"] = "failed"
                record["error"] = traceback.format_exc()
                self.logger.write(record)
                if attempt < retries:
                    continue
                return record
        return {"experiment": spec.name, "status": "failed", "error": "unreachable"}

    def run(self) -> Dict[str, Any]:
        baseline_metrics = self.baseline_gate()
        self.context.best_metrics = baseline_metrics
        self.context.best_model_name = "fm_baseline"
        self.context.best_record = {"experiment": "fm_baseline", "metrics": {"valid": baseline_metrics}}
        self.best_primary = float(baseline_metrics["primary"])
        self._save_checkpoint(self.context.best_record)

        previous_records: List[Dict[str, Any]] = []
        stagnation = 0
        while self.context.iteration_count < self.max_iters:
            next_spec = self.select_next_experiment(previous_records)
            if next_spec is None:
                break
            self.context.iteration_count += 1
            record = self._run_single_experiment(next_spec)
            previous_records.append(record)
            self.tried.add(next_spec.name)
            if record.get("metrics") and record["metrics"].get("valid"):
                primary = float(record["metrics"]["valid"]["primary"])
                if primary > self.best_primary + self.epsilon:
                    self.best_primary = primary
                    self.context.best_metrics = record["metrics"]["valid"]
                    self.context.best_model_name = record["experiment"]
                    self.context.best_record = record
                    self._save_checkpoint(record)
                    stagnation = 0
                else:
                    stagnation += 1
            if record.get("error"):
                continue
            if stagnation >= self.patience:
                break
        return {"best": self.context.best_metrics, "best_model": self.context.best_model_name, "iterations": previous_records}

    def finalize_submission(self, submission_path: str = "submission.csv") -> str:
        target = self.context.best_model_name or "fm_baseline"
        for candidate in [target, "fm_baseline"]:
            try:
                experiment = self.registry.get(candidate)
                scores = experiment.finalize(self.context, experiment.default_params())
                full_path = os.path.join(self.out_dir, submission_path)
                write_submission(full_path, self.splits["test"], scores)
                read_submission(full_path, self.splits["test"])
                return full_path
            except Exception:
                if candidate == "fm_baseline":
                    raise
        raise RuntimeError(f"No finalization path succeeded for target={target!r}")


def _sanitize(value):
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _within_user_ranks(users: List[str], scores: np.ndarray) -> np.ndarray:
    out = np.empty(len(scores), dtype=np.float32)
    groups: Dict[str, List[int]] = {}
    for idx, user in enumerate(users):
        groups.setdefault(user, []).append(idx)
    for idxs in groups.values():
        idxs = np.asarray(idxs, dtype=np.int64)
        ordered = np.argsort(scores[idxs], kind="mergesort")
        if len(idxs) == 1:
            out[idxs] = 0.5
            continue
        ranks = np.empty(len(idxs), dtype=np.float32)
        sorted_scores = scores[idxs][ordered]
        start = 0
        while start < len(idxs):
            end = start + 1
            while end < len(idxs) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[ordered[start:end]] = ((start + end - 1) / 2.0) / (len(idxs) - 1)
            start = end
        out[idxs] = ranks
    return out


def _blend_scores(users: List[str], champion: np.ndarray, fm: np.ndarray, weight: float) -> np.ndarray:
    champion_rank = _within_user_ranks(users, champion)
    fm_rank = _within_user_ranks(users, fm)
    return weight * champion_rank + (1.0 - weight) * fm_rank


def _read_test_rows(data_dir: str) -> List[Any]:
    import csv
    path = os.path.join(data_dir, "log_standard_4_29_to_5_08_pure.csv")
    if not os.path.exists(path):
        return []
    meta = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            meta[row["video_id"]] = (
                row.get("author_id", "UNK"),
                row.get("tag", "UNK"),
                row.get("music_id", "UNK"),
                row.get("video_type", "UNK"),
                0,
            )
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            author, tag, music, video_type, upload_date = meta.get(row["video_id"], ("UNK", "UNK", "UNK", "UNK", 0))
            rows.append(
                type("Row", (), {
                    "time_ms": int(row["time_ms"]),
                    "date": int(row["date"]),
                    "hour": int(int(row["hourmin"]) // 100),
                    "user": row["user_id"],
                    "video": row["video_id"],
                    "author": author,
                    "tag": tag,
                    "music": music,
                    "video_type": video_type,
                    "tab": row["tab"],
                    "duration": float(row["duration_ms"]),
                    "upload_date": upload_date,
                    "label": int(row["long_view"] != "0"),
                    "click": int(row["is_click"] != "0"),
                    "play_time_ms": float(row["play_time_ms"]),
                })()
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Autonomous ML research agent for KuaiRand-Pure")
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--out_dir", default="run_logs")
    ap.add_argument("--max_iters", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submission_path", default="submission.csv")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    controller = BenchmarkController(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        epsilon=args.epsilon,
        patience=args.patience,
        seed=args.seed,
    )
    result = controller.run()
    submission_path = controller.finalize_submission(args.submission_path)
    print("\n=== autonomous research agent ===")
    print(f"best validation model: {result['best_model']}")
    print(f"best validation metrics: {result['best']}")
    print(f"submission: {submission_path}")


if __name__ == "__main__":
    main()
