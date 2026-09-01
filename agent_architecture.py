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
from llm_model_experiment import (CandidateCodeError, compile_candidate_model,
                                  compile_feature_transform, train_and_eval_candidate,
                                  _apply_feature_transform, DEFAULT_CANDIDATE_CODE)
from models import NumpyNNModel
from submit import write_submission, read_submission
from llm_planner import LLMPlanner, LLMPlannerConfig, ProposedExperiment, load_dotenv


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
            "dropout": 0.1,
            "weight_decay": 1e-6,
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
            dropout=p.get("dropout", 0.1),
            weight_decay=p.get("weight_decay", 1e-6),
        )
        return {"valid": result["valid"]}

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        rows = load_history_rows(context.data_dir)
        train = rows["train"] + rows["valid"]
        # NOTE: context.splits["test"] holds the tuple-encoded rows used by the
        # plain FM/tabular pipeline (data.py:encode). build_causal_features needs
        # the Row objects with .time_ms/.author/etc that _read_test_rows produces.
        # `context.splits.get("test") or ...` used to short-circuit here because
        # the tuple rows are truthy, silently feeding the wrong row type into
        # build_causal_features and crashing with AttributeError — which
        # finalize_submission() then swallowed and fell back to the FM baseline
        # without telling anyone. Always use the Row-object loader here.
        test_rows = _read_test_rows(context.data_dir)
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
            dropout=p.get("dropout", 0.1),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p.get("weight_decay", 1e-6))
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


class LLMCodeExperiment(BaseExperiment):
    """Trains whatever `CandidateModel` architecture the LLM planner wrote for
    this iteration -- not a pre-built class from this file. Feature
    engineering, the training loop, and the eval protocol are fixed and
    shared (see llm_model_experiment.py); only the model architecture itself
    is generated, sandboxed, and dry-run-validated before real training.

    One registry entry serves every iteration: the controller gives each
    proposal its own record name (llm_generated_<n>) so several distinct
    architectures can be tried across a run, but they all route through this
    same experiment object with a different params["code"] each time.
    """
    name = "llm_generated"
    family = "llm_code"

    def default_params(self) -> Dict[str, Any]:
        return {
            "epochs": 6, "lr": 1e-3, "emb_dim": 12, "hidden": 96,
            "dropout": 0.1, "weight_decay": 1e-6, "seed": 0,
            "history_len": 0, "batch_size": 8192, "patience": 3,
            "code": "", "feature_code": "",
            "loss": "bce", "scheduler": "none", "grad_clip": 0.0,
        }

    def run(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        code = (p.pop("code", "") or "").strip()
        feature_code = (p.pop("feature_code", "") or "").strip()
        if not code:
            raise CandidateCodeError("llm_generated experiment requires a non-empty params['code']")
        model_cls = compile_candidate_model(code)
        return train_and_eval_candidate(model_cls, context.data_dir, hyperparams=p,
                                        feature_code=feature_code or None)

    def finalize(self, context: "BenchmarkContext", params: Optional[Dict[str, Any]] = None):
        p = self.default_params(); p.update(params or {})
        code = (p.pop("code", "") or "").strip()
        feature_code = (p.pop("feature_code", "") or "").strip()
        if not code:
            raise CandidateCodeError("llm_generated experiment requires params['code'] to finalize")
        model_cls = compile_candidate_model(code)
        transform = compile_feature_transform(feature_code) if feature_code else None

        rows = load_history_rows(context.data_dir)
        train = rows["train"] + rows["valid"]
        test_rows = _read_test_rows(context.data_dir)
        if not test_rows:
            raise RuntimeError("No test rows available in data_dir; cannot finalize submission.")
        split = {"train": train, "valid": test_rows}
        (tr_cat, tr_dense, tr_hist, tr_y, *_), (va_cat, va_dense, va_hist, va_y, *_, va_users), \
            vocab_sizes, _ = build_causal_features(split, history_len=p.get("history_len", 0))

        tr_dense = _apply_feature_transform(transform, tr_cat, np.asarray(tr_dense, dtype=np.float32))
        va_dense = _apply_feature_transform(transform, va_cat, np.asarray(va_dense, dtype=np.float32))

        tr_cat = torch.from_numpy(np.asarray(tr_cat, dtype=np.int64))
        tr_dense = torch.from_numpy(np.asarray(tr_dense, dtype=np.float32))
        tr_y = torch.from_numpy(np.asarray(tr_y, dtype=np.float32))
        va_cat = torch.from_numpy(np.asarray(va_cat, dtype=np.int64))
        va_dense = torch.from_numpy(np.asarray(va_dense, dtype=np.float32))
        tr_hist = torch.from_numpy(np.asarray(tr_hist, dtype=np.int32)) if tr_hist is not None else None
        va_hist = torch.from_numpy(np.asarray(va_hist, dtype=np.int32)) if va_hist is not None else None

        model = model_cls(vocab_sizes, tr_dense.shape[1], emb_dim=p["emb_dim"],
                           hidden=p["hidden"], dropout=p["dropout"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p.get("weight_decay", 1e-6))
        rng = np.random.default_rng(p["seed"])
        for _ in range(p["epochs"]):
            model.train()
            order = rng.permutation(len(tr_y))
            for start in range(0, len(order), p["batch_size"]):
                idx = torch.from_numpy(order[start:start + p["batch_size"]])
                h = tr_hist[idx] if tr_hist is not None else None
                optimizer.zero_grad()
                logits = model(tr_cat[idx], tr_dense[idx], h)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tr_y[idx])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = model(va_cat, va_dense, va_hist).sigmoid().numpy()
        return scores


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


# Spec 2.3: 50 iterations per benchmark run is a hard cap, with a 6h
# wall-clock backstop. The convergence rule (eps=0.002, N=3) normally fires
# first; these exist so a pathological run cannot spin forever.
MAX_ITERATIONS_HARD_CAP = 50
WALL_CLOCK_LIMIT_SEC = 6 * 60 * 60


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
    def __init__(self, data_dir: str, out_dir: str = "run_logs", max_iters: int = 10, epsilon: float = 0.002, patience: int = 3, seed: int = 0, llm_config: Optional[LLMPlannerConfig] = None, wall_clock_limit: float = WALL_CLOCK_LIMIT_SEC):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.epsilon = epsilon
        self.patience = patience
        self.seed = seed
        self.splits = load(data_dir)
        self.logger = RunLogger(out_dir)
        self.registry = ExperimentRegistry()
        for model in [FMExperiment(), HistoryDeepFMExperiment(), MultitaskDeepFMExperiment(), TabularExperiment(), BlendExperiment(), LLMCodeExperiment()]:
            self.registry.register(model)
        self.context = BenchmarkContext(data_dir=data_dir, out_dir=out_dir, splits=self.splits, seed=seed, max_iters=max_iters, epsilon=epsilon, patience=patience)
        if max_iters > MAX_ITERATIONS_HARD_CAP:
            print(f"[budget] --max_iters {max_iters} exceeds the {MAX_ITERATIONS_HARD_CAP}-iteration "
                  f"cap; clamping.")
        self.max_iters = min(max_iters, MAX_ITERATIONS_HARD_CAP)
        self.wall_clock_limit = float(wall_clock_limit)
        self.run_started_at: Optional[float] = None
        self.stop_reason: str = "not_started"
        # Signatures of experiments whose failure is deterministic, so the run
        # does not spend its budget re-deriving the same traceback.
        self.blocked: Dict[str, str] = {}
        self.schedule = self.registry.default_schedule()
        self.tried = set()
        self.best_primary = -1.0
        self.llm_planner = LLMPlanner(llm_config or LLMPlannerConfig.from_env())
        # Reject unusable generated code at proposal time (sandbox token check +
        # dry-run forward pass), so the planner can self-correct within one turn.
        self.llm_planner.code_validator = compile_candidate_model
        self.llm_planner.feature_validator = compile_feature_transform

    def _resolve_experiment(self, name: str) -> BaseExperiment:
        """Look up an experiment by record name.

        Pre-built experiments (fm_baseline, history_deepfm, ...) are keyed by
        their own name. LLM-authored architectures get a fresh record name
        per iteration (llm_generated_0, llm_generated_1, ...) so several
        distinct candidates can appear in the same run's logs, but they all
        route through the single registered `llm_generated` experiment object.
        """
        if name in self.registry._items:
            return self.registry.get(name)
        if name.startswith(("llm_generated", "llm_features", "llm_train")):
            return self.registry.get("llm_generated")
        raise KeyError(name)

    def _record_planner_failure(self, reason: str) -> None:
        """Make a planner failure visible instead of silently rescheduling.

        Before this, a failed LLM call left NO trace in iterations.jsonl: the
        controller just fell through to schedule[0] and the run printed
        "best validation model: fm_baseline" as if nothing had gone wrong.
        """
        planner = self.llm_planner
        record = {
            "timestamp": time.time(),
            "iteration": self.context.iteration_count,
            "experiment": "llm_planner",
            "family": "planner",
            "hypothesis": "LLM planner proposal",
            "code_diff": "",
            "params": {
                "provider": planner.config.provider,
                "model": planner.config.model,
                "force_code": planner.config.force_code,
                "max_tokens": planner.config.max_tokens,
            },
            "metrics": None,
            "error": reason,
            "status": "planner_failed",
            "finish_reason": getattr(planner, "last_finish_reason", None),
            "raw_response": (getattr(planner, "last_raw_content", "") or "")[:2000],
        }
        self.logger.write(record)
        if planner.config.force_code:
            raise RuntimeError(
                "LLM planner failed while KUAI_LLM_FORCE_CODE=true: " + str(reason) +
                " -- refusing to fall back to the fixed schedule. "
                "See run_logs/llm_calls.jsonl for the raw provider response."
            )

    def _maybe_llm_next_spec(self, previous_records: Iterable[Dict[str, Any]]) -> Optional[ExperimentSpec]:
        if not self.llm_planner.enabled:
            return None

        proposal = self.llm_planner.suggest_next_experiment(self.registry, previous_records)
        if proposal is None:
            self._record_planner_failure(self.llm_planner.last_error or "planner returned no proposal")
            return None

        # Code-mode: the LLM wrote a new CandidateModel architecture rather
        # than picking params for an existing pre-built experiment. Each such
        # proposal gets its own record name so it always runs (no dedup
        # against a fixed registry name) and shows up as its own line in
        # iterations.jsonl/results, distinct from earlier candidates.
        mode = getattr(proposal, "mode", "tune")
        code = (getattr(proposal, "code", "") or "").strip()
        feature_code = (getattr(proposal, "feature_code", "") or "").strip()
        n = len(list(previous_records))

        # Feature- and training-stage proposals run against a fixed reference
        # architecture: if the model changed at the same time, the resulting
        # delta could not be attributed to the stage under test.
        if mode == "features":
            if not feature_code:
                self._record_planner_failure("mode='features' proposal carried no feature_code")
                return None
            return ExperimentSpec(
                name=f"llm_features_{n}",
                family="llm_features",
                params={**dict(proposal.params), "code": code or DEFAULT_CANDIDATE_CODE,
                        "feature_code": feature_code},
                hypothesis=proposal.hypothesis,
                code_diff=feature_code,
            )

        if mode == "train":
            if not proposal.params:
                self._record_planner_failure("mode='train' proposal carried no params")
                return None
            return ExperimentSpec(
                name=f"llm_train_{n}",
                family="llm_train",
                params={**dict(proposal.params), "code": code or DEFAULT_CANDIDATE_CODE},
                hypothesis=proposal.hypothesis,
                code_diff=json.dumps(dict(proposal.params), sort_keys=True)[:2000],
            )

        if mode == "code" or proposal.family == "llm_code" or code:
            if not code:
                self._record_planner_failure("proposal claimed family='llm_code' but carried no code")
                return None
            return ExperimentSpec(
                name=f"llm_generated_{n}",
                family="llm_code",
                params={**dict(proposal.params), "code": code},
                hypothesis=proposal.hypothesis,
                code_diff=code,
            )

        if proposal.name not in self.registry._items:
            self._record_planner_failure(f"proposal named unknown experiment {proposal.name!r}")
            return None

        known = self.registry.get(proposal.name)
        spec = ExperimentSpec(
            name=proposal.name,
            family=proposal.family or known.family,
            params=dict(proposal.params),
            hypothesis=proposal.hypothesis,
            code_diff=proposal.code_diff,
        )
        if spec.name in {rec.get("experiment") for rec in previous_records if rec.get("experiment")}:
            return None
        return spec

    def apply_epochs_override(self, epochs: Optional[int]) -> None:
        """Cap every registered experiment's epoch budget for a quick smoke run.

        Only touches specs that already declare an 'epochs' param, so it never
        adds a meaningless field to experiments that don't train iteratively.
        """
        if epochs is None:
            return
        for spec in self.schedule:
            if "epochs" in spec.params:
                spec.params["epochs"] = epochs

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

        llm_choice = self._maybe_llm_next_spec(previous)
        if llm_choice is not None:
            return llm_choice

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
                if spec.family == family and spec.name not in seen and spec.name not in self.blocked:
                    return spec
        for spec in self.schedule:
            if spec.name not in seen and spec.name not in self.blocked:
                return spec
        return None

    # Errors that are a property of the proposal itself. Re-running the exact
    # same spec cannot change the outcome, so one attempt is the honest budget;
    # the error text goes to the planner, which routes around it next turn.
    _DETERMINISTIC_ERRORS = (CandidateCodeError, KeyError, TypeError, AttributeError,
                             NameError, IndexError, ValueError)

    def _run_single_experiment(self, spec: ExperimentSpec, retries: int = 2) -> Dict[str, Any]:
        experiment = self._resolve_experiment(spec.name)
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
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = traceback.format_exc()
                record["error_type"] = type(exc).__name__
                deterministic = isinstance(exc, self._DETERMINISTIC_ERRORS)
                record["retryable"] = not deterministic
                self.logger.write(record)
                if deterministic:
                    # Remember it so the fixed schedule does not re-offer the
                    # same broken spec, and so the planner sees it in history.
                    self.blocked[spec.name] = f"{type(exc).__name__}: {exc}"
                    return record
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))  # transient: back off, then retry
                    continue
                return record
        return {"experiment": spec.name, "status": "failed", "error": "unreachable"}

    def _next_scheduled_spec(self, previous_records: Iterable[Dict[str, Any]]) -> Optional[ExperimentSpec]:
        """The fixed schedule, ignoring the planner. Used as the recovery path."""
        seen = {rec.get("experiment") for rec in previous_records if rec.get("experiment")}
        for spec in self.schedule:
            if spec.name not in seen and spec.name not in self.blocked:
                return spec
        return None

    def run(self) -> Dict[str, Any]:
        self.run_started_at = time.time()
        baseline_metrics = self.baseline_gate()
        self.context.best_metrics = baseline_metrics
        self.context.best_model_name = "fm_baseline"
        self.context.best_record = {"experiment": "fm_baseline", "metrics": {"valid": baseline_metrics}}
        self.best_primary = float(baseline_metrics["primary"])
        self._save_checkpoint(self.context.best_record)

        previous_records: List[Dict[str, Any]] = []
        stagnation = 0
        self.stop_reason = "max_iterations"
        while self.context.iteration_count < self.max_iters:
            elapsed = time.time() - self.run_started_at
            if elapsed >= self.wall_clock_limit:
                self.stop_reason = "wall_clock_limit"
                print(f"[budget] wall-clock limit reached after {elapsed / 3600:.2f}h; stopping.")
                break
            try:
                next_spec = self.select_next_experiment(previous_records)
            except Exception as exc:
                # A planner that cannot produce a usable proposal must not end
                # the run: fall back to the fixed schedule and keep going.
                if self.llm_planner.config.force_code:
                    raise
                print(f"[planner] proposal failed ({type(exc).__name__}: {exc}); "
                      f"falling back to the fixed schedule.")
                next_spec = self._next_scheduled_spec(previous_records)
            if next_spec is None:
                self.stop_reason = "schedule_exhausted"
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
            # Convergence rule: N consecutive iterations without an
            # epsilon-sized improvement over the incumbent.
            if stagnation >= self.patience:
                self.stop_reason = "converged"
                print(f"[convergence] {self.patience} iterations without a "
                      f"+{self.epsilon} improvement; stopping.")
                break
        return {
            "best": self.context.best_metrics,
            "best_model": self.context.best_model_name,
            "iterations": previous_records,
            "stop_reason": self.stop_reason,
            "wall_clock_sec": time.time() - self.run_started_at,
            "blocked": dict(self.blocked),
        }

    def finalize_submission(self, submission_path: str = "submission.csv") -> str:
        target = self.context.best_model_name or "fm_baseline"
        # NOTE: this used to call experiment.finalize(context, experiment.default_params()),
        # silently ignoring whatever params actually won during the loop (tuned
        # dropout, an llm_generated candidate's code, ...) and retraining the
        # experiment's *default* config instead. That bug is why the published
        # test numbers only matched the real champion when finalize_and_score.py
        # was run by hand with the winning config re-supplied manually -- the
        # automated agent.py path never reproduced it. Use the winning record's
        # actual params here, layered over the experiment's defaults.
        best_params = dict((self.context.best_record or {}).get("params") or {})
        for candidate, candidate_params in [(target, best_params), ("fm_baseline", {})]:
            try:
                experiment = self._resolve_experiment(candidate)
                merged_params = dict(experiment.default_params())
                merged_params.update(candidate_params)
                scores = experiment.finalize(self.context, merged_params)
                full_path = os.path.join(self.out_dir, submission_path)
                write_submission(full_path, self.splits["test"], scores)
                read_submission(full_path, self.splits["test"])
                if candidate != target:
                    print(
                        f"[finalize_submission] WARNING: best model '{target}' failed to "
                        f"finalize; submission was generated with fallback '{candidate}' instead. "
                        f"This means the shipped submission does NOT reflect the best validation "
                        f"model. See traceback above."
                    )
                status_path = os.path.join(self.out_dir, "finalize_status.json")
                with open(status_path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "finalized_with": candidate,
                        "requested_best_model": target,
                        "fallback_used": candidate != target,
                    }, fh, indent=2)
                return full_path
            except Exception:
                print(f"[finalize_submission] '{candidate}' failed to finalize:")
                traceback.print_exc()
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


def _read_rows_from_log(data_dir: str, filename: str, lo: int, hi: int) -> List[Any]:
    """Shared Row-object loader: read `filename`, keep rows with lo<=date<=hi,
    and join video-side metadata. Used for both the official test slice and the
    unbiased random-exposure validation slice below — same row shape either way,
    so build_causal_features doesn't care which one it's scoring.
    """
    import csv
    from history_model import _date_int

    path = os.path.join(data_dir, filename)
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
                _date_int(row.get("upload_dt", "")),
            )
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            date = int(row["date"])
            if not (lo <= date <= hi):
                continue
            author, tag, music, video_type, upload_date = meta.get(row["video_id"], ("UNK", "UNK", "UNK", "UNK", 0))
            rows.append(
                type("Row", (), {
                    "time_ms": int(row["time_ms"]),
                    "date": date,
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


def _read_test_rows(data_dir: str) -> List[Any]:
    """Load the held-out test rows (2022-04-29..05-08) as Row objects.

    NOTE: KuaiRand-Pure does NOT ship a separate `log_standard_4_29_to_5_08_pure.csv`
    file. Valid (04-22..04-28) and test (04-29..05-08) both live inside
    `log_standard_4_22_to_5_08_pure.csv` and must be split by date, exactly like
    `data.py:load()` does. A previous version of this function pointed at a
    nonexistent filename and silently returned [] (`if not os.path.exists: return []`),
    which caused finalize() for history_deepfm to always raise, which
    finalize_submission() then silently swallowed by falling back to the plain
    FM baseline — so the shipped submission never actually reflected the champion
    model. See SPLITS in data.py for the authoritative date ranges.
    """
    from data import SPLITS
    lo, hi = SPLITS["test"]
    return _read_rows_from_log(data_dir, "log_standard_4_22_to_5_08_pure.csv", lo, hi)


def read_unbiased_valid_rows(data_dir: str) -> List[Any]:
    """A second, unbiased validation slice from KuaiRand-Pure's randomly-exposed
    log (`log_random_4_22_to_5_08_pure.csv`), restricted to the SAME date window
    as the official (biased) validation split (2022-04-22..04-28).

    Deliberately does not touch any date in the test window (04-29..05-08) —
    this stays a validation-time tool, not a second peek at test. Its point is
    that log_standard's validation rows come from KuaiRand's normal
    recommender-exposed traffic (same selection bias as train), while
    log_random's rows are randomly exposed — so a model that's overfitting to
    exposure-correlated patterns in the biased validation split should show a
    smaller (or negative) edge here, even though neither slice touches test.
    """
    from data import SPLITS
    lo, hi = SPLITS["valid"]
    return _read_rows_from_log(data_dir, "log_random_4_22_to_5_08_pure.csv", lo, hi)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Autonomous ML research agent for KuaiRand-Pure")
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--out_dir", default="run_logs")
    ap.add_argument("--max_iters", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submission_path", default="submission.csv")
    ap.add_argument(
        "--wall_clock_hours", type=float, default=WALL_CLOCK_LIMIT_SEC / 3600.0,
        help="Backstop wall-clock ceiling for the iteration loop (spec 2.3: 6h).",
    )
    ap.add_argument(
        "--epochs", type=int, default=None,
        help="Optional cap applied to every registered experiment's epoch budget "
             "(e.g. for a quick smoke test). Leave unset to use each experiment's "
             "own tuned default (FM=40 w/ patience, history_deepfm=8, etc).",
    )
    ap.add_argument(
        "--llm_enabled",
        action="store_true",
        default=False,
        help="Enable the optional JSON-based LLM planner. Requires KUAI_LLM_API_KEY or OPENAI_API_KEY.",
    )
    ap.add_argument(
        "--llm_provider",
        default=None,
        help="LLM provider: openai, openrouter, azure. Defaults to env or openai.",
    )
    ap.add_argument(
        "--llm_model",
        default=None,
        help="Model name for the LLM planner. Defaults to env or gpt-4o-mini.",
    )
    return ap


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    args = build_parser().parse_args()
    llm_config = LLMPlannerConfig.disabled()
    if args.llm_enabled or os.getenv("KUAI_LLM_ENABLED", "").lower() in {"1", "true", "yes", "on"}:
        llm_config = LLMPlannerConfig(
            enabled=True,
            provider=(args.llm_provider or os.getenv("KUAI_LLM_PROVIDER") or "openai").strip() or "openai",
            model=(args.llm_model or os.getenv("KUAI_LLM_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini",
            api_key=os.getenv("KUAI_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("KUAI_LLM_BASE_URL") or None,
            temperature=float(os.getenv("KUAI_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("KUAI_LLM_MAX_TOKENS", "300")),
            system_prompt=os.getenv("KUAI_LLM_SYSTEM_PROMPT", "You are a disciplined ML research assistant for the KuaiRand benchmark. Return only valid JSON.")
        )
    controller = BenchmarkController(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        epsilon=args.epsilon,
        patience=args.patience,
        seed=args.seed,
        llm_config=llm_config,
    )
    controller.apply_epochs_override(args.epochs)
    result = controller.run()
    submission_path = controller.finalize_submission(args.submission_path)
    print("\n=== autonomous research agent ===")
    print(f"best validation model: {result['best_model']}")
    print(f"best validation metrics: {result['best']}")
    print(f"submission: {submission_path}")


if __name__ == "__main__":
    main()
