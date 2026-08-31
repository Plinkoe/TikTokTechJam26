"""LLM-driven research loop that can CREATE new experiment implementations.

Unlike the original registry planner, this module asks the LLM for executable
experiment source code. Each candidate is syntax/static checked and run in an
isolated subprocess using train + public validation only. The agent keeps the
best validation candidate and records the complete research trajectory.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional

from agent_architecture import BenchmarkController
from code_experiment import run_candidate
from llm_planner import LLMPlanner, LLMPlannerConfig, normalize_proposal


CODE_SYSTEM_PROMPT = r"""You are an autonomous ML research engineer working on KuaiRand-Pure.
You are NOT choosing from a fixed experiment registry. Your job is to invent a
new experiment and write its complete Python implementation.

Hard rules:
1. Development may use TRAIN and PUBLIC VALIDATION only. Never read or infer the
   hidden test split. Do not mention or open test files in the generated code.
2. Return JSON only, with top-level object {"proposal": {...}}.
3. proposal must contain: name, family, hypothesis, code_diff, source.
4. source must be a complete Python module defining:
       def run(train_csv: str, valid_csv: str, data_dir: str) -> dict:
   It must return {"valid": {"GAUC": float, "nDCG@5": float, "primary": float,
   "users": int, "rows": int}}.
5. The module runs from an isolated temporary directory, with the repository on
   PYTHONPATH. It may import trusted modules: data, baseline, evaluate, features,
   history_model, numpy, torch, sklearn if installed. It should preferably reuse
   trusted loading/evaluation functions rather than reimplement the metric.
6. The experiment must contain a substantive algorithmic change: feature
   engineering, sampling, model architecture, regularization, loss, calibration,
   ranking, or an ensemble. Hyperparameter-only proposals are not enough.
7. Keep runtime reasonable for KuaiRand-Pure. Prefer one focused hypothesis over
   a huge model.

Useful trusted APIs:
- data.load(data_dir) -> dict with train/valid/test tuples. Your code MUST ignore
  the test entry and only use train/valid.
- data.encode(splits) -> (encoded, field_dim), where encoded[name] is (X,y,users).
- baseline.FM(field_dim, k=..., lr=..., seed=...) has step(X,y) and predict(X).
- evaluate.evaluate(users, y, scores) -> metrics containing primary, GAUC, nDCG@5.
- features.build_tabular_features(...) and history_model APIs are available if useful.

The benchmark baseline primary is approximately 0.6015 on this local validation
protocol. Study the history supplied by the caller and target a genuine improvement.
If the last idea failed, diagnose why and choose a different direction.
"""


def _extract_source(response: Any) -> Dict[str, Any]:
    proposal = normalize_proposal(response)
    raw = response.get("proposal", response) if isinstance(response, dict) else {}
    source = str(raw.get("source") or "") if isinstance(raw, dict) else ""
    if not source.strip():
        raise ValueError("LLM proposal did not contain executable 'source'")
    proposal["source"] = source
    return proposal


def _prompt(planner: LLMPlanner, records: Iterable[Dict[str, Any]], best: float) -> Dict[str, Any]:
    history = list(records)[-12:]
    compact = []
    for r in history:
        m = (r.get("metrics") or {}).get("valid") or {}
        compact.append({
            "name": r.get("experiment"), "family": r.get("family"),
            "hypothesis": r.get("hypothesis"), "primary": m.get("primary"),
            "status": r.get("status"), "error": r.get("error"),
        })
    payload = {
        "model": planner.config.model,
        "temperature": planner.config.temperature,
        "max_tokens": max(planner.config.max_tokens, 2500),
        "messages": [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "best_validation_primary": best,
                "recent_experiments": compact,
                "task": "Invent and implement the next experiment. Learn from failures. Return executable source, not a description.",
            }, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    return payload


class AutonomousCodeAgent:
    def __init__(self, controller: BenchmarkController, max_iters: int = 10, retries: int = 2):
        self.controller = controller
        self.max_iters = max_iters
        self.retries = retries
        self.planner = LLMPlanner(controller.llm_planner.config)
        self.records: List[Dict[str, Any]] = []
        self.best_primary = float(controller.best_primary)
        self.best_source: Optional[str] = None
        self.best_proposal: Optional[Dict[str, Any]] = None

    def _ask(self) -> Dict[str, Any]:
        payload = _prompt(self.planner, self.records, self.best_primary)
        response = self.planner._call_provider(payload)
        return _extract_source(response)

    def run(self) -> Dict[str, Any]:
        baseline = self.controller.baseline_gate()
        self.best_primary = float(baseline["primary"])
        self.records = [{"experiment": "fm_baseline", "family": "fm", "metrics": {"valid": baseline}, "status": "ok", "hypothesis": "official baseline"}]

        for iteration in range(1, self.max_iters + 1):
            started = time.time()
            proposal: Optional[Dict[str, Any]] = None
            record: Dict[str, Any] = {"iteration": iteration, "timestamp": started, "status": "failed"}
            try:
                proposal = self._ask()
                record.update({
                    "experiment": proposal["name"],
                    "family": proposal.get("family", "generated"),
                    "hypothesis": proposal.get("hypothesis", ""),
                    "code_diff": proposal.get("code_diff", ""),
                })
                result = run_candidate(
                    proposal["source"],
                    repo_dir=os.path.dirname(os.path.abspath(__file__)),
                    data_dir=self.controller.data_dir,
                )
                record["metrics"] = result
                record["status"] = "ok"
                record["duration_sec"] = time.time() - started
                primary = float(result["valid"]["primary"])
                record["primary"] = primary
                if primary > self.best_primary + self.controller.epsilon:
                    self.best_primary = primary
                    self.best_source = proposal["source"]
                    self.best_proposal = proposal
                    record["accepted"] = True
                else:
                    record["accepted"] = False
            except Exception:
                record["error"] = traceback.format_exc()
                record["duration_sec"] = time.time() - started
            self.records.append(record)
            self.controller.logger.write(record)
            # Failures are fed back to the next LLM call. We intentionally do not
            # terminate on one bad generated program.

        out = {
            "best_validation_primary": self.best_primary,
            "baseline_primary": float(baseline["primary"]),
            "improved_over_baseline": self.best_primary > float(baseline["primary"]),
            "best_proposal": self.best_proposal,
            "iterations": self.records,
        }
        path = os.path.join(self.controller.out_dir, "generated_research_result.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        return out


def run_code_agent(args) -> Dict[str, Any]:
    controller = BenchmarkController(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        epsilon=args.epsilon,
        patience=args.patience,
        seed=args.seed,
    )
    agent = AutonomousCodeAgent(controller, max_iters=args.max_iters)
    return agent.run()
