"""Generic training/eval scaffold for LLM-authored model architectures.

Feature engineering, data loading, and the evaluation protocol stay fixed
(the causal-history features from history_model.py, evaluate.py's metric
definitions) -- those have correctness constraints (no test leakage, no
label leakage, fixed metrics) that shouldn't be inside the blast radius of
generated code. Only the model architecture (`CandidateModel`) is authored
per-iteration by the LLM. That's the trade-off: this makes the agent
genuinely write and revise model code instead of picking between five
pre-built classes, but it does NOT make it free to rewrite feature
engineering, the training loop, or the eval protocol -- those remain
human-authored and fixed, same as evaluate.py always was.
"""
from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate import evaluate
from history_model import build_causal_features, load_history_rows

MODEL_INTERFACE_CONTRACT = """
Define exactly one class named `CandidateModel`, subclassing `torch.nn.Module`.

Required constructor signature:
    def __init__(self, vocab_sizes, dense_dim, emb_dim=12, hidden=96, dropout=0.1, **kwargs):
        ...
    # vocab_sizes: list[int], one entry per categorical field (10 fields:
    # user, video, author, tag, music, video_type, tab, duration_bucket, hour, video_age_bucket)
    # dense_dim: int, width of the standardized dense feature vector

Required forward signature:
    def forward(self, cats, dense, history=None):
        # cats: (batch, num_fields) int64 categorical ids
        # dense: (batch, dense_dim) float32 standardized dense features
        # history: (batch, history_len) int64 padded video-id history, or None
        #          -- only meaningful if you asked for history_len > 0 in hyperparams
        # returns: (batch,) float tensor of logits (pre-sigmoid) for long_view
        ...

Only `torch`, `nn` (torch.nn), `F` (torch.nn.functional), and `np` (numpy) are
available in scope -- do not write import statements, do not touch files,
sockets, or the environment. Write only the CandidateModel class; there is no
separate hook for the training loop, optimizer, or feature engineering --
those are fixed and shared across all candidates so results stay comparable.
"""

_ALLOWED_BUILTINS = {
    "range": range, "len": len, "list": list, "dict": dict, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "min": min, "max": max, "sum": sum,
    "int": int, "float": float, "bool": bool, "str": str, "super": super,
    "isinstance": isinstance, "abs": abs, "print": print, "object": object,
    "__build_class__": __build_class__,  # required for `class ...:` statements
    "__name__": "llm_candidate_model",
}

_FORBIDDEN_SUBSTRINGS = (
    "import ", "__import__", "eval(", "exec(", "open(", "os.", "sys.",
    "socket", "subprocess", "shutil", "requests", "urllib", "pathlib",
    "__class__", "__bases__", "__subclasses__", "__globals__",
)


class CandidateCodeError(ValueError):
    """LLM-authored code failed to compile, exec, or pass contract validation.

    Deliberately a plain exception (not a crash) -- the controller catches
    this, logs the message into iterations.jsonl, and the *next* LLM prompt
    includes it, so a broken candidate is something the agent can see and
    fix on its next turn instead of a run that silently dies.
    """


def compile_candidate_model(code: str) -> Type[nn.Module]:
    """Exec candidate source in a restricted namespace and return the class.

    Also runs a one-batch dry-run forward pass on dummy tensors so shape and
    contract bugs (wrong return shape, crashing on history=None, stray
    imports) are caught here in under a second, rather than after several
    minutes of real training.
    """
    lowered = code.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token in lowered:
            raise CandidateCodeError(f"disallowed token in candidate code: {token!r}")

    namespace: Dict[str, Any] = {
        "torch": torch, "nn": nn, "F": F, "np": np,
        "__builtins__": _ALLOWED_BUILTINS,
    }
    try:
        exec(compile(code, "<llm_candidate_model>", "exec"), namespace)
    except Exception as exc:
        raise CandidateCodeError(f"candidate code failed to compile/exec: {exc}") from exc

    cls = namespace.get("CandidateModel")
    if cls is None or not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        raise CandidateCodeError("candidate code must define `class CandidateModel(nn.Module)`")

    try:
        dummy_vocab = [5] * 10
        model = cls(dummy_vocab, dense_dim=10, emb_dim=4, hidden=8, dropout=0.0)
        cats = torch.randint(0, 4, (3, len(dummy_vocab)))
        dense = torch.zeros(3, 10)
        out = model(cats, dense, history=None)
        if not torch.is_tensor(out) or tuple(out.shape) != (3,):
            got = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
            raise CandidateCodeError(f"forward() must return a (batch,) tensor; got {got}")
    except CandidateCodeError:
        raise
    except Exception as exc:
        raise CandidateCodeError(f"candidate model failed its dry-run forward pass: {exc}") from exc

    return cls


def _predict(model, cats, dense, history, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(cats), batch_size):
            h = history[i:i + batch_size] if history is not None else None
            out.append(model(cats[i:i + batch_size], dense[i:i + batch_size], h).cpu().numpy())
    return np.concatenate(out)


def train_and_eval_candidate(model_cls: Type[nn.Module], data_dir: str,
                              hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Same train/early-stop/eval loop as history_model.run_history_deepfm,
    parameterized by an arbitrary CandidateModel class instead of the one
    fixed HistoryDeepFM architecture. Validation-only: never reads test rows.
    """
    p = {
        "epochs": 8, "lr": 1e-3, "emb_dim": 12, "hidden": 96, "batch_size": 8192,
        "patience": 3, "seed": 0, "dropout": 0.1, "weight_decay": 1e-6,
        "history_len": 0,
    }
    p.update(hyperparams or {})
    np.random.seed(p["seed"]); torch.manual_seed(p["seed"])

    rows = load_history_rows(data_dir)
    (tr_cat, tr_dense, tr_hist, tr_y, *_), (va_cat, va_dense, va_hist, va_y, *_, va_users), \
        vocab_sizes, _ = build_causal_features(rows, history_len=p["history_len"])

    tr_cat = torch.from_numpy(tr_cat); tr_dense = torch.from_numpy(tr_dense); tr_y = torch.from_numpy(tr_y)
    va_cat = torch.from_numpy(va_cat); va_dense = torch.from_numpy(va_dense)
    tr_hist = torch.from_numpy(tr_hist) if tr_hist is not None else None
    va_hist = torch.from_numpy(va_hist) if va_hist is not None else None

    model = model_cls(vocab_sizes, tr_dense.shape[1], emb_dim=p["emb_dim"],
                       hidden=p["hidden"], dropout=p["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    best, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(p["seed"])

    for _epoch in range(1, p["epochs"] + 1):
        model.train()
        order = rng.permutation(len(tr_y))
        for start in range(0, len(order), p["batch_size"]):
            idx = torch.from_numpy(order[start:start + p["batch_size"]])
            h = tr_hist[idx] if tr_hist is not None else None
            logits = model(tr_cat[idx], tr_dense[idx], h)
            loss = F.binary_cross_entropy_with_logits(logits, tr_y[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        scores = _predict(model, va_cat, va_dense, va_hist, p["batch_size"])
        metrics = evaluate(va_users, va_y, scores)
        if metrics["primary"] > best + 1e-5:
            best, best_state, bad = metrics["primary"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= p["patience"]:
                break

    model.load_state_dict(best_state)
    final_scores = _predict(model, va_cat, va_dense, va_hist, p["batch_size"])
    return {"valid": evaluate(va_users, va_y, final_scores)}
