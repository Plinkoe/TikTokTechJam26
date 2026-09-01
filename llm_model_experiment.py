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
import sys
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

FEATURE_INTERFACE_CONTRACT = """
Define exactly one function named `transform_features`.

Required signature:
    def transform_features(cats, dense):
        # cats:  (n, 10) int64 categorical ids (same 10 fields as the model contract)
        # dense: (n, 10) float32 standardized dense features
        # returns: (n, d) float32 array, d <= 64, that REPLACES `dense`
        ...

The function MUST be row-wise: row i of the output may depend only on row i of
the inputs. Do NOT use column statistics (mean/std/min/max/percentile over the
batch), sorting, or anything else that mixes rows -- the transform is applied to
each split separately, so cross-row statistics leak the split's own
distribution and are rejected by a permutation check before training.

Keep the original columns (e.g. `np.concatenate([dense, new_cols], axis=1)`)
unless you have a reason to drop them. Only `np` (numpy) is in scope -- do not
write import statements. Output must be finite (no NaN/inf).
"""

# A fixed, decent architecture used to score feature-stage and training-stage
# proposals in isolation: if the model changes at the same time as the
# features, the run cannot attribute the delta to either.
DEFAULT_CANDIDATE_CODE = """
class CandidateModel(nn.Module):
    def __init__(self, vocab_sizes, dense_dim, emb_dim=12, hidden=96, dropout=0.1, **kwargs):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in vocab_sizes])
        self.mlp = nn.Sequential(
            nn.Linear(len(vocab_sizes) * emb_dim + dense_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cats, dense, history=None):
        e = torch.cat([emb(cats[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.mlp(torch.cat([e, dense], dim=1)).view(-1)
"""

_ALLOWED_BUILTINS = {
    "range": range, "len": len, "list": list, "dict": dict, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "min": min, "max": max, "sum": sum,
    "int": int, "float": float, "bool": bool, "str": str, "super": super,
    "isinstance": isinstance, "abs": abs, "print": print, "object": object,
    "__build_class__": __build_class__,  # required for `class ...:` statements
    "__name__": "llm_candidate_model",
}

# Roots that must never be reachable, even indirectly through a library.
_IMPORT_DENYLIST = {
    "os", "sys", "subprocess", "shutil", "socket", "importlib", "builtins",
    "pathlib", "ctypes", "pickle", "marshal", "requests", "urllib", "http",
    "runpy", "code", "codeop", "inspect", "gc", "threading", "multiprocessing",
}


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Allow libraries to resolve their own lazily-imported submodules.

    numpy's ndarray methods (.mean/.std/.sum) import numpy._core._methods at
    CALL time, so a namespace with no __import__ turns an ordinary row-wise
    feature into `KeyError: '__import__'`. Rather than drop the sandbox, allow
    only modules already loaded before the candidate ran -- nothing new is read
    from disk -- and refuse the dangerous roots outright.

    Candidate source still cannot contain "import ", "__import__", "os." or
    "sys." (see _FORBIDDEN_SUBSTRINGS), so this only serves library internals.
    """
    root = name.split(".")[0]
    if root in _IMPORT_DENYLIST:
        raise CandidateCodeError(f"import of {name!r} is not permitted in candidate code")
    if name not in sys.modules and root not in sys.modules:
        raise CandidateCodeError(f"import of {name!r} is not permitted in candidate code")
    return __import__(name, globals, locals, fromlist, level)


_ALLOWED_BUILTINS["__import__"] = _guarded_import

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


def compile_feature_transform(code: str):
    """Exec an agent-authored feature transform and prove it is row-wise.

    The permutation check is the important part: applying the transform to a
    shuffled batch must equal shuffling the transform of the original batch.
    Any use of column statistics (batch mean/std, sorting, ranking) breaks that
    identity -- and would silently leak each split's own distribution into its
    features, inflating validation and collapsing on the hidden test set.
    """
    lowered = code.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token in lowered:
            raise CandidateCodeError(f"disallowed token in feature code: {token!r}")

    namespace: Dict[str, Any] = {"np": np, "__builtins__": _ALLOWED_BUILTINS}
    try:
        exec(compile(code, "<llm_feature_transform>", "exec"), namespace)
    except Exception as exc:
        raise CandidateCodeError(f"feature code failed to compile/exec: {exc}") from exc

    fn = namespace.get("transform_features")
    if not callable(fn):
        raise CandidateCodeError("feature code must define `def transform_features(cats, dense)`")

    rng = np.random.default_rng(0)
    cats = rng.integers(0, 4, size=(64, 10)).astype(np.int64)
    dense = rng.normal(size=(64, 10)).astype(np.float32)
    try:
        out = np.asarray(fn(cats, dense))
    except Exception as exc:
        raise CandidateCodeError(f"transform_features raised on a dry run: {exc}") from exc

    if out.ndim != 2 or out.shape[0] != cats.shape[0]:
        raise CandidateCodeError(
            f"transform_features must return (n, d); got {out.shape} for n={cats.shape[0]}")
    if out.shape[1] < 1 or out.shape[1] > 64:
        raise CandidateCodeError(f"transform_features must return 1..64 columns; got {out.shape[1]}")
    if not np.all(np.isfinite(out)):
        raise CandidateCodeError("transform_features produced NaN or inf")

    # Two independent invariants, because either one alone has a blind spot.
    #
    # 1. Subset consistency: transforming the first half must equal the first
    #    half of transforming everything. Batch statistics (mean/std/percentile)
    #    change with the subset, so this catches them.
    # 2. Permutation equivariance: catches order-dependent transforms such as
    #    np.sort(axis=0) or cumulative sums, which ARE subset-stable at the
    #    head of the array but reorder rows.
    #
    # Mean-centering passes (2) on its own -- the batch mean does not change
    # under a permutation -- which is exactly why (1) has to be there.
    half = cats.shape[0] // 2
    subset = np.asarray(fn(cats[:half], dense[:half]))
    if subset.shape != (half, out.shape[1]) or not np.allclose(
            subset, out[:half], atol=1e-5, rtol=1e-4):
        raise CandidateCodeError(
            "transform_features is not row-wise: transforming a subset of the rows "
            "differs from transforming all of them. Remove any column statistics "
            "(mean/std/min/max/percentile over the batch) -- they leak each split's "
            "own distribution into its features.")

    perm = rng.permutation(cats.shape[0])
    shuffled = np.asarray(fn(cats[perm], dense[perm]))
    if shuffled.shape != out.shape or not np.allclose(shuffled, out[perm], atol=1e-5, rtol=1e-4):
        raise CandidateCodeError(
            "transform_features is not row-wise: transforming a shuffled batch differs "
            "from shuffling the transformed batch. Remove anything order-dependent "
            "(sorting, ranking, cumulative sums over the batch).")
    return fn


def _apply_feature_transform(fn, cats, dense):
    """Apply a validated transform to one split, keeping dtype and row count."""
    if fn is None:
        return dense
    out = np.asarray(fn(cats, dense), dtype=np.float32)
    if out.ndim != 2 or out.shape[0] != dense.shape[0]:
        raise CandidateCodeError(
            f"transform_features returned {out.shape} for {dense.shape[0]} rows")
    if not np.all(np.isfinite(out)):
        raise CandidateCodeError("transform_features produced NaN or inf on real data")
    return out


def _make_loss(p: Dict[str, Any]):
    """Training-stage recipes the agent can select without writing code.

    The label is heavily imbalanced, so pos_weight and focal loss are the two
    levers most likely to move a ranking metric.
    """
    kind = str(p.get("loss", "bce")).lower()
    if kind in ("bce", "", "none"):
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y)
    if kind == "weighted_bce":
        weight = torch.tensor(float(p.get("pos_weight", 1.0)))
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
    if kind == "focal":
        gamma = float(p.get("focal_gamma", 2.0))
        alpha = float(p.get("focal_alpha", 0.5))
        def _focal(logits, y):
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            prob = torch.sigmoid(logits)
            p_t = prob * y + (1 - prob) * (1 - y)
            a_t = alpha * y + (1 - alpha) * (1 - y)
            return (a_t * (1 - p_t).pow(gamma) * bce).mean()
        return _focal
    raise CandidateCodeError(f"unsupported loss {kind!r}; use bce, weighted_bce or focal")


def _make_scheduler(optimizer, p: Dict[str, Any], epochs: int):
    kind = str(p.get("scheduler", "none")).lower()
    if kind in ("none", "", "constant"):
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(p.get("step_size", 3))), gamma=float(p.get("gamma", 0.5)))
    raise CandidateCodeError(f"unsupported scheduler {kind!r}; use none, cosine or step")


def _predict(model, cats, dense, history, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(cats), batch_size):
            h = history[i:i + batch_size] if history is not None else None
            out.append(model(cats[i:i + batch_size], dense[i:i + batch_size], h).cpu().numpy())
    return np.concatenate(out)


def train_and_eval_candidate(model_cls: Type[nn.Module], data_dir: str,
                              hyperparams: Optional[Dict[str, Any]] = None,
                              feature_code: Optional[str] = None) -> Dict[str, Any]:
    """Same train/early-stop/eval loop as history_model.run_history_deepfm,
    parameterized by an arbitrary CandidateModel class instead of the one
    fixed HistoryDeepFM architecture. Validation-only: never reads test rows.
    """
    p = {
        "epochs": 8, "lr": 1e-3, "emb_dim": 12, "hidden": 96, "batch_size": 8192,
        "patience": 3, "seed": 0, "dropout": 0.1, "weight_decay": 1e-6,
        "history_len": 0, "loss": "bce", "scheduler": "none", "grad_clip": 0.0,
    }
    p.update(hyperparams or {})
    np.random.seed(p["seed"]); torch.manual_seed(p["seed"])

    rows = load_history_rows(data_dir)
    (tr_cat, tr_dense, tr_hist, tr_y, *_), (va_cat, va_dense, va_hist, va_y, *_, va_users), \
        vocab_sizes, _ = build_causal_features(rows, history_len=p["history_len"])

    # Feature stage. Applied per split with the SAME validated row-wise
    # function, so validation features never see training statistics.
    transform = compile_feature_transform(feature_code) if feature_code else None
    tr_dense = _apply_feature_transform(transform, tr_cat, tr_dense)
    va_dense = _apply_feature_transform(transform, va_cat, va_dense)

    tr_cat = torch.from_numpy(tr_cat); tr_dense = torch.from_numpy(tr_dense); tr_y = torch.from_numpy(tr_y)
    va_cat = torch.from_numpy(va_cat); va_dense = torch.from_numpy(va_dense)
    tr_hist = torch.from_numpy(tr_hist) if tr_hist is not None else None
    va_hist = torch.from_numpy(va_hist) if va_hist is not None else None

    model = model_cls(vocab_sizes, tr_dense.shape[1], emb_dim=p["emb_dim"],
                       hidden=p["hidden"], dropout=p["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = _make_loss(p)
    scheduler = _make_scheduler(optimizer, p, p["epochs"])
    grad_clip = float(p.get("grad_clip", 0.0) or 0.0)
    best, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(p["seed"])

    for _epoch in range(1, p["epochs"] + 1):
        model.train()
        order = rng.permutation(len(tr_y))
        for start in range(0, len(order), p["batch_size"]):
            idx = torch.from_numpy(order[start:start + p["batch_size"]])
            h = tr_hist[idx] if tr_hist is not None else None
            logits = model(tr_cat[idx], tr_dense[idx], h)
            loss = loss_fn(logits, tr_y[idx])
            optimizer.zero_grad(); loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        scores = _predict(model, va_cat, va_dense, va_hist, p["batch_size"])
        metrics = evaluate(va_users, va_y, scores)
        if metrics["primary"] > best + 1e-5:
            best, best_state, bad = metrics["primary"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= p["patience"]:
                break

    if best_state is None:
        raise CandidateCodeError("training produced no usable epoch (validation never improved)")
    model.load_state_dict(best_state)
    final_scores = _predict(model, va_cat, va_dense, va_hist, p["batch_size"])
    return {"valid": evaluate(va_users, va_y, final_scores),
            "dense_dim": int(tr_dense.shape[1])}
