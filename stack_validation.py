"""Learned stacking combiner over validation-only rank features.

blend_validation.py's fixed weighted rank-average tops out at primary~0.6050
across every model pair tried (FM, attention, BPR, LightGBM). This tries a
learned combiner (logistic regression over within-user rank features) instead
of a hand-swept fixed weight, to see if the ceiling is the combination method
rather than a lack of complementary signal.

Fitting the combiner's weights on validation and then scoring it on the same
validation rows would leak (the weights would be tuned to this exact label
set). To keep this an honest, single read of validation, the combiner is
cross-fitted: split validation rows into K folds, fit weights on all folds but
one, predict on the held-out fold, and repeat so every row's reported score
comes from a fit that never saw its label. Test is never read here.
"""
import argparse

import numpy as np

from data import load
from evaluate import evaluate
from blend_validation import within_user_ranks


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, epochs=300, lr=0.5, l2=1e-4):
    """Plain-gradient-descent logistic regression; a handful of rank features
    and >1e4 rows per fold makes this converge in well under 300 steps."""
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--scores', nargs='+', required=True,
                        help='name=path.npy pairs, e.g. champion=run_logs/history_deepfm_seed0_v2.npy fm=run_logs/fm_seed0.npy')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rows = load(args.data_dir)['valid']
    users = [row[1] for row in rows]
    labels = np.asarray([row[6] for row in rows], dtype=np.float32)

    names, arrays = [], []
    for kv in args.scores:
        name, path = kv.split('=', 1)
        arr = np.load(path)
        if arr.ndim != 1 or len(arr) != len(rows):
            raise ValueError(f'{name}: score array must be 1-D and match validation row count')
        if not np.isfinite(arr).all():
            raise ValueError(f'{name}: contains NaN/Inf')
        names.append(name)
        arrays.append(arr)

    # Same within-user rank transform blend_validation.py uses, so the combiner
    # learns to weight calibrated within-user preferences, not raw score scales.
    feats = np.stack([within_user_ranks(users, a) for a in arrays], axis=1).astype(np.float64)

    n = len(rows)
    rng = np.random.default_rng(args.seed)
    fold_id = rng.integers(0, args.folds, size=n)

    oof_scores = np.zeros(n, dtype=np.float64)
    fold_weights = []
    for f in range(args.folds):
        held_out = fold_id == f
        w, b = fit_logreg(feats[~held_out], labels[~held_out])
        oof_scores[held_out] = sigmoid(feats[held_out] @ w + b)
        fold_weights.append(w)

    metrics = evaluate(users, labels, oof_scores)
    print(f'stacked logistic combiner (out-of-fold, {args.folds}-fold) | '
          f'GAUC {metrics["GAUC"]:.4f} | nDCG@5 {metrics["nDCG@5"]:.4f} | primary {metrics["primary"]:.4f}')
    avg_w = np.mean(fold_weights, axis=0)
    for name, wt in zip(names, avg_w):
        print(f'  avg learned weight[{name}] = {wt:.4f}')


if __name__ == '__main__':
    main()
