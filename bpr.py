"""BPR (Bayesian Personalized Ranking) loss on top of the same FM parameterization.

Hypothesis (see README 'headroom' list, item 1): the task is scored with ranking
metrics (GAUC, nDCG@5) but the baseline trains pointwise logloss. Training a loss
that directly optimizes within-user pairwise order should align better with what's
scored, without touching features or model capacity (both already shown to be
saturated in ablation_features.py).

Sampling: for each user with at least one positive AND one negative row in TRAIN,
draw random (pos_row, neg_row) pairs from that user's own impressions. This mirrors
exactly how GAUC/nDCG are computed (within-user, using the discriminative subset).
Users that are all-positive or all-negative contribute no pairs, matching how GAUC
already excludes them.

Reuses baseline.FM's V/W/b parameters and Adam state; only the gradient (pairwise
vs pointwise) and the sampler differ.
"""
import time
import numpy as np
from baseline import FM, sigmoid
from data import encode
from evaluate import evaluate


def build_pair_index(X, y, users):
    """Group row indices by user into positive/negative pools for pair sampling."""
    by_user = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, [[], []])[int(y[i])].append(i)
    valid_users = [u for u, (neg, pos) in by_user.items() if neg and pos]
    pos_concat, pos_start, pos_count = [], [], []
    neg_concat, neg_start, neg_count = [], [], []
    for u in valid_users:
        neg, pos = by_user[u]
        pos_start.append(len(pos_concat)); pos_count.append(len(pos)); pos_concat.extend(pos)
        neg_start.append(len(neg_concat)); neg_count.append(len(neg)); neg_concat.extend(neg)
    return dict(
        pos_concat=np.asarray(pos_concat, dtype=np.int64),
        pos_start=np.asarray(pos_start, dtype=np.int64),
        pos_count=np.asarray(pos_count, dtype=np.int64),
        neg_concat=np.asarray(neg_concat, dtype=np.int64),
        neg_start=np.asarray(neg_start, dtype=np.int64),
        neg_count=np.asarray(neg_count, dtype=np.int64),
        n_users=len(valid_users),
    )


def sample_pairs(idx, rng, n_pairs):
    u = rng.integers(0, idx['n_users'], size=n_pairs)
    po = rng.integers(0, 2**31 - 1, size=n_pairs) % idx['pos_count'][u]
    no = rng.integers(0, 2**31 - 1, size=n_pairs) % idx['neg_count'][u]
    pos_rows = idx['pos_concat'][idx['pos_start'][u] + po]
    neg_rows = idx['neg_concat'][idx['neg_start'][u] + no]
    return pos_rows, neg_rows


def bpr_step(m, Xpos, Xneg):
    """One Adam step on a batch of (pos,neg) row pairs. Mirrors FM.step's gradient
    accumulation but with pairwise gradient sigmoid(z_pos - z_neg) - 1 in place of
    the pointwise sigmoid(z) - y."""
    B = len(Xpos)
    zp, Ep, Sp = m.logits(Xpos)
    zn, En, Sn = m.logits(Xneg)
    d = (sigmoid(zp - zn) - 1.0) / B          # dL/dz_pos ; dL/dz_neg = -d
    d = d.astype(np.float32)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, Xpos, d[:, None])
    np.add.at(gW, Xneg, -d[:, None])
    np.add.at(gV, Xpos, d[:, None, None] * (Sp[:, None, :] - Ep))
    np.add.at(gV, Xneg, -d[:, None, None] * (Sn[:, None, :] - En))
    gV += m.l2 * m.V; gW += m.l2 * m.W

    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * d.sum()
    return float(-np.mean(np.log(sigmoid(zp - zn) + 1e-9)))


def run_bpr(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0,
            steps_per_epoch=None, l2=1e-6, verbose=True, evaluate_test=False,
            validation_scores_path=None):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    idx = build_pair_index(Xtr, ytr, utr)
    if verbose:
        print(f"  BPR: {idx['n_users']} users with both pos & neg rows in train "
              f"(out of {len(set(utr))} total)")

    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    spe = steps_per_epoch or max(1, len(ytr) // bs)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(spe):
            pr, nr = sample_pairs(idx, rng, bs)
            losses.append(bpr_step(m, Xtr[pr], Xtr[nr]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | bpr_loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    validation_scores = m.predict(Xva)
    if validation_scores_path:
        np.save(validation_scores_path, validation_scores)
    out = {'valid': evaluate(uva, yva, validation_scores)}
    # Mirror baseline.run_fm: test stays opt-in only, never touched during
    # iterative development or blend screening.
    if evaluate_test:
        Xte, yte, ute = enc['test']
        out['test'] = evaluate(ute, yte, m.predict(Xte))
    return out


if __name__ == '__main__':
    import argparse
    from data import load
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--validation_scores_path',
                    help='optional .npy path for validation scores in log row order')
    ap.add_argument('--validation_only', action='store_true',
                    help='do not evaluate test (required for development experiments)')
    a = ap.parse_args()
    splits = load(a.data_dir)
    res = run_bpr(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, l2=a.l2, patience=a.patience,
                  evaluate_test=not a.validation_only, validation_scores_path=a.validation_scores_path)
    print(f"\n=== bpr (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        if sp in res:
            r = res[sp]
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
