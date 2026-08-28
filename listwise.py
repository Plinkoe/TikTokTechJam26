"""Listwise softmax loss on top of the same FM parameterization.

Rationale: BPR (bpr.py) tested the "pointwise loss doesn't match ranking metrics"
hypothesis via random pair sampling and it hurt (see run notes) -- most likely
because sampling one random (pos,neg) pair per step throws away most of a user's
list and adds sampling noise. Listwise softmax instead uses each user's FULL set
of train impressions in one shot per step: softmax over the list's scores vs. a
target that is uniform over that user's positive rows. This is a closer match to
nDCG (which scores the whole ranked list, not a single pair) and adds no sampling
noise, at the cost of only training on users with a mix of pos/neg impressions.

Only users with 0 < positive_count < list_length are used -- this is exactly the
same discriminative-user criterion evaluate.py already applies for GAUC, and it's
principled: an all-positive or all-negative list carries zero ranking signal (the
correct softmax target for an all-positive list is already the uniform prediction,
so there's nothing to learn from it).
"""
import time
import numpy as np
from baseline import FM
from data import encode
from evaluate import evaluate


def build_list_index(X, y, users):
    """Group train row indices by user, keep only discriminative users
    (0 < positives < list length), sorted contiguously for reduceat."""
    by_user = {}
    for i, u in enumerate(users):
        by_user.setdefault(u, []).append(i)
    row_concat, start, count, pos_count = [], [], [], []
    for u, rows in by_user.items():
        npos = sum(int(y[r]) for r in rows)
        if 0 < npos < len(rows):
            start.append(len(row_concat)); count.append(len(rows)); pos_count.append(npos)
            row_concat.extend(rows)
    return dict(
        row_concat=np.asarray(row_concat, dtype=np.int64),
        start=np.asarray(start, dtype=np.int64),
        count=np.asarray(count, dtype=np.int64),
        pos_count=np.asarray(pos_count, dtype=np.float32),
        n_users=len(start),
    )


def sample_batch(idx, rng, n_users_per_step):
    sel = rng.integers(0, idx['n_users'], size=n_users_per_step)
    sel_counts = idx['count'][sel]
    local_starts = np.concatenate(([0], np.cumsum(sel_counts)[:-1])).astype(np.int64)
    total_rows = int(sel_counts.sum())
    within = np.arange(total_rows) - np.repeat(local_starts, sel_counts)
    src_pos = np.repeat(idx['start'][sel], sel_counts) + within
    batch_rows = idx['row_concat'][src_pos]
    pos_count_per_row = np.repeat(idx['pos_count'][sel], sel_counts)
    return batch_rows, local_starts, sel_counts, pos_count_per_row


def listwise_step(m, X, y, local_starts, counts, pos_count_per_row):
    B = len(y)
    z, E, S = m.logits(X)
    group_max = np.maximum.reduceat(z, local_starts)
    z_shift = z - np.repeat(group_max, counts)
    expz = np.exp(z_shift)
    group_sum = np.add.reduceat(expz, local_starts)
    p = expz / np.repeat(group_sum, counts)
    t = y / pos_count_per_row
    d = ((p - t) / B).astype(np.float32)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, d[:, None])
    np.add.at(gV, X, d[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W

    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * d.sum()
    return float(-np.mean(t * np.log(p + 1e-9)) * counts.mean())


def run_listwise(splits, k=16, lr=0.001, epochs=40, users_per_step=2048, patience=4,
                  seed=0, steps_per_epoch=None, l2=1e-6, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    idx = build_list_index(Xtr, ytr, utr)
    if verbose:
        print(f"  listwise: {idx['n_users']} discriminative users in train "
              f"(out of {len(set(utr))} total)")

    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    spe = steps_per_epoch or max(1, idx['n_users'] // users_per_step)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(spe):
            rows, starts, counts, pcpr = sample_batch(idx, rng, users_per_step)
            losses.append(listwise_step(m, Xtr[rows], ytr[rows], starts, counts, pcpr))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
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
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    import argparse
    from data import load
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--users_per_step', type=int, default=2048)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    splits = load(a.data_dir)
    res = run_listwise(splits, k=a.k, lr=a.lr, epochs=a.epochs, users_per_step=a.users_per_step,
                        l2=a.l2, patience=a.patience, seed=a.seed)
    print(f"\n=== listwise (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
