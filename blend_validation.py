"""Evaluate validation-only, within-user rank-normalized score blends.

Inputs must be one-dimensional ``.npy`` score arrays aligned to
``data.load(data_dir)['valid']``.  This command intentionally has no test mode:
choose a blend weight on development data before the one-time final test score.
"""
import argparse

import numpy as np

from data import load
from evaluate import evaluate


def within_user_ranks(users, scores):
    """Map scores to [0, 1] percentile ranks independently for every user."""
    out = np.empty(len(scores), dtype=np.float32)
    positions = {}
    for index, user in enumerate(users):
        positions.setdefault(user, []).append(index)
    for indices in positions.values():
        indices = np.asarray(indices)
        order = np.argsort(scores[indices], kind='mergesort')
        if len(indices) == 1:
            out[indices] = 0.5
        else:
            ranks = np.empty(len(indices), dtype=np.float32)
            sorted_scores = scores[indices][order]
            start = 0
            while start < len(indices):
                end = start + 1
                while end < len(indices) and sorted_scores[end] == sorted_scores[start]:
                    end += 1
                # Preserve score ties: otherwise an arbitrary stable-sort order
                # could create a synthetic within-user preference.
                ranks[order[start:end]] = ((start + end - 1) / 2) / (len(indices) - 1)
                start = end
            out[indices] = ranks
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--champion', required=True, help='champion validation .npy score file')
    parser.add_argument('--fm', required=True, help='FM validation .npy score file')
    parser.add_argument('--weights', type=float, nargs='+', default=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    args = parser.parse_args()
    rows = load(args.data_dir)['valid']
    champion, fm = np.load(args.champion), np.load(args.fm)
    if champion.ndim != 1 or fm.ndim != 1 or len(champion) != len(rows) or len(fm) != len(rows):
        raise ValueError('both score arrays must be 1-D and match the validation row count')
    if not (np.isfinite(champion).all() and np.isfinite(fm).all()):
        raise ValueError('score arrays must not contain NaN or Inf')
    users = [row[1] for row in rows]
    labels = [row[6] for row in rows]
    champion_rank, fm_rank = within_user_ranks(users, champion), within_user_ranks(users, fm)
    best = None
    for weight in args.weights:
        if not 0 <= weight <= 1:
            raise ValueError('weights must be in [0, 1]')
        metrics = evaluate(users, labels, weight * champion_rank + (1 - weight) * fm_rank)
        print(f'champion_weight={weight:.3f} | GAUC {metrics["GAUC"]:.4f} | '
              f'nDCG@5 {metrics["nDCG@5"]:.4f} | primary {metrics["primary"]:.4f}')
        if best is None or metrics['primary'] > best[1]['primary']:
            best = (weight, metrics)
    print(f'best validation blend: champion_weight={best[0]:.3f}, primary={best[1]["primary"]:.4f}')


if __name__ == '__main__':
    main()
