"""Cohort error analysis: where does the champion (causal-history DeepFM) win
or lose relative to FM, and where does it still fall short of "good"?

Never touches test. Reuses history_model.load_history_rows for the exact
validation row order that run_logs/*_seed0.npy were saved in (loader order,
not time-sorted — see history_model.build_causal_features docstring).
"""
import collections
import csv
import os

import numpy as np

from evaluate import evaluate
from history_model import load_history_rows

DATA_DIR = './KuaiRand-Pure/data'


def load_user_features(data_dir):
    out = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            out[r['user_id']] = r
    return out


def bucket_count(n, edges, labels):
    for e, lab in zip(edges, labels):
        if n <= e:
            return lab
    return labels[-1]


def main():
    rows = load_history_rows(DATA_DIR)
    train, valid = rows['train'], rows['valid']
    user_feat = load_user_features(DATA_DIR)

    # Train-only popularity / activity counts (no leakage: valid never mutates these).
    user_train_count = collections.Counter(r.user for r in train)
    user_train_pos = collections.Counter()
    for r in train:
        if r.label:
            user_train_pos[r.user] += 1
    video_train_count = collections.Counter(r.video for r in train)

    va_users = [r.user for r in valid]
    va_labels = np.array([r.label for r in valid], dtype=np.float32)

    champion = np.load('run_logs/history_deepfm_seed0.npy')
    fm = np.load('run_logs/fm_seed0.npy')
    assert len(champion) == len(fm) == len(valid), 'row-order mismatch — do not proceed'

    def cohort_table(name, keyfn, order=None):
        by_key = collections.defaultdict(list)
        for i, r in enumerate(valid):
            by_key[keyfn(r)].append(i)
        keys = order if order else sorted(by_key)
        print(f'\n### {name}')
        header = f'{"segment":<18}{"rows":>8}{"users":>8}{"champion":>10}{"fm":>10}{"delta":>8}'
        print(header)
        print('-' * len(header))
        overall_rows = len(valid)
        for k in keys:
            idx = by_key.get(k, [])
            if not idx:
                continue
            idx = np.array(idx)
            u = [va_users[i] for i in idx]
            y = va_labels[idx]
            m_champ = evaluate(u, y, champion[idx])
            m_fm = evaluate(u, y, fm[idx])
            pct = 100 * len(idx) / overall_rows
            print(f'{str(k):<18}{len(idx):>8}{m_champ["users"]:>8}'
                  f'{m_champ["primary"]:>10.4f}{m_fm["primary"]:>10.4f}'
                  f'{m_champ["primary"] - m_fm["primary"]:>8.4f}'
                  f'   ({pct:.1f}% of rows)')

    # 1. Cold-start vs warm users (clean per-user cohort: a user's train count
    #    doesn't change across their valid rows).
    def cold_warm_key(r):
        n = user_train_count.get(r.user, 0)
        return bucket_count(n, [0, 2, 5, 15, 50, 10**9],
                             ['0 (cold)', '1-2', '3-5', '6-15', '16-50', '50+'])
    cohort_table('User train-interaction count (cold-start check)', cold_warm_key,
                 order=['0 (cold)', '1-2', '3-5', '6-15', '16-50', '50+'])

    # 2. User's historical positive rate in train (engagement level), warm users only.
    def engagement_key(r):
        n = user_train_count.get(r.user, 0)
        if n == 0:
            return 'cold'
        rate = user_train_pos.get(r.user, 0) / n
        return bucket_count(rate, [0.0, 0.1, 0.3, 0.6, 1.01],
                             ['0%', '<10%', '10-30%', '30-60%', '60-100%'])
    cohort_table('User historical positive rate (train)', engagement_key,
                 order=['cold', '0%', '<10%', '10-30%', '30-60%', '60-100%'])

    # 3. Video popularity in train (unseen items are the hardest cold-item case).
    def video_pop_key(r):
        n = video_train_count.get(r.video, 0)
        return bucket_count(n, [0, 2, 10, 50, 10**9],
                             ['unseen', '1-2', '3-10', '11-50', '50+'])
    cohort_table('Video train-impression count (item cold-start)', video_pop_key,
                 order=['unseen', '1-2', '3-10', '11-50', '50+'])

    # 4. User activity segment from user_features_pure.csv — NOT used by any
    #    current model. Purely diagnostic: is there headroom here?
    def active_degree_key(r):
        f = user_feat.get(r.user)
        return f['user_active_degree'] if f else 'unknown'
    cohort_table('user_active_degree (unused feature — diagnostic only)', active_degree_key)

    # 5. Tab (surface) the impression was served on.
    def tab_key(r):
        return r.tab
    cohort_table('tab', tab_key)

    # 6. Video tag/category.
    def tag_key(r):
        return r.tag
    cohort_table('video tag', tag_key)

    print('\nOverall (sanity check against README figures):')
    print('  champion:', evaluate(va_users, va_labels, champion))
    print('  fm      :', evaluate(va_users, va_labels, fm))


if __name__ == '__main__':
    main()
