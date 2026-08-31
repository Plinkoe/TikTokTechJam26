"""Simple feature engineering helpers for tabular models (LightGBM) and FM-compatible encodings.
"""
import collections
import numpy as np


def build_tabular_features(splits):
    """Build small, dense feature vectors per row using train statistics.

    Features per row:
      - item_count, item_pos_rate
      - user_count, user_pos_rate
      - author_count, author_pos_rate
      - tab (one-hot with small vocabulary)
      - dur_bucket (one-hot, quantile-bucketed into 10 bins from train durations)
    Returns dict with per-split arrays (X, y, users).
    """
    tr = splits['train']
    item_cnt = collections.Counter()
    item_pos = collections.Counter()
    user_cnt = collections.Counter()
    user_pos = collections.Counter()
    author_cnt = collections.Counter()
    author_pos = collections.Counter()
    tabs = set()

    for x in tr:
        user = x[1]; vid = x[2]; author = x[3]; tab = x[4]
        label = x[6]
        item_cnt[vid] += 1; item_pos[vid] += label
        user_cnt[user] += 1; user_pos[user] += label
        author_cnt[author] += 1; author_pos[author] += label
        tabs.add(tab)

    # Quantile-bucket raw duration_ms into 10 bins (mirrors data.py's encode()).
    # The previous version one-hotted the raw millisecond value directly, which
    # has thousands of distinct values in train and blew up the feature matrix.
    dur_edges = np.quantile(np.asarray([x[5] for x in tr], dtype=np.float32),
                             np.linspace(0, 1, 11)[1:-1])
    n_dur_buckets = len(dur_edges) + 1

    tab_list = sorted(list(tabs))
    tab_idx = {v: i for i, v in enumerate(tab_list)}

    def make_for(rws):
        X = []
        y = []
        users = []
        for x in rws:
            user = x[1]; vid = x[2]; author = x[3]; tab = x[4]; dur = x[5]
            label = x[6]
            f = []
            # counts and rates (use log(1+cnt) and rate)
            ic = item_cnt.get(vid, 0); ip = item_pos.get(vid, 0)
            uc = user_cnt.get(user, 0); up = user_pos.get(user, 0)
            ac = author_cnt.get(author, 0); ap = author_pos.get(author, 0)
            f.extend([np.log1p(ic), (ip / ic) if ic else 0.0,
                      np.log1p(uc), (up / uc) if uc else 0.0,
                      np.log1p(ac), (ap / ac) if ac else 0.0])
            # tab one-hot
            tb = [0.0] * len(tab_list)
            if tab in tab_idx:
                tb[tab_idx[tab]] = 1.0
            f.extend(tb)
            # dur bucket one-hot (quantile bucket, not raw duration_ms)
            db = [0.0] * n_dur_buckets
            db[int(np.searchsorted(dur_edges, dur))] = 1.0
            f.extend(db)
            X.append(f); y.append(label); users.append(user)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), users

    out = {}
    for name, rws in splits.items():
        out[name] = make_for(rws)
    return out
