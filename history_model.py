"""Validation-only causal-history DeepFM for KuaiRand-Pure.

Every train feature is computed using events strictly earlier than that row's
``time_ms``.  Validation features are built from the completed training history
only; validation labels are never used as features or to update state.  This is
therefore safe to use in the experiment loop without inspecting test labels.
"""
import argparse
import collections
import copy
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate import evaluate


@dataclass(frozen=True)
class Row:
    time_ms: int
    date: int
    hour: int
    user: str
    video: str
    author: str
    tag: str
    music: str
    video_type: str
    tab: str
    duration: float
    upload_date: int
    label: int


def _date_int(value):
    try:
        return int(value.replace('-', '')[:8])
    except (AttributeError, ValueError):
        return 0


@lru_cache(maxsize=None)
def _ordinal(date):
    """Convert YYYYMMDD to a calendar day without repeatedly parsing it."""
    if not date:
        return 0
    return datetime.strptime(str(date), '%Y%m%d').toordinal()


def load_history_rows(data_dir):
    """Load only development splits with serving-time item metadata."""
    meta = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            meta[r['video_id']] = (
                r.get('author_id', 'UNK'), r.get('tag', 'UNK'),
                r.get('music_id', 'UNK'), r.get('video_type', 'UNK'),
                _date_int(r.get('upload_dt', '')),
            )

    out = {'train': [], 'valid': []}
    for filename in ('log_standard_4_08_to_4_21_pure.csv',
                     'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, filename)) as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                split = 'train' if 20220408 <= date <= 20220421 else (
                    'valid' if 20220422 <= date <= 20220428 else None)
                if split is None:
                    continue
                author, tag, music, video_type, upload_date = meta.get(
                    r['video_id'], ('UNK', 'UNK', 'UNK', 'UNK', 0))
                out[split].append(Row(
                    time_ms=int(r['time_ms']), date=date,
                    hour=int(r['hourmin']) // 100, user=r['user_id'],
                    video=r['video_id'], author=author, tag=tag, music=music,
                    video_type=video_type, tab=r['tab'],
                    duration=float(r['duration_ms']), upload_date=upload_date,
                    label=int(r['long_view'] != '0'),
                ))
    return out


def _duration_bucket(values):
    edges = np.quantile(np.asarray(values, dtype=np.float32), np.linspace(0, 1, 11)[1:-1])
    return lambda x: str(int(np.searchsorted(edges, x)))


def _age_bucket(row):
    age = max(0, _ordinal(row.date) - _ordinal(row.upload_date)) if row.upload_date else 9999
    return str(int(np.searchsorted([1, 7, 30, 90, 365], age)))


def _fields(row, dur_bucket):
    return (row.user, row.video, row.author, row.tag, row.music, row.video_type,
            row.tab, dur_bucket(row.duration), str(row.hour), _age_bucket(row))


def _rate(pos, count, prior=8.0, base=0.33):
    return (pos + prior * base) / (count + prior)


def build_causal_features(rows_by_split, history_len=0):
    """Return encoded categorical/dense data, preserving original row order.

    State advances only for train rows after their feature vector is materialized.
    Validation is read-only against final train state.
    """
    train, valid = rows_by_split['train'], rows_by_split['valid']
    dur_bucket = _duration_bucket([r.duration for r in train])
    field_names = ('user', 'video', 'author', 'tag', 'music', 'video_type', 'tab',
                   'duration_bucket', 'hour', 'video_age_bucket')
    vocabs = [dict() for _ in field_names]
    for r in train:
        for i, value in enumerate(_fields(r, dur_bucket)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i]) + 1  # 0 is UNK
    vocab_sizes = [len(v) + 1 for v in vocabs]

    # Counters keyed by entities available when an impression is served.
    video = collections.defaultdict(lambda: [0, 0])
    author = collections.defaultdict(lambda: [0, 0])
    user_author = collections.defaultdict(lambda: [0, 0])
    user_tag = collections.defaultdict(lambda: [0, 0])
    user_total = collections.defaultdict(lambda: [0, 0])
    last_positive = {}
    # This survives train -> validation.  Validation reads it but never mutates it.
    positive_history = (collections.defaultdict(lambda: collections.deque(maxlen=history_len))
                        if history_len else None)

    def make(split_rows, advance):
        cats = np.zeros((len(split_rows), len(field_names)), dtype=np.int64)
        dense = np.zeros((len(split_rows), 10), dtype=np.float32)
        # int32 halves the storage cost of the large, fixed-width sequence.
        # Zero is both the vocabulary UNK/padding slot and an attention mask.
        histories = (np.zeros((len(split_rows), history_len), dtype=np.int32)
                     if history_len else None)
        labels = np.empty(len(split_rows), dtype=np.float32)
        users = [None] * len(split_rows)
        # Sorting changes only the state-processing order; output stays aligned
        # with the loader/submission order.
        ordered = sorted(range(len(split_rows)), key=lambda j: split_rows[j].time_ms)
        cursor = 0
        while cursor < len(ordered):
            # A timestamp tie is simultaneous, not "earlier".  Materialize all
            # features in the tie group before advancing the history state.
            end = cursor + 1
            timestamp = split_rows[ordered[cursor]].time_ms
            while end < len(ordered) and split_rows[ordered[end]].time_ms == timestamp:
                end += 1
            group = ordered[cursor:end]
            for i in group:
                r = split_rows[i]
                for j, value in enumerate(_fields(r, dur_bucket)):
                    cats[i, j] = vocabs[j].get(value, 0)
                if history_len:
                    prior = positive_history[r.user]
                    if prior:
                        histories[i, -len(prior):] = prior
                vc, ac = video[r.video], author[r.author]
                uac, utc, ut = user_author[(r.user, r.author)], user_tag[(r.user, r.tag)], user_total[r.user]
                since = min(30.0, max(0.0, (r.time_ms - last_positive.get(r.user, r.time_ms)) / 86_400_000.0))
                dense[i] = (
                    np.log1p(vc[0]), _rate(vc[1], vc[0]),
                    np.log1p(ac[0]), _rate(ac[1], ac[0]),
                    np.log1p(uac[0]), _rate(uac[1], uac[0]),
                    np.log1p(utc[0]), _rate(utc[1], utc[0]),
                    np.log1p(ut[1]), np.log1p(since),
                )
                labels[i] = r.label
                users[i] = r.user
            if advance:
                for i in group:
                    r = split_rows[i]
                    vc, ac = video[r.video], author[r.author]
                    uac, utc, ut = user_author[(r.user, r.author)], user_tag[(r.user, r.tag)], user_total[r.user]
                    for stat in (vc, ac, uac, utc, ut):
                        stat[0] += 1
                        stat[1] += r.label
                    if r.label:
                        last_positive[r.user] = r.time_ms
                        if history_len:
                            positive_history[r.user].append(cats[i, 1])
            cursor = end
        return cats, dense, histories, labels, users

    train_data = make(train, advance=True)
    valid_data = make(valid, advance=False)
    mean = train_data[1].mean(axis=0, keepdims=True)
    std = train_data[1].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_data = (train_data[0], (train_data[1] - mean) / std, train_data[2], train_data[3], train_data[4])
    valid_data = (valid_data[0], (valid_data[1] - mean) / std, valid_data[2], valid_data[3], valid_data[4])
    return train_data, valid_data, vocab_sizes, field_names


class HistoryDeepFM(nn.Module):
    def __init__(self, vocab_sizes, dense_dim, emb_dim=12, hidden=96, sequence_attention=False):
        super().__init__()
        self.sequence_attention = sequence_attention
        self.linear = nn.ModuleList([nn.Embedding(n, 1) for n in vocab_sizes])
        self.embed = nn.ModuleList([nn.Embedding(n, emb_dim) for n in vocab_sizes])
        # The attention branch contributes a history context and its interaction
        # with the candidate-video embedding.  It is absent in the control.
        sequence_dim = 2 * emb_dim if sequence_attention else 0
        input_dim = len(vocab_sizes) * emb_dim + dense_dim + sequence_dim
        self.deep = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.dense_linear = nn.Linear(dense_dim, 1)
        for layer in list(self.linear) + list(self.embed):
            nn.init.normal_(layer.weight, std=0.01)
            with torch.no_grad():
                layer.weight[0].zero_()

    def forward(self, cats, dense, history=None):
        e = torch.stack([layer(cats[:, i]) for i, layer in enumerate(self.embed)], dim=1)
        summed = e.sum(dim=1)
        fm = 0.5 * ((summed * summed).sum(dim=1) - (e * e).sum(dim=(1, 2)))
        linear = torch.cat([layer(cats[:, i]) for i, layer in enumerate(self.linear)], dim=1).sum(dim=1)
        deep_input = [e.flatten(1), dense]
        if self.sequence_attention:
            if history is None:
                raise ValueError('sequence_attention=True requires a history tensor')
            # Candidate-to-history dot-product attention (DIN-style).  Only
            # preceding long-view videos are present; zero padding is masked.
            query = self.embed[1](cats[:, 1])
            keys = self.embed[1](history.long())
            logits = (keys * query.unsqueeze(1)).sum(dim=-1) / np.sqrt(query.shape[1])
            mask = history.ne(0)
            logits = logits.masked_fill(~mask, -1e9)
            weights = torch.softmax(logits, dim=1) * mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            context = (weights.unsqueeze(-1) * keys).sum(dim=1)
            deep_input.extend([context, query * context])
        deep = self.deep(torch.cat(deep_input, dim=1)).squeeze(1)
        return linear + fm + self.dense_linear(dense).squeeze(1) + deep


def _predict(model, cats, dense, history, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(cats), batch_size):
            h = history[i:i + batch_size] if history is not None else None
            out.append(model(cats[i:i + batch_size], dense[i:i + batch_size], h).cpu().numpy())
    return np.concatenate(out)


def run_history_deepfm(data_dir='./KuaiRand-Pure/data', epochs=8, lr=1e-3, emb_dim=12,
                       hidden=96, batch_size=8192, patience=3, seed=0, verbose=True,
                       sequence_attention=False, history_len=20):
    """Train on train and return validation metrics only (never reads test rows)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = load_history_rows(data_dir)
    (tr_cat, tr_dense, tr_hist, tr_y, _), (va_cat, va_dense, va_hist, va_y, va_users), vocab_sizes, _ = \
        build_causal_features(rows, history_len=history_len if sequence_attention else 0)
    tr_cat = torch.from_numpy(tr_cat); tr_dense = torch.from_numpy(tr_dense); tr_y = torch.from_numpy(tr_y)
    va_cat = torch.from_numpy(va_cat); va_dense = torch.from_numpy(va_dense)
    tr_hist = torch.from_numpy(tr_hist) if tr_hist is not None else None
    va_hist = torch.from_numpy(va_hist) if va_hist is not None else None
    model = HistoryDeepFM(vocab_sizes, tr_dense.shape[1], emb_dim=emb_dim, hidden=hidden,
                          sequence_attention=sequence_attention)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    best, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tr_y))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[start:start + batch_size])
            h = tr_hist[idx] if tr_hist is not None else None
            logits = model(tr_cat[idx], tr_dense[idx], h)
            loss = F.binary_cross_entropy_with_logits(logits, tr_y[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        scores = _predict(model, va_cat, va_dense, va_hist, batch_size)
        metrics = evaluate(va_users, va_y, scores)
        if verbose:
            print(f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | valid GAUC {metrics['GAUC']:.4f} "
                  f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f}")
        if metrics['primary'] > best + 1e-5:
            best, best_state, bad = metrics['primary'], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {'valid': evaluate(va_users, va_y, _predict(model, va_cat, va_dense, va_hist, batch_size))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sequence_attention', action='store_true',
                        help='attend from each candidate video to recent positive history')
    parser.add_argument('--history_len', type=int, default=20)
    args = parser.parse_args()
    result = run_history_deepfm(**vars(args))
    m = result['valid']
    print(f"\n=== causal-history DeepFM (validation only) ===\n  GAUC {m['GAUC']:.4f} | "
          f"nDCG@5 {m['nDCG@5']:.4f} | primary {m['primary']:.4f}")
