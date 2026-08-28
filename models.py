"""Model wrappers: FM (re-export) and LightGBM optional wrapper."""
try:
    import lightgbm as lgb
except Exception:
    lgb = None

import numpy as np

from baseline import FM as FMBase


class FMModel:
    def __init__(self, dim, k=16, lr=0.001, seed=0):
        self.model = FMBase(dim, k=k, lr=lr, seed=seed)

    def fit(self, X, y, epochs=40, bs=8192):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        for ep in range(epochs):
            for i in range(0, len(idx), bs):
                self.model.step(X[idx[i:i+bs]], y[idx[i:i+bs]])

    def predict(self, X):
        return self.model.predict(X)


class LightGBMModel:
    def __init__(self, params=None):
        if lgb is None:
            raise ImportError("lightgbm not available")
        self.params = params or {"objective": "binary", "metric": "binary_logloss"}
        self.bst = None

    def fit(self, X, y, valid=None, num_round=100):
        dtrain = lgb.Dataset(X, label=y)
        evals = None
        if valid is not None:
            Xv, yv = valid
            deval = lgb.Dataset(Xv, label=yv)
            evals = [deval]
        self.bst = lgb.train(self.params, dtrain, num_boost_round=num_round, valid_sets=evals, verbose_eval=False)

    def predict(self, X):
        return self.bst.predict(X)


class NumpyNNModel:
    """A tiny fully-connected network implemented in NumPy for binary prediction.
    Architecture: input -> Dense(hidden, relu) -> Dense(1, sigmoid).
    """
    def __init__(self, input_dim, hidden=64, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.01, (input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.01, (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X, y, epochs=10, bs=4096):
        N = len(y)
        for ep in range(epochs):
            idx = np.random.permutation(N)
            for i in range(0, N, bs):
                batch = idx[i:i+bs]
                xb = X[batch].astype(np.float32)
                yb = y[batch].astype(np.float32)
                h = xb.dot(self.W1) + self.b1
                h_relu = np.maximum(h, 0)
                logits = h_relu.dot(self.W2) + self.b2
                pred = self._sigmoid(logits).reshape(-1)
                # binary cross-entropy gradient
                g = (pred - yb)[:, None] / len(yb)
                # backprop
                dW2 = h_relu.T.dot(g)
                db2 = g.sum(0)
                dh = g.dot(self.W2.T)
                dh[h <= 0] = 0
                dW1 = xb.T.dot(dh)
                db1 = dh.sum(0)
                # update
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

    def predict(self, X):
        xb = X.astype(np.float32)
        h = xb.dot(self.W1) + self.b1
        h_relu = np.maximum(h, 0)
        logits = h_relu.dot(self.W2) + self.b2
        return self._sigmoid(logits).reshape(-1)
