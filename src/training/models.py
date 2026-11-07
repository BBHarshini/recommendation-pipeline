"""Recommendation models.

* PopularityRecommender - baseline: recommends the most popular items the user has not seen.
* ImplicitALSRecommender - matrix factorisation for implicit feedback (Hu, Koren & Volinsky, 2008),
  trained with alternating least squares on "who rated what", ratings used as confidence.
* ExplicitMFRecommender  - biased matrix factorisation (Funk-SVD style) trained with SGD to
  predict the star rating itself; recommendations are the highest predicted ratings.

All models share the same interface: fit(train, store) and recommend(user_ids, k, seen).
"""

import numpy as np
import pandas as pd
from scipy import sparse


class BaseRecommender:
    name = "base"
    predicts_ratings = False

    def __init__(self, min_item_ratings=1, seed=42, **params):
        self.min_item_ratings = min_item_ratings
        self.seed = seed
        self.params = params

    def fit(self, train, store):
        raise NotImplementedError

    def scores(self, user_id):
        """Score for every item in self.item_ids (higher = better)."""
        raise NotImplementedError

    def _set_catalog(self, train, store):
        self.item_ids = np.sort(train["item_id"].unique())
        eligible = set(store.eligible_items(self.min_item_ratings))
        self.eligible_mask = np.isin(self.item_ids, list(eligible))
        self.item_pos = {it: j for j, it in enumerate(self.item_ids)}

    def recommend(self, user_ids, k, seen):
        """Top-k unseen, eligible items for each user -> DataFrame(user_id, rank, item_id, score)."""
        rows = []
        for u in user_ids:
            s = self.scores(u).astype(float).copy()
            s[~self.eligible_mask] = -np.inf
            seen_pos = [self.item_pos[i] for i in seen.get(u, ()) if i in self.item_pos]
            s[seen_pos] = -np.inf
            top = np.argpartition(-s, k)[:k]
            top = top[np.argsort(-s[top])]
            rows += [(u, r + 1, int(self.item_ids[j]), float(s[j])) for r, j in enumerate(top)]
        return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "score"])


class PopularityRecommender(BaseRecommender):
    """Same list for everyone: items ranked by popularity (or Bayesian average rating)."""
    name = "popularity"
    predicts_ratings = True     # predicts the item's Bayesian average rating

    def fit(self, train, store):
        self._set_catalog(train, store)
        feats = store.item_features(self.item_ids)
        column = "item_popularity" if self.params.get("score", "count") == "count" else "item_bayes_rating"
        self._scores = feats[column].to_numpy(dtype=float)
        self._rating = feats["item_bayes_rating"].to_numpy(dtype=float)
        self.global_mean = float(train["rating"].mean())
        return self

    def scores(self, user_id):
        return self._scores

    def predict_ratings(self, user_ids, item_ids):
        return np.array([self._rating[self.item_pos[i]] if i in self.item_pos else self.global_mean
                         for i in item_ids])


class ImplicitALSRecommender(BaseRecommender):
    name = "als_implicit"

    def fit(self, train, store):
        self._set_catalog(train, store)
        f = self.params.get("factors", 32)
        reg = self.params.get("regularization", 0.1)
        alpha = self.params.get("alpha", 10.0)
        iters = self.params.get("iterations", 15)
        rng = np.random.default_rng(self.seed)

        self.user_ids = np.sort(train["user_id"].unique())
        self.user_pos = {u: i for i, u in enumerate(self.user_ids)}
        rows = train["user_id"].map(self.user_pos).to_numpy()
        cols = train["item_id"].map(self.item_pos).to_numpy()
        conf = 1.0 + alpha * train["rating"].to_numpy(dtype=float) / 5.0
        C = sparse.csr_matrix((conf, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))

        X = rng.normal(0, 0.01, (C.shape[0], f))
        Y = rng.normal(0, 0.01, (C.shape[1], f))
        for _ in range(iters):
            X = self._solve(C, Y, reg)
            Y = self._solve(C.T.tocsr(), X, reg)
        self.X, self.Y = X, Y
        return self

    @staticmethod
    def _solve(C, Y, reg):
        """Closed-form least squares for each row of C, holding the other factor fixed."""
        f = Y.shape[1]
        YtY = Y.T @ Y
        out = np.zeros((C.shape[0], f))
        for u in range(C.shape[0]):
            start, end = C.indptr[u], C.indptr[u + 1]
            idx, c = C.indices[start:end], C.data[start:end]
            Yu = Y[idx]
            A = YtY + (Yu.T * (c - 1.0)) @ Yu + reg * np.eye(f)
            b = (Yu * c[:, None]).sum(axis=0)        # preference p = 1 for observed items
            out[u] = np.linalg.solve(A, b)
        return out

    def scores(self, user_id):
        if user_id not in self.user_pos:
            return np.zeros(len(self.item_ids))
        return self.Y @ self.X[self.user_pos[user_id]]


class ExplicitMFRecommender(BaseRecommender):
    name = "mf_explicit"
    predicts_ratings = True

    def fit(self, train, store):
        self._set_catalog(train, store)
        f = self.params.get("factors", 32)
        lr = self.params.get("learning_rate", 0.01)
        reg = self.params.get("regularization", 0.05)
        epochs = self.params.get("epochs", 30)
        rng = np.random.default_rng(self.seed)

        self.user_ids = np.sort(train["user_id"].unique())
        self.user_pos = {u: i for i, u in enumerate(self.user_ids)}
        u_idx = train["user_id"].map(self.user_pos).to_numpy()
        i_idx = train["item_id"].map(self.item_pos).to_numpy()
        r = train["rating"].to_numpy(dtype=float)

        self.mu = r.mean()
        self.bu = np.zeros(len(self.user_ids))
        self.bi = np.zeros(len(self.item_ids))
        self.P = rng.normal(0, 0.1, (len(self.user_ids), f))
        self.Q = rng.normal(0, 0.1, (len(self.item_ids), f))
        self.train_rmse = []

        for _ in range(epochs):
            sq_err = 0.0
            for n in rng.permutation(len(r)):
                u, i = u_idx[n], i_idx[n]
                pu, qi = self.P[u].copy(), self.Q[i]
                err = r[n] - (self.mu + self.bu[u] + self.bi[i] + pu @ qi)
                sq_err += err * err
                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                self.P[u] += lr * (err * qi - reg * pu)
                self.Q[i] += lr * (err * pu - reg * qi)
            self.train_rmse.append(float(np.sqrt(sq_err / len(r))))
        return self

    def scores(self, user_id):
        if user_id not in self.user_pos:
            return self.mu + self.bi
        u = self.user_pos[user_id]
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]

    def predict_ratings(self, user_ids, item_ids):
        preds = []
        for u, i in zip(user_ids, item_ids):
            p = self.mu
            if u in self.user_pos:
                p += self.bu[self.user_pos[u]]
            if i in self.item_pos:
                p += self.bi[self.item_pos[i]]
                if u in self.user_pos:
                    p += self.P[self.user_pos[u]] @ self.Q[self.item_pos[i]]
            preds.append(p)
        return np.clip(np.array(preds), 1.0, 5.0)


MODEL_CLASSES = {cls.name: cls for cls in [PopularityRecommender, ImplicitALSRecommender, ExplicitMFRecommender]}
