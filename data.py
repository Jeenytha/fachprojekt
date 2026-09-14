"""
OpenML dataset loading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from config import (DATASET_REGISTRY, MAX_FEATURES, MAX_SAMPLES,
                    SUBSAMPLE_SEED)


def subsample_rows(X, y, task, n_max, seed):
    n = X.shape[0]
    if n_max is None or n <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    if task == "regression":
        idx = rng.choice(n, size=n_max, replace=False)
    else:
        idx_parts = []
        for c in np.unique(y):
            c_idx = np.flatnonzero(y == c)
            take = max(1, round(n_max * c_idx.size / n))
            idx_parts.append(rng.choice(c_idx, size=min(take, c_idx.size),
                                        replace=False))
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
    return X[idx], y[idx]


def cap_features(X, n_max):
    if n_max is None or X.shape[1] <= n_max:
        return X
    keep = np.argsort(X.var(axis=0))[::-1][:n_max]
    return X[:, np.sort(keep)]


def load_dataset(name: str):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(DATASET_REGISTRY)}")
    data_id, task = DATASET_REGISTRY[name]

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X_df = bunch.data.copy()
    y = bunch.target.copy()

    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.dropna(axis=1, how="all")
    mask_complete = X_df.notna().all(axis=1)
    X_df = X_df.loc[mask_complete]
    y = y.loc[mask_complete]

    nunique = X_df.nunique()
    X_df = X_df.loc[:, nunique > 1]

    X = X_df.to_numpy(dtype=float)

    if task == "regression":
        y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        keep = ~np.isnan(y)
        X, y = X[keep], y[keep]
        y = (y - y.mean()) / (y.std() + 1e-12)
    else:
        y = LabelEncoder().fit_transform(y.astype(str))
        n_classes = len(np.unique(y))
        task = "binary" if n_classes == 2 else "multiclass"

    X, y = subsample_rows(X, y, task, MAX_SAMPLES, SUBSAMPLE_SEED)
    X = cap_features(X, MAX_FEATURES)

    if X.shape[0] < 50 or X.shape[1] < 2:
        raise ValueError(f"Dataset '{name}' too small after cleaning: {X.shape}")

    return X, y, task