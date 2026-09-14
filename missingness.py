"""
Artificial missingness generation.
"""
from __future__ import annotations

import numpy as np

from config import MIN_OBSERVED_PER_COL

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def choose_informative_columns(d: int, inf_prob: float, rng: np.random.Generator):
    informative = rng.random(d) < inf_prob
    if not informative.any():
        informative[rng.integers(d)] = True
    if d > 1 and informative.all():
        informative[rng.integers(d)] = False
    return informative

def _calibrate_intercept(logits: np.ndarray, rate: float) -> float:
    lo, hi = -50.0, 50.0
    for _ in range(60):
        b = 0.5 * (lo + hi)
        if _sigmoid(logits + b).mean() < rate:
            lo = b
        else:
            hi = b
    return 0.5 * (lo + hi)

def _enforce_observed(mask: np.ndarray, rng: np.random.Generator):
    n, d = mask.shape
    min_obs = min(MIN_OBSERVED_PER_COL, n - 1)
    for j in range(d):
        missing_idx = np.flatnonzero(mask[:, j])
        n_obs = n - missing_idx.size
        if n_obs < min_obs:
            n_to_unmask = min_obs - n_obs
            unmask = rng.choice(missing_idx, size=n_to_unmask, replace=False)
            mask[unmask, j] = False
    return mask

def make_mcar_mask(X: np.ndarray, rate: float, rng: np.random.Generator):
    mask = rng.random(X.shape) < rate
    return _enforce_observed(mask, rng)

def make_mnar_mask(X: np.ndarray, rate: float, informative: np.ndarray,
                   power: float, rng: np.random.Generator):
    n, d = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Z = (X - mu) / sd

    probs = np.empty((n, d), dtype=float)
    for j in range(d):
        lam = power if informative[j] else 0.0
        logits = lam * Z[:, j]
        b = _calibrate_intercept(logits, rate)
        probs[:, j] = _sigmoid(logits + b)

    mask = rng.random((n, d)) < probs
    return _enforce_observed(mask, rng)

def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xm = X.astype(float).copy()
    Xm[mask] = np.nan
    return Xm