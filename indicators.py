"""
Missing-indicator construction and selection.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency, ttest_ind


def candidate_columns(train_mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(train_mask.any(axis=0))

def _benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        rejected[order[: kmax + 1]] = True
    return rejected

TEST_FAILURES: dict[str, int] = {"count": 0, "last_error": ""}

def _indicator_target_pvalue(ind: np.ndarray, y: np.ndarray, task: str) -> float:
    if np.unique(ind).size < 2:
        return 1.0
    try:
        if task == "regression":
            g0, g1 = y[ind == 0], y[ind == 1]
            if g0.size < 2 or g1.size < 2:
                return 1.0
            return float(ttest_ind(g0, g1, equal_var=False).pvalue)
        else:
            classes = np.unique(y)
            table = np.array([[np.sum((ind == k) & (y == c)) for c in classes]
                              for k in (0, 1)], dtype=float)
            if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
                return 1.0
            return float(chi2_contingency(table)[1])
    except Exception as e:
        TEST_FAILURES["count"] += 1
        TEST_FAILURES["last_error"] = repr(e)
        return 1.0

def select_indicators(strategy: str, train_mask: np.ndarray, y_train: np.ndarray,
                      task: str, alpha: float) -> np.ndarray:
    cand = candidate_columns(train_mask)
    if strategy == "none" or cand.size == 0:
        return np.array([], dtype=int)
    if strategy == "MIM":
        return cand
    if strategy == "SMIM":
        pvals = np.array([_indicator_target_pvalue(train_mask[:, j].astype(int),
                                                   y_train, task) for j in cand])
        rejected = _benjamini_hochberg(pvals, alpha)
        return cand[rejected]
    raise ValueError(f"Unknown indicator strategy '{strategy}'")

def indicator_block(mask: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if cols.size == 0:
        return np.empty((mask.shape[0], 0), dtype=float)
    return mask[:, cols].astype(float)