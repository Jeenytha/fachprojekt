"""
Imputation strategies.
"""
from __future__ import annotations

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import IMPUTER_MAX_ITER, IMPUTER_N_NEAREST_FEATURES


def _make_imputer(sample_posterior: bool, random_state: int):
    return make_pipeline(
        StandardScaler(),
        IterativeImputer(sample_posterior=sample_posterior,
                         max_iter=IMPUTER_MAX_ITER,
                         n_nearest_features=IMPUTER_N_NEAREST_FEATURES,
                         random_state=random_state),
    )


def impute(strategy: str, X_train_m, X_test_m, m: int, seed: int):
    pairs = []
    if strategy == "single":
        imp = _make_imputer(sample_posterior=False, random_state=seed)
        Xtr = imp.fit_transform(X_train_m)
        Xte = imp.transform(X_test_m)
        pairs.append((Xtr, Xte))
    elif strategy == "multiple":
        for k in range(m):
            imp = _make_imputer(sample_posterior=True, random_state=seed * 100 + k)
            Xtr = imp.fit_transform(X_train_m)
            Xte = imp.transform(X_test_m)
            pairs.append((Xtr, Xte))
    else:
        raise ValueError(f"Unknown imputation strategy '{strategy}'")
    return pairs