from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from config import MLP_PARAMS

METRIC_NAME = {"binary": "1-AUC", "multiclass": "1-accuracy", "regression": "RMSE"}

def make_model(model_name: str, task: str, seed: int):
    is_clf = task in ("binary", "multiclass")
    if model_name == "linear":
        if is_clf:
            return LogisticRegression(C=1e12, max_iter=1000)
        return LinearRegression()
    if model_name == "xgb":
        if is_clf:
            return XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=seed)
        return XGBRegressor(n_jobs=1, random_state=seed)
    if model_name == "mlp":
        if is_clf:
            return MLPClassifier(random_state=seed, **MLP_PARAMS)
        return MLPRegressor(random_state=seed, **MLP_PARAMS)
    raise ValueError(f"Unknown model '{model_name}'")

def _predict_one(model, X_test, task, classes):
    if task == "regression":
        return model.predict(X_test)
    proba = model.predict_proba(X_test)
    full = np.zeros((X_test.shape[0], len(classes)))
    for i, c in enumerate(model.classes_):
        full[:, list(classes).index(c)] = proba[:, i]
    return full

def fit_pool_score(model_name, task, seed, imputed_pairs, indicator_blocks,
                   y_train, y_test):
    tr_block, te_block = indicator_blocks
    classes = np.unique(y_train) if task != "regression" else None

    preds = []
    for (Xtr_imp, Xte_imp) in imputed_pairs:
        Xtr = np.hstack([Xtr_imp, tr_block])
        Xte = np.hstack([Xte_imp, te_block])
        model = make_model(model_name, task, seed)
        model.fit(Xtr, y_train)
        preds.append(_predict_one(model, Xte, task, classes))

    pooled = np.mean(preds, axis=0)

    if task == "regression":
        loss = float(mean_squared_error(y_test, pooled)) ** 0.5
    elif task == "binary":
        loss = 1.0 - float(roc_auc_score(y_test, pooled[:, 1]))
    else:
        y_pred = classes[np.argmax(pooled, axis=1)]
        loss = 1.0 - float(accuracy_score(y_test, y_pred))
    return loss, len(preds)