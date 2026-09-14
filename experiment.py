"""
Experiment driver.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
import zlib

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import KFold, StratifiedKFold

import config as C
from data import load_dataset
from missingness import (apply_mask, choose_informative_columns, make_mcar_mask,
                         make_mnar_mask)
from indicators import TEST_FAILURES, indicator_block, select_indicators
from imputation import impute
from models import METRIC_NAME, fit_pool_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FIELDS = ["dataset", "task", "n", "d", "fold", "missing_rate", "mechanism",
          "inf_prob", "imputation", "indicator", "model", "metric", "score",
          "n_models", "n_indicators", "impute_time", "train_time", "status",
          "error"]


def _seq(dataset_name, fold, rate_idx, mech_idx):
    did = zlib.crc32(dataset_name.encode()) & 0xFFFFFFFF
    return np.random.default_rng(
        np.random.SeedSequence([did, fold, rate_idx, mech_idx]))


def _key(dname, fold, rate, mech, inf_prob, imp, ind, model):
    return (dname, str(fold), str(rate), mech, str(inf_prob), imp, ind, model)


def _load_completed(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in "ok":
                done.add(_key(row["dataset"], row["fold"], row["missing_rate"],
                              row["mechanism"], row.get("inf_prob", "NA"),
                              row["imputation"], row["indicator"], row["model"]))
    return done


def _make_masks(X_train, X_test, mechanism, rate, inf_prob, rng):
    if mechanism == "MCAR":
        tr = make_mcar_mask(X_train, rate, rng)
        te = make_mcar_mask(X_test, rate, rng)
        informative = np.zeros(X_train.shape[1], dtype=bool)
    elif mechanism == "MNAR":
        informative = choose_informative_columns(X_train.shape[1],
                                                 inf_prob, rng)
        tr = make_mnar_mask(X_train, rate, informative, C.MNAR_POWER, rng)
        te = make_mnar_mask(X_test, rate, informative, C.MNAR_POWER, rng)
    else:
        raise ValueError(mechanism)
    return tr, te, informative


def _make_cv(task):
    if task == "regression":
        return KFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.CV_SEED)
    return StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True,
                           random_state=C.CV_SEED)


def run(datasets, resume=False):
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    path = C.RESULTS_CSV
    completed = _load_completed(path) if resume else set()
    mode = "a" if (resume and os.path.exists(path)) else "w"
    print(f"Mode: {'RESUME' if mode == 'a' else 'FRESH'} "
          f"({len(completed)} cells already done)")

    with open(path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()

        for dname in datasets:
            try:
                X, y, task = load_dataset(dname)
            except Exception as e:
                print(f"[skip dataset] {dname}: {e}")
                continue
            metric = METRIC_NAME[task]
            print(f"\n=== {dname}  (task={task}, shape={X.shape}) ===")

            cv = _make_cv(task)
            splits = list(cv.split(X, y))

            for fold, (tr_idx, te_idx) in enumerate(splits):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                for r_idx, rate in enumerate(C.MISSING_RATES):
                    for m_idx, mech in enumerate(C.MECHANISMS):
                        inf_probs = C.MNAR_INF_PROBS if mech == "MNAR" else [None]
                        for inf_prob in inf_probs:
                            ip = "NA" if inf_prob is None else inf_prob
                            rng = _seq(dname, fold, r_idx, m_idx)
                            tr_mask, te_mask, informative = _make_masks(
                                X_tr, X_te, mech, rate, inf_prob, rng)
                            X_tr_m = apply_mask(X_tr, tr_mask)
                            X_te_m = apply_mask(X_te, te_mask)

                            cols = {s: select_indicators(s, tr_mask, y_tr, task,
                                                         C.SMIM_ALPHA, informative)
                                    for s in C.INDICATORS}
                            blocks = {s: (indicator_block(tr_mask, cols[s]),
                                          indicator_block(te_mask, cols[s]))
                                      for s in C.INDICATORS}

                            for imp in C.IMPUTATIONS:
                                needed = [(ind, model)
                                          for ind in C.INDICATORS
                                          for model in C.MODELS
                                          if _key(dname, fold, rate, mech, ip, imp,
                                                  ind, model) not in completed]
                                if not needed:
                                    continue

                                t0 = time.perf_counter()
                                try:
                                    pairs = impute(imp, X_tr_m, X_te_m,
                                                   C.M_IMPUTATIONS, fold)
                                    impute_time = time.perf_counter() - t0
                                    imp_error = None
                                except Exception as e:
                                    impute_time = time.perf_counter() - t0
                                    pairs, imp_error = None, repr(e)

                                for ind, model in needed:
                                    row = dict(
                                        dataset=dname, task=task, n=X.shape[0],
                                        d=X.shape[1], fold=fold,
                                        missing_rate=rate, mechanism=mech,
                                        inf_prob=ip, imputation=imp, indicator=ind,
                                        model=model, metric=metric, score="",
                                        n_models="", n_indicators=cols[ind].size,
                                        impute_time=round(impute_time, 3),
                                        train_time="", status="", error="")

                                    if pairs is None:
                                        row["status"] = "failed"
                                        row["error"] = f"impute: {imp_error}"
                                        writer.writerow(row); fh.flush(); continue

                                    t1 = time.perf_counter()
                                    try:
                                        loss, nmod = fit_pool_score(
                                            model, task, fold, pairs,
                                            blocks[ind], y_tr, y_te)
                                        row["score"] = round(loss, 6)
                                        row["n_models"] = nmod
                                        row["status"] = "ok"
                                    except Exception as e:
                                        row["status"] = "failed"
                                        row["error"] = repr(e)
                                    row["train_time"] = round(
                                        time.perf_counter() - t1, 3)
                                    writer.writerow(row); fh.flush()

                            print(f"  fold={fold} rate={rate} {mech} "
                                  f"inf_prob={ip}: done")
    print(f"\nResults written to {path}")
    if TEST_FAILURES["count"]:
        print(f"WARNING: {TEST_FAILURES['count']} SMIM association tests raised "
              f"and were treated as uninformative (last: "
              f"{TEST_FAILURES['last_error']}). SMIM selection is biased toward "
              f"selecting nothing; investigate before trusting SMIM rows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present (ok) in results.csv")
    args = ap.parse_args()
    run(datasets=C.DATASETS, resume=args.resume)

if __name__ == "__main__":
    main()