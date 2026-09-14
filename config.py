"""
Central configuration experiment.
"""
from __future__ import annotations
DATASET_REGISTRY = {
    "dilbert":    (41163, "multiclass"),
    "christine":  (41142, "binary"),
    "volkert":    (41166, "multiclass"),
    "yolanda":    (42705, "regression"),
    "arcene":     (1458,  "binary"),
    "higgs":      (23512, "binary"),
    "miniboone":  (41150, "binary"),
    "philippine": (41145, "binary"),
    "phoneme":    (1489,  "binary"),
    "wine":       (40498, "multiclass"),
    "space":      (507,   "regression"),
    "housing":    (537,   "regression"),
}

DATASETS = ["phoneme", "space", "wine", "housing", "higgs", "miniboone",
            "yolanda", "volkert", "philippine", "christine", "dilbert", "arcene"]

MAX_SAMPLES = 5000
MAX_FEATURES = 500
SUBSAMPLE_SEED = 20240714

N_FOLDS = 5
CV_SEED = 0
MISSING_RATES = [0.2, 0.5]
MECHANISMS = ["MCAR", "MNAR"]
IMPUTATIONS = ["single", "multiple"]
INDICATORS = ["none", "MIM", "SMIM"]
MODELS = ["linear", "xgb", "mlp"]

M_IMPUTATIONS = 5

SMIM_ALPHA = 0.05
MNAR_POWER = 2.0
MNAR_INF_PROBS = [0.1, 0.5]
IMPUTER_MAX_ITER = 10
IMPUTER_N_NEAREST_FEATURES = 50
MIN_OBSERVED_PER_COL = 5

MLP_PARAMS = dict(hidden_layer_sizes=(32, 32), alpha=0.0,
                  batch_size=128, max_iter=30, solver="adam")

RESULTS_CSV = "outputs/results.csv"
OUTPUT_DIR = "outputs"