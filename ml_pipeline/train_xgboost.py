"""
XGBoost — main leakage prediction model.

Uses early stopping on the validation set, scale_pos_weight for class
imbalance, and outputs both a probability score and a binary prediction.

Output columns: xgb_leakage_prob, xgb_leakage_pred
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from ml_pipeline.utils import (
    get_logger, save_model, load_model, prepare_X_y,
)
from ml_pipeline.feature_engineering import INVOICE_FEATURE_COLS

log = get_logger("train_xgboost")

MODEL_NAME = "xgboost_v1"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Tuned conservatively for generalisation over raw accuracy.
XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}


def _scale_pos_weight(y: pd.Series) -> float:
    """Compensates for class imbalance: neg_count / pos_count."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    log.info("scale_pos_weight = %.4f  (neg=%d  pos=%d)", spw, neg, pos)
    return float(spw)


def train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    early_stopping_rounds: int = 40,
) -> xgb.XGBClassifier:
    """
    Train XGBoost with early stopping evaluated on the validation AUC.
    The best iteration's model is automatically selected.
    """
    X_train, y_train = prepare_X_y(df_train, feature_cols)
    X_val, y_val = prepare_X_y(df_val, feature_cols)

    log.info(
        "Training XGBoost — train: %d  val: %d  features: %d",
        len(X_train), len(X_val), X_train.shape[1],
    )
    log.info("Leakage rate — train: %.2f%%  val: %.2f%%",
             y_train.mean() * 100, y_val.mean() * 100)

    params = {
        **XGB_PARAMS,
        "scale_pos_weight": _scale_pos_weight(y_train),
    }

    model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log.info("Best iteration: %d  |  Val ROC-AUC: %.4f", best_iter, val_auc)

    save_model(model, MODEL_NAME)
    return model


def score(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Score df with the trained XGBoost model.
    Adds columns: xgb_leakage_prob, xgb_leakage_pred
    """
    X, _ = prepare_X_y(df, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = df.copy()
    result["xgb_leakage_prob"] = probs
    result["xgb_leakage_pred"] = preds

    log.info(
        "XGB scored %d rows — mean prob: %.3f  predicted positive: %.2f%%",
        len(df), probs.mean(), preds.mean() * 100,
    )
    return result


def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Returns a DataFrame of feature importances sorted descending."""
    available = [c for c in feature_cols]
    importances = model.feature_importances_
    # align with available features that were actually used
    n = min(len(importances), len(available))
    fi = pd.DataFrame({
        "feature": available[:n],
        "importance": importances[:n],
    }).sort_values("importance", ascending=False)
    return fi.head(top_n)


def load_and_score(
    df: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    model = load_model(MODEL_NAME)
    return score(model, df, feature_cols=feature_cols, threshold=threshold)
