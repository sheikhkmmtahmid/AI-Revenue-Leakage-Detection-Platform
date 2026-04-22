"""
Logistic Regression — baseline supervised leakage classifier.

Uses a Pipeline of StandardScaler + LogisticRegression with cross-validated
threshold tuning. Gives a calibrated probability baseline against which
XGBoost is compared.

Output column: lr_leakage_prob, lr_leakage_pred
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml_pipeline.utils import (
    get_logger, save_model, load_model, prepare_X_y,
)
from ml_pipeline.feature_engineering import INVOICE_FEATURE_COLS

log = get_logger("train_baseline")

MODEL_NAME = "logistic_regression_v1"
CV_FOLDS = 5
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", lr),
    ])


def train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
) -> Pipeline:
    """
    Train Logistic Regression with 5-fold stratified CV.
    LR with lbfgs produces well-calibrated probabilities natively,
    so no separate calibration step is needed.
    The val set is used for logging only (LR doesn't use early stopping).
    """
    X_train, y_train = prepare_X_y(df_train, feature_cols)
    X_val, y_val = prepare_X_y(df_val, feature_cols)

    log.info(
        "Training Logistic Regression -- train: %d  val: %d  features: %d",
        len(X_train), len(X_val), X_train.shape[1],
    )
    log.info("Leakage rate -- train: %.2f%%  val: %.2f%%",
             y_train.mean() * 100, y_val.mean() * 100)

    pipe = build_pipeline()

    # 5-fold CV on training data
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info("CV ROC-AUC: %.4f +/- %.4f", cv_aucs.mean(), cv_aucs.std())

    # Final fit on full training split
    pipe.fit(X_train, y_train)

    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    log.info("Val ROC-AUC: %.4f", val_auc)

    save_model(pipe, MODEL_NAME)
    return pipe


def score(
    model,
    df: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Score df with the calibrated LR model.
    Adds columns: lr_leakage_prob, lr_leakage_pred
    """
    X, _ = prepare_X_y(df, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = df.copy()
    result["lr_leakage_prob"] = probs
    result["lr_leakage_pred"] = preds

    log.info(
        "LR scored %d rows — mean prob: %.3f  predicted positive: %.2f%%",
        len(df), probs.mean(), preds.mean() * 100,
    )
    return result


def load_and_score(
    df: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    threshold: float = 0.5,
) -> pd.DataFrame:
    model = load_model(MODEL_NAME)
    return score(model, df, feature_cols=feature_cols, threshold=threshold)
