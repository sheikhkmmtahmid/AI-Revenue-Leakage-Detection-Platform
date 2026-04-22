"""
Isolation Forest — unsupervised anomaly detection.

No labels are needed. The model learns the normal revenue pattern
and flags invoices that deviate significantly from it.

Output columns added to the feature df:
  if_score      — raw anomaly score (lower = more anomalous in sklearn convention)
  if_is_anomaly — 1 if flagged as anomaly, else 0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml_pipeline.utils import (
    get_logger, save_model, load_model, prepare_X_y,
    MODELS_DIR,
)
from ml_pipeline.feature_engineering import INVOICE_FEATURE_COLS

log = get_logger("train_anomaly")

# Contamination: expected fraction of anomalies.
# Our injection rates: missing_payment 6% + underbilling 4% + dup_refund 3%
# + discount_abuse 5% + payment_delay 12% → ~25% worst-case unique events.
CONTAMINATION = 0.25
N_ESTIMATORS = 200
RANDOM_STATE = 42
MODEL_NAME = "isolation_forest_v1"


def build_pipeline(contamination: float = CONTAMINATION) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("iso_forest", IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            max_samples="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def train(
    df_train: pd.DataFrame,
    contamination: float = CONTAMINATION,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
) -> Pipeline:
    """
    Fit Isolation Forest on training split.
    Isolation Forest is unsupervised — labels are NOT used during fitting.
    """
    X_train, _ = prepare_X_y(df_train, feature_cols)
    log.info("Training Isolation Forest on %d samples × %d features", *X_train.shape)

    pipe = build_pipeline(contamination)
    pipe.fit(X_train)
    log.info("Training complete. Contamination=%.2f  n_estimators=%d", contamination, N_ESTIMATORS)

    save_model(pipe, MODEL_NAME)
    return pipe


def score(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Score a DataFrame.
    Returns the original df with two new columns:
      if_score      — decision_function output (negative = anomalous)
      if_is_anomaly — binary flag (1 = anomaly)

    threshold: override the model's built-in threshold.
               If None, uses the model's predict() which respects contamination.
    """
    X, _ = prepare_X_y(df, feature_cols)
    scores = pipe.decision_function(X)       # lower → more anomalous
    if threshold is not None:
        labels = (scores < threshold).astype(int)
    else:
        raw_pred = pipe.predict(X)           # sklearn: -1 = anomaly, 1 = normal
        labels = (raw_pred == -1).astype(int)

    result = df.copy()
    result["if_score"] = scores
    result["if_is_anomaly"] = labels

    anomaly_rate = labels.mean()
    log.info(
        "Scored %d rows — anomaly rate: %.2f%% (threshold=%s)",
        len(df), anomaly_rate * 100, threshold,
    )
    return result


def train_and_score(
    df_train: pd.DataFrame,
    df_full: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
    contamination: float = CONTAMINATION,
) -> tuple[Pipeline, pd.DataFrame]:
    """Convenience wrapper: train on df_train, score df_full."""
    pipe = train(df_train, contamination=contamination, feature_cols=feature_cols)
    scored = score(pipe, df_full, feature_cols=feature_cols)
    return pipe, scored


def load_and_score(
    df: pd.DataFrame,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
) -> pd.DataFrame:
    """Load saved model and score df."""
    pipe = load_model(MODEL_NAME)
    return score(pipe, df, feature_cols=feature_cols)
