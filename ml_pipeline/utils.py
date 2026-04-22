"""
Shared utilities for the ML pipeline:
  - artifact path resolution
  - model serialisation (joblib)
  - time-based train / val / test split
  - standardised logger
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Project root & artifact dirs ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SHAP_DIR = ARTIFACTS_DIR / "shap"
FORECASTS_DIR = ARTIFACTS_DIR / "forecasts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for _d in (MODELS_DIR, SHAP_DIR, FORECASTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Logger ────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Artifact helpers ──────────────────────────────────────────────────────────

def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"


def save_model(obj, name: str) -> Path:
    path = model_path(name)
    joblib.dump(obj, path)
    get_logger("utils").info("Saved model → %s", path)
    return path


def load_model(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_report(data: dict, name: str) -> Path:
    path = REPORTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    get_logger("utils").info("Saved report → %s", path)
    return path


# ── Time-based train / val / test split ──────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    date_col: str = "issue_date",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits df chronologically — no data leakage from the future into training.
    Returns (train, val, test) DataFrames.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    log = get_logger("utils")
    log.info(
        "Time split — train: %d (%.0f%%)  val: %d (%.0f%%)  test: %d (%.0f%%)",
        len(train), 100 * train_pct,
        len(val), 100 * val_pct,
        len(test), 100 * (1 - train_pct - val_pct),
    )
    if date_col in train.columns:
        log.info(
            "Date ranges -- train: %s to %s  |  val: %s to %s  |  test: %s to %s",
            train[date_col].min().date(), train[date_col].max().date(),
            val[date_col].min().date(),   val[date_col].max().date(),
            test[date_col].min().date(),  test[date_col].max().date(),
        )
    return train, val, test


# ── Feature matrix helpers ────────────────────────────────────────────────────

def prepare_X_y(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "leakage_label",
    drop_cols: tuple = ("invoice_id", "customer_id", "issue_date", "subscription_id"),
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Returns (X, y) ready for sklearn.
    Fills remaining NaN with 0 and clips infinities.
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    y = df[target_col].copy() if target_col in df.columns else None
    return X, y


def leakage_rate(y: pd.Series) -> float:
    return float(y.mean())
