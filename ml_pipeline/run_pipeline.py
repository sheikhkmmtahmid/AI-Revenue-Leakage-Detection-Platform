"""
Master ML pipeline runner.

Executes the full cycle:
  1. Load invoice features from MySQL
  2. Engineer features
  3. Time-based train / val / test split (70 / 15 / 15)
  4. Train Isolation Forest  (unsupervised)
  5. Train Logistic Regression (baseline)
  6. Train XGBoost  (main model)
  7. Evaluate all three on the test set
  8. Print comparison table
  9. Run inference on the FULL dataset
 10. Write anomaly_scores + risk_scores to MySQL

Usage:
    venv/Scripts/python.exe ml_pipeline/run_pipeline.py

Optional flags:
    --skip-train    skip training, load saved models from disk
    --no-db-write   skip writing scores to MySQL (for fast local testing)
"""

import argparse
import os
import sys
import time

# ── Bootstrap Django ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

# ── Imports after Django setup ────────────────────────────────────────────────
import numpy as np
import pandas as pd

from ml_pipeline.data_loading import load_invoice_features
from ml_pipeline.feature_engineering import (
    engineer_invoice_features,
    INVOICE_FEATURE_COLS,
)
from ml_pipeline.utils import (
    get_logger,
    time_split,
    prepare_X_y,
    load_model,
    save_report,
)
from ml_pipeline import train_anomaly, train_baseline, train_xgboost
from ml_pipeline.evaluation import (
    evaluate_classifier,
    evaluate_isolation_forest,
    compare_models,
)
from ml_pipeline.inference import run_inference, run_inference_from_disk

log = get_logger("run_pipeline")


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_load_and_engineer() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_full_engineered, df_raw).
    df_raw keeps invoice_id + customer_id + issue_date for splitting and DB writes.
    df_full_engineered has all ML features plus meta columns.
    """
    log.info("Loading invoice features from MySQL…")
    df_raw = load_invoice_features()
    log.info("  Loaded %d invoices", len(df_raw))

    # Stash meta columns before engineering drops them
    meta_cols = ["invoice_id", "customer_id", "subscription_id"]
    date_col = "issue_date"

    log.info("Engineering features…")
    df_eng = engineer_invoice_features(df_raw)

    # Re-attach meta + date so we can split and write to DB
    for col in meta_cols + [date_col]:
        if col in df_raw.columns and col not in df_eng.columns:
            df_eng[col] = df_raw[col].values

    log.info("  Feature matrix: %d rows × %d cols", *df_eng.shape)
    log.info("  Leakage rate: %.2f%%", df_eng["leakage_label"].mean() * 100)

    return df_eng, df_raw


def stage_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return time_split(df, date_col="issue_date", train_pct=0.70, val_pct=0.15)


def stage_train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple:
    log.info("=" * 55)
    log.info("Training Isolation Forest…")
    iso_pipe = train_anomaly.train(df_train, feature_cols=INVOICE_FEATURE_COLS)

    log.info("=" * 55)
    log.info("Training Logistic Regression (baseline)…")
    lr_model = train_baseline.train(df_train, df_val, feature_cols=INVOICE_FEATURE_COLS)

    log.info("=" * 55)
    log.info("Training XGBoost…")
    xgb_model = train_xgboost.train(df_train, df_val, feature_cols=INVOICE_FEATURE_COLS)

    return iso_pipe, lr_model, xgb_model


def stage_evaluate(
    df_test: pd.DataFrame,
    iso_pipe,
    lr_model,
    xgb_model,
) -> list[dict]:
    log.info("=" * 55)
    log.info("Evaluating on test set (%d rows)…", len(df_test))

    X_test, y_test = prepare_X_y(df_test, INVOICE_FEATURE_COLS)

    # ── Isolation Forest ──
    scored_iso = train_anomaly.score(iso_pipe, df_test, feature_cols=INVOICE_FEATURE_COLS)
    r_iso = evaluate_isolation_forest(y_test, scored_iso["if_score"].values,
                                      model_name="isolation_forest_v1")

    # ── Logistic Regression ──
    scored_lr = train_baseline.score(lr_model, df_test, feature_cols=INVOICE_FEATURE_COLS)
    r_lr = evaluate_classifier(y_test, scored_lr["lr_leakage_prob"].values,
                               model_name="logistic_regression_v1")

    # ── XGBoost ──
    scored_xgb = train_xgboost.score(xgb_model, df_test, feature_cols=INVOICE_FEATURE_COLS)
    r_xgb = evaluate_classifier(y_test, scored_xgb["xgb_leakage_prob"].values,
                                model_name="xgboost_v1")

    # ── Ensemble ──
    ens_prob = (
        0.70 * scored_xgb["xgb_leakage_prob"].values +
        0.30 * scored_lr["lr_leakage_prob"].values
    )
    r_ens = evaluate_classifier(y_test, ens_prob, model_name="ensemble_v1")

    reports = [r_iso, r_lr, r_xgb, r_ens]

    log.info("=" * 55)
    log.info("Model comparison:")
    cmp_df = compare_models(reports)
    log.info("\n%s", cmp_df.to_string())

    # Feature importance from XGBoost
    fi = train_xgboost.get_feature_importance(xgb_model, INVOICE_FEATURE_COLS, top_n=15)
    log.info("\nTop 15 XGBoost features:\n%s", fi.to_string(index=False))
    fi.to_csv(str(__import__("ml_pipeline.utils", fromlist=["REPORTS_DIR"])
                  .REPORTS_DIR / "xgb_feature_importance.csv"), index=False)

    return reports


def stage_inference(
    df_full: pd.DataFrame,
    iso_pipe,
    lr_model,
    xgb_model,
    write_to_db: bool = True,
) -> pd.DataFrame:
    log.info("=" * 55)
    log.info("Running inference on full dataset (%d invoices)…", len(df_full))

    if write_to_db:
        scored = run_inference(df_full, iso_pipe, lr_model, xgb_model)
    else:
        from ml_pipeline.inference import score_all_models
        scored = score_all_models(df_full, iso_pipe, lr_model, xgb_model)
        log.info("(DB write skipped — --no-db-write flag set)")

    return scored


# ── Entry-point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Revenue Leakage ML Pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load saved models from disk, skip training")
    parser.add_argument("--no-db-write", action="store_true",
                        help="Score but do not write results to MySQL")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 55)
    log.info("AI Revenue Leakage — ML Pipeline")
    log.info("=" * 55)

    # Stage 1: Load & engineer
    df_full, df_raw = stage_load_and_engineer()

    # Stage 2: Split
    df_train, df_val, df_test = stage_split(df_full)

    if args.skip_train:
        # Load pre-trained models
        log.info("Loading saved models from disk (--skip-train)…")
        iso_pipe = load_model("isolation_forest_v1")
        lr_model = load_model("logistic_regression_v1")
        xgb_model = load_model("xgboost_v1")
    else:
        # Stage 3: Train
        iso_pipe, lr_model, xgb_model = stage_train(df_train, df_val)

    # Stage 4: Evaluate
    reports = stage_evaluate(df_test, iso_pipe, lr_model, xgb_model)

    # Stage 5: Inference on full dataset + DB write
    scored = stage_inference(
        df_full, iso_pipe, lr_model, xgb_model,
        write_to_db=not args.no_db_write,
    )

    elapsed = time.time() - t0
    log.info("=" * 55)
    log.info("Pipeline complete in %.1fs", elapsed)
    log.info("Scored invoices: %d", len(scored))
    if not args.no_db_write:
        from apps.anomaly_detection.models import AnomalyScore
        from apps.risk_scoring.models import RiskScore
        log.info("anomaly_scores rows: %d", AnomalyScore.objects.count())
        log.info("risk_scores rows:    %d", RiskScore.objects.count())
    log.info("Reports saved to: artifacts/reports/")
    log.info("Models saved to:  artifacts/models/")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
