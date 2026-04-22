"""
Inference layer — batch-scores invoices and writes results to MySQL.

Tables written:
  anomaly_scores  — Isolation Forest output
  risk_scores     — XGBoost probability + LR probability (ensemble average)

Call patterns:
  1. Full pipeline (after training):
       run_inference(df_engineered, df_raw, iso_pipe, lr_model, xgb_model)

  2. Inference-only (models already saved):
       run_inference_from_disk(df_engineered, df_raw)
"""

import numpy as np
import pandas as pd
from django.utils import timezone

from ml_pipeline.utils import get_logger, load_model, prepare_X_y
from ml_pipeline.feature_engineering import INVOICE_FEATURE_COLS
from ml_pipeline import train_anomaly, train_baseline, train_xgboost

log = get_logger("inference")

# Probability threshold above which an invoice is marked as a leakage risk
DEFAULT_THRESHOLD = 0.50
# Ensemble: weighted average XGBoost 70% + LR 30%
ENSEMBLE_XGB_WEIGHT = 0.70
ENSEMBLE_LR_WEIGHT = 0.30

BATCH_SIZE = 2_000   # DB write batch size


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_all_models(
    df_engineered: pd.DataFrame,
    iso_pipe,
    lr_model,
    xgb_model,
    feature_cols: list[str] = INVOICE_FEATURE_COLS,
) -> pd.DataFrame:
    """
    Runs all three models and assembles a scored DataFrame.
    Expected input: output of engineer_invoice_features() with
    invoice_id and customer_id columns still present.
    """
    log.info("Scoring %d invoices with all models…", len(df_engineered))

    scored = df_engineered.copy()

    # Isolation Forest
    scored = train_anomaly.score(iso_pipe, scored, feature_cols=feature_cols)

    # Logistic Regression
    scored = train_baseline.score(lr_model, scored, feature_cols=feature_cols)

    # XGBoost
    scored = train_xgboost.score(xgb_model, scored, feature_cols=feature_cols)

    # Ensemble probability (weighted average)
    scored["ensemble_prob"] = (
        ENSEMBLE_XGB_WEIGHT * scored["xgb_leakage_prob"] +
        ENSEMBLE_LR_WEIGHT * scored["lr_leakage_prob"]
    )
    scored["ensemble_pred"] = (scored["ensemble_prob"] >= DEFAULT_THRESHOLD).astype(int)

    # Normalised IF score proxy (0–1, higher = more anomalous)
    s = -scored["if_score"]
    scored["if_prob_proxy"] = (s - s.min()) / (s.max() - s.min() + 1e-9)

    log.info(
        "Scoring complete — IF anomalies: %.1f%%  XGB positive: %.1f%%  Ensemble: %.1f%%",
        scored["if_is_anomaly"].mean() * 100,
        scored["xgb_leakage_pred"].mean() * 100,
        scored["ensemble_pred"].mean() * 100,
    )
    return scored


def _risk_severity(prob: float) -> str:
    if prob >= 0.80:
        return "critical"
    elif prob >= 0.60:
        return "high"
    elif prob >= 0.40:
        return "medium"
    return "low"


# ── DB writers ────────────────────────────────────────────────────────────────

def write_anomaly_scores(scored: pd.DataFrame, model_version: str = "v1"):
    """
    Bulk-insert AnomalyScore rows for every scored invoice.
    Skips invoices that already have a score for this model_version.
    """
    from apps.anomaly_detection.models import AnomalyScore

    # Determine which invoice_ids already have scores to avoid duplicates
    existing_ids = set(
        AnomalyScore.objects.filter(model_version=model_version)
        .values_list("invoice_id", flat=True)
    )

    rows = []
    now = timezone.now()
    period_month = now.strftime("%Y-%m")

    for _, row in scored.iterrows():
        inv_id = row.get("invoice_id")
        if not inv_id or inv_id in existing_ids:
            continue

        feature_snap = {
            "if_score": round(float(row["if_score"]), 6),
            "if_prob_proxy": round(float(row["if_prob_proxy"]), 6),
            "top_features": _extract_top_invoice_features(row),
        }

        rows.append(AnomalyScore(
            customer_id=int(row["customer_id"]),
            invoice_id=int(inv_id),
            model_version=model_version,
            score=float(row["if_score"]),
            is_anomaly=bool(row["if_is_anomaly"]),
            threshold_used=-0.0,   # Isolation Forest predict() default
            feature_snapshot=feature_snap,
            scored_at=now,
            period_month=period_month,
        ))

        if len(rows) >= BATCH_SIZE:
            AnomalyScore.objects.bulk_create(rows, ignore_conflicts=True)
            log.info("  Inserted %d anomaly scores (batch)…", BATCH_SIZE)
            rows = []

    if rows:
        AnomalyScore.objects.bulk_create(rows, ignore_conflicts=True)

    total = AnomalyScore.objects.count()
    log.info("AnomalyScore table: %d rows total", total)


def write_risk_scores(scored: pd.DataFrame, model_version: str = "v1"):
    """
    Bulk-insert RiskScore rows (XGBoost + ensemble) for every scored invoice.
    """
    from apps.risk_scoring.models import RiskScore

    existing_ids = set(
        RiskScore.objects.filter(model_version=model_version, model_name="xgboost")
        .values_list("invoice_id", flat=True)
    )

    rows = []
    now = timezone.now()
    period_month = now.strftime("%Y-%m")

    for _, row in scored.iterrows():
        inv_id = row.get("invoice_id")
        if not inv_id or inv_id in existing_ids:
            continue

        xgb_prob = float(row["xgb_leakage_prob"])
        ens_prob = float(row["ensemble_prob"])
        top_feats = _extract_top_invoice_features(row)

        rows.append(RiskScore(
            customer_id=int(row["customer_id"]),
            invoice_id=int(inv_id),
            model_name="xgboost",
            model_version=model_version,
            leakage_probability=xgb_prob,
            risk_severity=_risk_severity(xgb_prob),
            rank_percentile=None,        # computed after full batch (see below)
            feature_snapshot={
                "xgb_prob": round(xgb_prob, 6),
                "lr_prob": round(float(row["lr_leakage_prob"]), 6),
                "ensemble_prob": round(ens_prob, 6),
                "if_anomaly": int(row["if_is_anomaly"]),
            },
            shap_values={},              # populated by explainability.py
            top_features=top_feats,
            scored_at=now,
            period_month=period_month,
        ))

        if len(rows) >= BATCH_SIZE:
            RiskScore.objects.bulk_create(rows, ignore_conflicts=True)
            log.info("  Inserted %d risk scores (batch)…", BATCH_SIZE)
            rows = []

    if rows:
        RiskScore.objects.bulk_create(rows, ignore_conflicts=True)

    # Back-fill rank_percentile in a single UPDATE
    _backfill_percentiles(model_version)

    total = RiskScore.objects.count()
    log.info("RiskScore table: %d rows total", total)


def _backfill_percentiles(model_version: str):
    """Compute rank_percentile for all RiskScore rows with this version."""
    from apps.risk_scoring.models import RiskScore
    from django.db import connection

    with connection.cursor() as cur:
        cur.execute("""
            UPDATE risk_scores rs
            JOIN (
                SELECT id,
                       PERCENT_RANK() OVER (ORDER BY leakage_probability) AS pct
                FROM risk_scores
                WHERE model_version = %s AND model_name = 'xgboost'
            ) ranked ON rs.id = ranked.id
            SET rs.rank_percentile = ranked.pct * 100
            WHERE rs.model_version = %s AND rs.model_name = 'xgboost'
        """, [model_version, model_version])
    log.info("Rank percentiles back-filled for version=%s", model_version)


def _extract_top_invoice_features(row: pd.Series, top_n: int = 5) -> list[str]:
    """
    Heuristic: return the names of the features with the highest absolute
    values for this invoice (proxy for what made it unusual).
    """
    numeric_feats = [
        "failed_payment_rate", "overdue_days", "contract_gap",
        "refund_ratio", "discount_ratio", "outstanding_ratio",
        "billed_vs_expected_ratio", "duplicate_refund_count",
        "payment_delay_ratio", "retry_rate",
    ]
    vals = {f: abs(float(row[f])) for f in numeric_feats if f in row.index}
    top = sorted(vals, key=vals.get, reverse=True)[:top_n]
    return top


# ── Master inference entry-points ─────────────────────────────────────────────

def run_inference(
    df_engineered: pd.DataFrame,
    iso_pipe,
    lr_model,
    xgb_model,
    model_version: str = "v1",
) -> pd.DataFrame:
    """
    Score all invoices and write results to MySQL.
    Returns the fully-scored DataFrame.
    """
    scored = score_all_models(df_engineered, iso_pipe, lr_model, xgb_model)

    log.info("Writing anomaly scores to DB…")
    write_anomaly_scores(scored, model_version=model_version)

    log.info("Writing risk scores to DB…")
    write_risk_scores(scored, model_version=model_version)

    return scored


def run_inference_from_disk(
    df_engineered: pd.DataFrame,
    model_version: str = "v1",
) -> pd.DataFrame:
    """Load saved models from disk and run full inference."""
    log.info("Loading saved models from disk…")
    iso_pipe = load_model("isolation_forest_v1")
    lr_model = load_model("logistic_regression_v1")
    xgb_model = load_model("xgboost_v1")
    return run_inference(df_engineered, iso_pipe, lr_model, xgb_model, model_version)
