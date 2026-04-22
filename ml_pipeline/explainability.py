"""
SHAP Explainability — Phase 5.

Uses TreeExplainer (exact, fast for XGBoost) to compute:
  - Global feature importance (mean |SHAP|) saved to artifacts/shap/
  - Per-invoice top-3 SHAP drivers as human-readable strings
  - Updates risk_scores.shap_values + risk_scores.top_features in MySQL

Run:
    venv/Scripts/python.exe ml_pipeline/explainability.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from ml_pipeline.utils import (
    get_logger, load_model, prepare_X_y,
    SHAP_DIR, REPORTS_DIR,
)
from ml_pipeline.data_loading import load_invoice_features
from ml_pipeline.feature_engineering import engineer_invoice_features, INVOICE_FEATURE_COLS

log = get_logger("explainability")

DB_BATCH_SIZE = 2_000

# ── Human-readable feature labels ────────────────────────────────────────────

FEATURE_LABELS = {
    "billed_vs_expected_ratio":  "billing-to-contract ratio",
    "contract_gap":              "contract underbilling gap",
    "avg_days_late":             "average payment delay",
    "max_days_late":             "maximum payment delay",
    "outstanding_ratio":         "outstanding balance ratio",
    "failed_payment_rate":       "failed payment rate",
    "duplicate_refund_count":    "duplicate refund count",
    "refund_ratio":              "refund-to-payment ratio",
    "discount_ratio":            "discount ratio",
    "sub_discount_pct":          "subscription discount %",
    "overdue_days":              "days overdue",
    "paid_vs_billed_ratio":      "paid-to-billed ratio",
    "payment_delay_ratio":       "payment delay ratio",
    "retry_rate":                "payment retry rate",
    "total_amount":              "invoice total amount",
    "total_paid":                "total amount paid",
    "outstanding_amount":        "outstanding amount",
    "tax_amount":                "tax amount",
    "mrr":                       "monthly recurring revenue",
    "mrr_log":                   "MRR (log scale)",
    "failed_payment_count":      "failed payment count",
    "refund_count":              "refund count",
    "refund_amount":             "total refund amount",
    "payment_count":             "payment count",
    "total_attempts":            "total payment attempts",
    "quantity":                  "subscription quantity",
    "discount_amount":           "discount amount",
    "issue_month":               "invoice month",
    "issue_quarter":             "invoice quarter",
    "is_annual_billing":         "annual billing flag",
    "payment_terms_days":        "payment terms (days)",
    "contract_discount_pct":     "contract discount %",
}


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _explain_row(shap_vals: np.ndarray, feature_names: list[str], top_n: int = 3) -> str:
    """
    Produces a plain-English explanation for a single invoice.
    E.g. "Flagged due to high billing-to-contract ratio (+0.31),
          high maximum payment delay (+0.18), high discount ratio (+0.09)"
    """
    pairs = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    parts = []
    for feat, val in pairs:
        direction = "high" if val > 0 else "low"
        parts.append(f"{direction} {_label(feat)} ({val:+.3f})")

    return "Flagged due to: " + ", ".join(parts)


# ── Core SHAP computation ─────────────────────────────────────────────────────

def compute_shap_values(
    xgb_model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Computes SHAP values using TreeExplainer.
    Returns array of shape (n_samples, n_features).
    """
    log.info("Initialising TreeExplainer…")
    explainer = shap.TreeExplainer(xgb_model)

    log.info("Computing SHAP values for %d samples x %d features…", *X.shape)
    shap_values = explainer.shap_values(X)

    # XGBoost binary classification returns a single 2D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class

    log.info("SHAP values computed. Shape: %s", shap_values.shape)
    return shap_values


# ── Global summary ────────────────────────────────────────────────────────────

def global_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    save: bool = True,
) -> pd.DataFrame:
    """
    Mean absolute SHAP value per feature — global feature importance.
    Saved to artifacts/shap/global_shap_importance.csv + .json
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary["label"] = summary["feature"].map(lambda f: _label(f))
    summary["rank"] = range(1, len(summary) + 1)

    if save:
        csv_path = SHAP_DIR / "global_shap_importance.csv"
        summary.to_csv(csv_path, index=False)
        log.info("Global SHAP importance saved to %s", csv_path)

        json_path = SHAP_DIR / "global_shap_importance.json"
        summary.head(20).to_json(json_path, orient="records", indent=2)
        log.info("Top-20 SHAP importance saved to %s", json_path)

    log.info("\nTop 10 global SHAP features:")
    for _, row in summary.head(10).iterrows():
        log.info("  %2d. %-35s %.4f", row["rank"], row["label"], row["mean_abs_shap"])

    return summary


# ── Per-invoice local explanations ────────────────────────────────────────────

def build_local_explanations(
    shap_values: np.ndarray,
    feature_names: list[str],
    invoice_ids: pd.Series,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Builds a DataFrame with one row per invoice:
      invoice_id | top_features (list) | shap_dict (dict) | explanation (str)
    """
    log.info("Building local explanations for %d invoices…", len(invoice_ids))

    rows = []
    for i, inv_id in enumerate(invoice_ids):
        sv = shap_values[i]
        top_feats = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        rows.append({
            "invoice_id": int(inv_id),
            "top_features": [f for f, _ in top_feats],
            "shap_dict": {f: round(v, 6) for f, v in top_feats},
            "explanation": _explain_row(sv, feature_names, top_n=top_n),
        })

    df = pd.DataFrame(rows)

    # Save sample
    sample_path = SHAP_DIR / "local_explanations_sample.json"
    df.head(100).to_json(sample_path, orient="records", indent=2)
    log.info("Sample local explanations (100 rows) saved to %s", sample_path)

    return df


# ── Write SHAP values back to risk_scores table ───────────────────────────────

def update_risk_scores_with_shap(
    local_df: pd.DataFrame,
    model_version: str = "v1",
):
    """
    Updates risk_scores.shap_values and risk_scores.top_features
    for every invoice_id in local_df.
    Uses batched UPDATE queries.
    """
    from django.db import connection

    log.info("Updating risk_scores with SHAP values (%d rows)…", len(local_df))

    updated = 0
    for i in range(0, len(local_df), DB_BATCH_SIZE):
        batch = local_df.iloc[i : i + DB_BATCH_SIZE]
        with connection.cursor() as cur:
            for _, row in batch.iterrows():
                cur.execute(
                    """
                    UPDATE risk_scores
                       SET shap_values  = %s,
                           top_features = %s
                     WHERE invoice_id   = %s
                       AND model_version = %s
                       AND model_name   = 'xgboost'
                    """,
                    [
                        json.dumps(row["shap_dict"]),
                        json.dumps(row["top_features"]),
                        row["invoice_id"],
                        model_version,
                    ],
                )
        updated += len(batch)
        log.info("  Updated %d / %d risk_scores rows…", updated, len(local_df))

    log.info("SHAP update complete.")


# ── Master runner ─────────────────────────────────────────────────────────────

def run_explainability(model_version: str = "v1"):
    log.info("=" * 55)
    log.info("Phase 5 -- SHAP Explainability")
    log.info("=" * 55)

    # Load data
    log.info("Loading invoice features from MySQL…")
    df_raw = load_invoice_features()
    df_eng = engineer_invoice_features(df_raw)

    for col in ["invoice_id", "customer_id", "issue_date"]:
        if col in df_raw.columns and col not in df_eng.columns:
            df_eng[col] = df_raw[col].values

    # Build feature matrix
    X, _ = prepare_X_y(df_eng, INVOICE_FEATURE_COLS)
    feature_names = list(X.columns)
    invoice_ids = df_eng["invoice_id"]

    log.info("Feature matrix: %d x %d", *X.shape)

    # Load XGBoost model
    log.info("Loading XGBoost model from disk…")
    xgb_model = load_model("xgboost_v1")

    # Compute SHAP values
    shap_values = compute_shap_values(xgb_model, X)

    # Save raw SHAP matrix (compressed numpy)
    np_path = SHAP_DIR / "shap_values_full.npy"
    np.save(str(np_path), shap_values)
    log.info("Full SHAP matrix saved to %s", np_path)

    # Global summary
    summary = global_summary(shap_values, feature_names)

    # Local explanations
    local_df = build_local_explanations(shap_values, feature_names, invoice_ids)

    # Write to MySQL
    update_risk_scores_with_shap(local_df, model_version=model_version)

    log.info("=" * 55)
    log.info("Explainability complete.")
    log.info("  artifacts/shap/global_shap_importance.csv")
    log.info("  artifacts/shap/global_shap_importance.json")
    log.info("  artifacts/shap/local_explanations_sample.json")
    log.info("  artifacts/shap/shap_values_full.npy")
    log.info("  risk_scores table updated with shap_values + top_features")
    log.info("=" * 55)

    return summary, local_df


if __name__ == "__main__":
    run_explainability()
