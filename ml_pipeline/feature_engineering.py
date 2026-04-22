"""
Feature engineering layer.
Transforms raw DataFrames from data_loading into model-ready feature matrices.

Invoice-level features  → used by anomaly + risk scoring models
Customer-level features → used for account-level risk scoring
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Invoice-level feature engineering ────────────────────────────────────────

def engineer_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Type coercions — cast every numeric-ish column (MySQL may return Decimal)
    for col in df.columns:
        if col not in ("invoice_number", "invoice_status", "billing_cycle"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            except (TypeError, ValueError):
                pass

    for col in ["issue_date", "due_date", "period_start", "period_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Core derived ratios ──────────────────────────────────────────────────
    eps = 1e-6  # prevent divide-by-zero

    df["billed_vs_expected_ratio"] = df["total_amount"] / (df["contracted_value"] / 12 + eps)
    df["paid_vs_billed_ratio"] = df["total_paid"] / (df["total_amount"] + eps)
    df["refund_ratio"] = df["refund_amount"] / (df["total_paid"] + eps)
    df["discount_ratio"] = df["discount_amount"] / (df["subtotal"] + eps)
    df["outstanding_ratio"] = df["outstanding_amount"] / (df["total_amount"] + eps)
    df["tax_ratio"] = df["tax_amount"] / (df["total_amount"] + eps)
    df["contract_gap"] = df["contracted_value"] / 12 - df["total_amount"]

    # ── Payment behaviour ────────────────────────────────────────────────────
    df["overdue_days"] = np.where(
        df["invoice_status"] == "overdue",
        (pd.Timestamp.today() - df["due_date"]).dt.days.clip(lower=0),
        0,
    )
    df["payment_delay_ratio"] = df["max_days_late"] / (df["payment_terms_days"].fillna(30) + eps)
    df["failed_payment_rate"] = df["failed_payment_count"] / (df["payment_count"] + eps)
    df["retry_rate"] = df["total_attempts"] / (df["payment_count"] + eps)

    # ── Leakage flags (rule-based labels for supervised learning) ────────────
    df["flag_missing_payment"] = (
        (df["invoice_status"].isin(["overdue", "issued"])) &
        (df["total_paid"] == 0) &
        (df["due_date"] < pd.Timestamp.today())
    ).astype(int)

    df["flag_underbilling"] = (df["billed_vs_expected_ratio"] < 0.85).astype(int)
    df["flag_duplicate_refund"] = (df["duplicate_refund_count"] > 0).astype(int)
    df["flag_abnormal_discount"] = (df["discount_ratio"] > 0.30).astype(int)
    df["flag_payment_delay"] = (df["max_days_late"] > 30).astype(int)

    # Combined leakage label — any flag triggers a leakage suspicion
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["leakage_label"] = (df[flag_cols].sum(axis=1) > 0).astype(int)

    # ── Temporal features ────────────────────────────────────────────────────
    if "issue_date" in df.columns:
        df["issue_month"] = df["issue_date"].dt.month
        df["issue_quarter"] = df["issue_date"].dt.quarter
        df["issue_year"] = df["issue_date"].dt.year
        df["period_length_days"] = (df["period_end"] - df["period_start"]).dt.days

    # ── Billing cycle encoding ────────────────────────────────────────────────
    df["is_annual_billing"] = (df["billing_cycle"] == "annual").astype(int)

    # ── MRR normalised ────────────────────────────────────────────────────────
    df["mrr_log"] = np.log1p(df["mrr"].clip(lower=0))

    # ── Drop raw date/string cols not needed by the model ────────────────────
    drop_cols = [
        "invoice_number", "invoice_status", "issue_date", "due_date",
        "period_start", "period_end", "billing_cycle",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


# ── Customer-level feature engineering ───────────────────────────────────────

def engineer_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col not in ("segment", "customer_status", "acquisition_date", "churn_date", "risk_tier"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            except (TypeError, ValueError):
                pass

    eps = 1e-6

    df["collection_rate"] = df["total_paid"] / (df["total_invoiced"] + eps)
    df["overdue_rate"] = df["overdue_invoice_count"] / (df["invoice_count"] + eps)
    df["failed_payment_rate"] = df["failed_payment_count"] / (df["payment_count"] + eps)
    df["refund_rate"] = df["refund_count"] / (df["invoice_count"] + eps)
    df["duplicate_refund_rate"] = df["duplicate_refund_count"] / (df["refund_count"] + eps)
    df["outstanding_ratio"] = df["total_outstanding"] / (df["total_invoiced"] + eps)
    df["mrr_log"] = np.log1p(df["avg_mrr"].clip(lower=0))
    df["tenure_years"] = df["tenure_days"] / 365.25

    # Segment encoding
    segment_map = {"enterprise": 3, "mid_market": 2, "smb": 1, "startup": 0}
    df["segment_code"] = df["segment"].map(segment_map).fillna(1).astype(int)

    # Risk tier encoding
    risk_map = {"high": 2, "medium": 1, "low": 0}
    df["risk_tier_code"] = df["risk_tier"].map(risk_map).fillna(1).astype(int)

    # Churn flag
    df["churn_flag"] = (df["customer_status"] == "churned").astype(int)

    # Leakage label: customer at risk if any of these thresholds exceeded
    df["leakage_label"] = (
        (df["failed_payment_rate"] > 0.15) |
        (df["overdue_rate"] > 0.10) |
        (df["duplicate_refund_count"] > 0) |
        (df["avg_discount_pct"] > 30) |
        (df["outstanding_ratio"] > 0.20)
    ).astype(int)

    drop_cols = ["segment", "customer_status", "acquisition_date", "churn_date", "risk_tier"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


# ── Rolling time-series features for forecasting ─────────────────────────────

def engineer_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df from data_loading.load_monthly_revenue().
    Adds rolling window features and percentage changes.
    """
    df = df.copy().sort_values("period_month").reset_index(drop=True)

    for col in ["total_invoiced", "total_collected", "total_outstanding", "overdue_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["collection_rate"] = df["total_collected"] / (df["total_invoiced"] + 1e-6)

    for window in [3, 6, 12]:
        df[f"invoiced_ma{window}"] = df["total_invoiced"].rolling(window, min_periods=1).mean()
        df[f"collected_ma{window}"] = df["total_collected"].rolling(window, min_periods=1).mean()

    df["invoiced_mom_pct"] = df["total_invoiced"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["collected_mom_pct"] = df["total_collected"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    df["invoiced_yoy_pct"] = df["total_invoiced"].pct_change(12).replace([np.inf, -np.inf], 0).fillna(0)

    df["period_dt"] = pd.to_datetime(df["period_month"] + "-01")
    df["month_num"] = df["period_dt"].dt.month
    df["quarter"] = df["period_dt"].dt.quarter

    return df


# ── Feature selection lists used by ML models ─────────────────────────────────

INVOICE_FEATURE_COLS = [
    "total_amount", "paid_amount", "outstanding_amount", "discount_amount", "tax_amount",
    "total_paid", "payment_count", "failed_payment_count", "max_days_late", "avg_days_late",
    "total_attempts", "refund_count", "refund_amount", "duplicate_refund_count",
    "mrr", "sub_discount_pct", "quantity", "payment_terms_days",
    "billed_vs_expected_ratio", "paid_vs_billed_ratio", "refund_ratio", "discount_ratio",
    "outstanding_ratio", "contract_gap", "overdue_days", "payment_delay_ratio",
    "failed_payment_rate", "retry_rate", "is_annual_billing", "mrr_log",
    "issue_month", "issue_quarter",
]

CUSTOMER_FEATURE_COLS = [
    "tenure_days", "tenure_years", "active_subscriptions", "total_subscriptions",
    "avg_mrr", "total_arr", "avg_discount_pct", "invoice_count", "total_invoiced",
    "total_outstanding", "overdue_invoice_count", "payment_count", "failed_payment_count",
    "total_paid", "avg_days_late", "refund_count", "total_refunded", "duplicate_refund_count",
    "collection_rate", "overdue_rate", "failed_payment_rate", "refund_rate",
    "duplicate_refund_rate", "outstanding_ratio", "mrr_log",
    "segment_code", "risk_tier_code", "churn_flag",
]


if __name__ == "__main__":
    import os
    import sys
    import django

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()

    from ml_pipeline.data_loading import load_invoice_features, load_customer_features, load_monthly_revenue

    print("Engineering invoice features…")
    raw_inv = load_invoice_features()
    feat_inv = engineer_invoice_features(raw_inv)
    print(f"  Shape: {feat_inv.shape}")
    print(f"  Leakage rate: {feat_inv['leakage_label'].mean():.2%}")
    print(feat_inv[INVOICE_FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].head(10))

    print("\nEngineering customer features…")
    raw_cust = load_customer_features()
    feat_cust = engineer_customer_features(raw_cust)
    print(f"  Shape: {feat_cust.shape}")
    print(f"  Leakage rate: {feat_cust['leakage_label'].mean():.2%}")

    print("\nEngineering time-series features…")
    raw_ts = load_monthly_revenue()
    feat_ts = engineer_time_series_features(raw_ts)
    print(f"  Shape: {feat_ts.shape}")
    print(feat_ts[["period_month", "total_invoiced", "invoiced_ma3", "invoiced_mom_pct"]].tail(12))
