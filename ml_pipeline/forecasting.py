"""
Phase 6 — Revenue Forecasting with Prophet.

Trains a Prophet model on 35 months of monthly revenue data, then:
  - Forecasts the next 6 months
  - Back-tests against known actuals (last 6 months of history)
  - Flags months where actual deviates >15% from forecast (anomaly)
  - Writes all results to the forecast_results table in MySQL
  - Saves forecast artifacts to artifacts/forecasts/

Run:
    venv/Scripts/python.exe ml_pipeline/forecasting.py
"""

import os
import sys
import json
from decimal import Decimal

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from ml_pipeline.utils import get_logger, FORECASTS_DIR, REPORTS_DIR
from ml_pipeline.data_loading import load_monthly_revenue
from ml_pipeline.feature_engineering import engineer_time_series_features

log = get_logger("forecasting")

ANOMALY_THRESHOLD_PCT = 15.0   # flag months deviating >15% from forecast
FORECAST_HORIZON_MONTHS = 6    # predict 6 months ahead
BACKTEST_MONTHS = 6            # last N months used for back-test evaluation
MODEL_VERSION = "v1"

# ── Prophet model builder ─────────────────────────────────────────────────────

def build_prophet(seasonality_mode: str = "multiplicative") -> Prophet:
    """
    Revenue time-series tends to have multiplicative seasonality
    (seasonal swings scale with trend level).
    """
    return Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,     # conservative — avoids overfitting 35pts
        seasonality_prior_scale=10.0,
        interval_width=0.90,              # 90% confidence interval
    )


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_prophet_df(df_monthly: pd.DataFrame, metric: str = "total_invoiced") -> pd.DataFrame:
    """
    Prophet requires columns: ds (datetime), y (float).
    """
    df = df_monthly[["period_month", metric]].copy()
    df = df.rename(columns={"period_month": "ds", metric: "y"})
    df["ds"] = pd.to_datetime(df["ds"] + "-01")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df = df.sort_values("ds").reset_index(drop=True)
    return df


# ── Training + forecasting ────────────────────────────────────────────────────

def train_and_forecast(
    prophet_df: pd.DataFrame,
    metric: str,
    backtest_months: int = BACKTEST_MONTHS,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
) -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """
    1. Trains Prophet on all data except the last `backtest_months`.
    2. Forecasts backward (back-test) + forward (future horizon).
    3. Returns (model, forecast_df, backtest_df).
    """
    cutoff_idx = len(prophet_df) - backtest_months
    df_train = prophet_df.iloc[:cutoff_idx].copy()
    df_actual = prophet_df.copy()

    log.info(
        "Training Prophet [%s] on %d months (%s to %s)",
        metric,
        len(df_train),
        df_train["ds"].min().strftime("%Y-%m"),
        df_train["ds"].max().strftime("%Y-%m"),
    )

    model = build_prophet()
    model.fit(df_train)

    # Future dataframe covers history + forward horizon
    total_months = len(prophet_df) + horizon_months
    future = model.make_future_dataframe(periods=total_months - len(df_train), freq="MS")
    forecast = model.predict(future)

    log.info(
        "Forecast generated: %d rows (%s to %s)",
        len(forecast),
        forecast["ds"].min().strftime("%Y-%m"),
        forecast["ds"].max().strftime("%Y-%m"),
    )

    # Back-test: join actuals onto forecast for the held-out months
    backtest = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    backtest = backtest.merge(df_actual.rename(columns={"y": "actual"}), on="ds", how="left")
    backtest["period_month"] = backtest["ds"].dt.strftime("%Y-%m")
    backtest["metric"] = metric

    # Deviation %
    backtest["deviation_pct"] = np.where(
        backtest["actual"].notna() & (backtest["yhat"] != 0),
        (backtest["actual"] - backtest["yhat"]) / backtest["yhat"].abs() * 100,
        np.nan,
    )
    backtest["is_anomalous"] = (
        backtest["deviation_pct"].abs() > ANOMALY_THRESHOLD_PCT
    ).fillna(False)

    return model, forecast, backtest


# ── Cross-validation diagnostics ─────────────────────────────────────────────

def run_cross_validation(model: Prophet, metric: str) -> pd.DataFrame:
    """
    Prophet's built-in time-series cross-validation.
    With 35 months, use initial=18, period=3, horizon=6.
    """
    log.info("Running Prophet cross-validation for [%s]…", metric)
    try:
        cv_df = cross_validation(
            model,
            initial="548 days",   # ~18 months
            period="91 days",     # ~3 months
            horizon="182 days",   # ~6 months
            parallel=None,
        )
        perf = performance_metrics(cv_df)
        log.info(
            "CV metrics [%s] -- RMSE: %.2f  MAE: %.2f  MAPE: %.4f",
            metric,
            perf["rmse"].mean(),
            perf["mae"].mean(),
            perf["mape"].mean(),
        )
        return perf
    except Exception as e:
        log.warning("Cross-validation failed for [%s]: %s", metric, e)
        return pd.DataFrame()


# ── DB writer ─────────────────────────────────────────────────────────────────

def write_forecast_results(backtest: pd.DataFrame, model_version: str = MODEL_VERSION):
    """
    Upserts rows into forecast_results table.
    Skips if a row already exists for (model_name, metric, period_month, model_version).
    """
    from apps.forecasting.models import ForecastResult

    metric = backtest["metric"].iloc[0]
    rows = []

    for _, row in backtest.iterrows():
        forecasted = row["yhat"]
        actual = row["actual"] if pd.notna(row.get("actual")) else None
        lower = row["yhat_lower"]
        upper = row["yhat_upper"]
        dev_pct = float(row["deviation_pct"]) if pd.notna(row.get("deviation_pct")) else None
        is_anom = bool(row["is_anomalous"])

        obj = ForecastResult(
            model_name="prophet",
            model_version=model_version,
            metric=metric if metric in ("revenue", "collections", "renewals", "mrr") else "revenue",
            period_month=row["period_month"],
            forecasted_value=Decimal(str(round(forecasted, 2))),
            actual_value=Decimal(str(round(float(actual), 2))) if actual is not None else None,
            lower_bound=Decimal(str(round(lower, 2))),
            upper_bound=Decimal(str(round(upper, 2))),
            deviation_pct=dev_pct,
            is_anomalous=is_anom,
            anomaly_threshold_pct=ANOMALY_THRESHOLD_PCT,
        )
        rows.append(obj)

    ForecastResult.objects.bulk_create(rows, ignore_conflicts=True)
    log.info("Wrote %d forecast rows to DB for metric=[%s]", len(rows), metric)


# ── Artifact savers ───────────────────────────────────────────────────────────

def save_forecast_artifacts(backtest: pd.DataFrame, metric: str, cv_perf: pd.DataFrame):
    # Full forecast table
    csv_path = FORECASTS_DIR / f"forecast_{metric}.csv"
    backtest.to_csv(csv_path, index=False)
    log.info("Forecast CSV saved to %s", csv_path)

    # Anomalous months
    anomalies = backtest[backtest["is_anomalous"] == True]
    if not anomalies.empty:
        anom_path = FORECASTS_DIR / f"anomalies_{metric}.csv"
        anomalies.to_csv(anom_path, index=False)
        log.info("Anomalous months saved to %s", anom_path)

    # Summary JSON
    summary = {
        "metric": metric,
        "model": "prophet",
        "model_version": MODEL_VERSION,
        "training_months": int(backtest["actual"].notna().sum()),
        "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
        "anomaly_threshold_pct": ANOMALY_THRESHOLD_PCT,
        "anomalous_months": anomalies["period_month"].tolist(),
        "backtest_mae": float(
            (backtest["actual"] - backtest["yhat"]).abs().dropna().mean()
        ),
        "backtest_mape": float(
            ((backtest["actual"] - backtest["yhat"]).abs() / backtest["actual"].abs())
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .mean()
        ),
        "cv_rmse_mean": float(cv_perf["rmse"].mean()) if not cv_perf.empty else None,
        "cv_mape_mean": float(cv_perf["mape"].mean()) if not cv_perf.empty else None,
    }
    json_path = FORECASTS_DIR / f"summary_{metric}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Summary saved to %s", json_path)
    return summary


# ── Master runner ─────────────────────────────────────────────────────────────

def run_forecasting():
    log.info("=" * 55)
    log.info("Phase 6 -- Revenue Forecasting (Prophet)")
    log.info("=" * 55)

    # Load monthly revenue
    log.info("Loading monthly revenue from MySQL…")
    df_raw = load_monthly_revenue()
    df_ts = engineer_time_series_features(df_raw)
    log.info("Time-series data: %d months (%s to %s)",
             len(df_ts),
             df_ts["period_month"].iloc[0],
             df_ts["period_month"].iloc[-1])

    # Run for two metrics: total invoiced (revenue) + total collected (collections)
    metrics = [
        ("total_invoiced",  "revenue"),
        ("total_collected", "collections"),
    ]

    all_summaries = []

    for ts_col, db_metric in metrics:
        log.info("-" * 55)
        log.info("Forecasting metric: %s", db_metric)

        # Prepare
        prophet_df = prepare_prophet_df(df_ts, metric=ts_col)
        prophet_df_with_metric = prophet_df.copy()

        # Train + forecast
        model, forecast, backtest = train_and_forecast(
            prophet_df_with_metric,
            metric=db_metric,
        )

        # Cross-validation
        cv_perf = run_cross_validation(model, metric=db_metric)

        # Save artifacts
        summary = save_forecast_artifacts(backtest, metric=db_metric, cv_perf=cv_perf)

        # Write to DB
        write_forecast_results(backtest, model_version=MODEL_VERSION)

        all_summaries.append(summary)

        # Log key results
        known = backtest[backtest["actual"].notna()]
        future = backtest[backtest["actual"].isna()]
        log.info("Back-test period: %d months | MAPE: %.2f%%",
                 len(known), summary["backtest_mape"] * 100)
        log.info("Anomalous months: %s", summary["anomalous_months"] or "None")
        log.info("Future forecast (%d months):", len(future))
        for _, r in future.iterrows():
            log.info("  %s  yhat=%.0f  [%.0f, %.0f]",
                     r["period_month"], r["yhat"], r["yhat_lower"], r["yhat_upper"])

    # Save combined summary
    combined_path = FORECASTS_DIR / "forecast_summary_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    log.info("=" * 55)
    log.info("Forecasting complete.")
    log.info("  artifacts/forecasts/forecast_revenue.csv")
    log.info("  artifacts/forecasts/forecast_collections.csv")
    log.info("  artifacts/forecasts/summary_revenue.json")
    log.info("  artifacts/forecasts/summary_collections.json")
    log.info("  forecast_results table updated in MySQL")
    log.info("=" * 55)


if __name__ == "__main__":
    run_forecasting()
