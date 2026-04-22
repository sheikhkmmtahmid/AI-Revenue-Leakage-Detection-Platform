"""
Model evaluation utilities.

Computes and reports:
  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Full sklearn classification report
  - Confusion matrix
  - Optimal threshold via Youden's J (max sensitivity + specificity)
  - Saves JSON report to artifacts/reports/
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from ml_pipeline.utils import get_logger, save_report, REPORTS_DIR

log = get_logger("evaluation")


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
    save: bool = True,
) -> dict:
    """
    Full evaluation suite for a binary leakage classifier.

    Args:
        y_true:     ground-truth labels (0/1)
        y_prob:     predicted probabilities for the positive class
        model_name: identifier for logging and file naming
        threshold:  decision threshold (default 0.5)
        save:       if True, write report JSON to artifacts/reports/

    Returns:
        dict with all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    opt_threshold, youden_j = _optimal_threshold(y_true, y_prob)

    report = {
        "model_name": model_name,
        "threshold_used": threshold,
        "optimal_threshold": opt_threshold,
        "youden_j": youden_j,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "leakage_rate_pct": float(y_true.mean() * 100),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    _print_report(report)

    if save:
        save_report(report, f"eval_{model_name}")

    return report


def evaluate_isolation_forest(
    y_true: pd.Series | np.ndarray,
    if_scores: np.ndarray,
    model_name: str = "isolation_forest",
    save: bool = True,
) -> dict:
    """
    Isolation Forest doesn't output calibrated probabilities,
    so we evaluate using the raw decision_function scores.
    Lower score = more anomalous → we negate to get anomaly probability proxy.
    """
    # Normalise to [0, 1] so ROC-AUC is computable
    s = -if_scores  # flip sign: higher now = more anomalous
    s_min, s_max = s.min(), s.max()
    prob_proxy = (s - s_min) / (s_max - s_min + 1e-9)

    return evaluate_classifier(
        y_true, prob_proxy, model_name=model_name, threshold=0.5, save=save
    )


# ── Threshold optimisation ────────────────────────────────────────────────────

def _optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Youden's J statistic: maximises sensitivity + specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_models(reports: list[dict]) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame from a list of eval reports.
    """
    rows = []
    for r in reports:
        rows.append({
            "Model": r["model_name"],
            "ROC-AUC": r["roc_auc"],
            "PR-AUC": r["pr_auc"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1_score"],
            "Opt Threshold": r.get("optimal_threshold", "-"),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_report(r: dict):
    sep = "─" * 55
    log.info(sep)
    log.info("Model          : %s", r["model_name"])
    log.info("Samples        : %d  (leakage rate: %.1f%%)", r["n_samples"], r["leakage_rate_pct"])
    log.info("ROC-AUC        : %.4f", r["roc_auc"])
    log.info("PR-AUC         : %.4f", r["pr_auc"])
    log.info("Accuracy       : %.4f", r["accuracy"])
    log.info("Precision      : %.4f", r["precision"])
    log.info("Recall         : %.4f", r["recall"])
    log.info("F1-Score       : %.4f", r["f1_score"])
    log.info("Optimal thresh : %.4f  (Youden J=%.4f)", r["optimal_threshold"], r["youden_j"])
    cm = r["confusion_matrix"]
    log.info("Confusion matrix:")
    log.info("  TN=%d  FP=%d", cm[0][0], cm[0][1])
    log.info("  FN=%d  TP=%d", cm[1][0], cm[1][1])
    log.info(sep)
