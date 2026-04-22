from django.db import models
from apps.common.models import TimeStampedModel
from apps.customers.models import Customer
from apps.invoices.models import Invoice


class RiskScore(TimeStampedModel):
    """XGBoost / Logistic Regression leakage probability score."""

    MODEL_CHOICES = [
        ("logistic_regression", "Logistic Regression"),
        ("xgboost", "XGBoost"),
        ("ensemble", "Ensemble"),
    ]

    SEVERITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="risk_scores")
    invoice = models.ForeignKey(
        Invoice, on_delete=models.SET_NULL, null=True, blank=True, related_name="risk_scores"
    )
    model_name = models.CharField(max_length=30, choices=MODEL_CHOICES, default="xgboost")
    model_version = models.CharField(max_length=50, default="v1")
    leakage_probability = models.FloatField(help_text="0.0 – 1.0")
    risk_severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="low")
    rank_percentile = models.FloatField(null=True, blank=True)
    feature_snapshot = models.JSONField(default=dict, blank=True)
    shap_values = models.JSONField(default=dict, blank=True)
    top_features = models.JSONField(default=list, blank=True)
    scored_at = models.DateTimeField(auto_now_add=True, db_index=True)
    period_month = models.CharField(max_length=7, blank=True, help_text="YYYY-MM")

    class Meta:
        db_table = "risk_scores"
        indexes = [
            models.Index(fields=["customer", "risk_severity"]),
            models.Index(fields=["leakage_probability"]),
            models.Index(fields=["scored_at"]),
        ]

    def __str__(self):
        return (
            f"RiskScore {self.customer} | {self.model_name} | "
            f"prob={self.leakage_probability:.3f} | {self.risk_severity}"
        )

    def save(self, *args, **kwargs):
        p = self.leakage_probability
        if p >= 0.8:
            self.risk_severity = "critical"
        elif p >= 0.6:
            self.risk_severity = "high"
        elif p >= 0.4:
            self.risk_severity = "medium"
        else:
            self.risk_severity = "low"
        super().save(*args, **kwargs)
