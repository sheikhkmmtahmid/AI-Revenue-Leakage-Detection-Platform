from django.db import models
from apps.common.models import TimeStampedModel
from apps.customers.models import Customer
from apps.invoices.models import Invoice


class AnomalyScore(TimeStampedModel):
    """Isolation Forest anomaly score for an invoice or customer-period."""

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="anomaly_scores")
    invoice = models.ForeignKey(
        Invoice, on_delete=models.SET_NULL, null=True, blank=True, related_name="anomaly_scores"
    )
    model_version = models.CharField(max_length=50, default="v1")
    score = models.FloatField(help_text="Isolation Forest raw score (lower = more anomalous)")
    is_anomaly = models.BooleanField(default=False, db_index=True)
    threshold_used = models.FloatField(default=-0.1)
    feature_snapshot = models.JSONField(default=dict, blank=True)
    scored_at = models.DateTimeField(auto_now_add=True, db_index=True)
    period_month = models.CharField(max_length=7, blank=True, help_text="YYYY-MM")

    class Meta:
        db_table = "anomaly_scores"
        indexes = [
            models.Index(fields=["customer", "is_anomaly"]),
            models.Index(fields=["scored_at"]),
        ]

    def __str__(self):
        return f"AnomalyScore {self.customer} | {self.score:.4f} | anomaly={self.is_anomaly}"
