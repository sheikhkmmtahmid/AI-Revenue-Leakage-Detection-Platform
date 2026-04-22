from django.db import models
from apps.common.models import TimeStampedModel


class ForecastResult(TimeStampedModel):
    """Revenue / collection forecast from Prophet or SARIMA."""

    MODEL_CHOICES = [
        ("prophet", "Prophet"),
        ("sarima", "SARIMA"),
    ]

    METRIC_CHOICES = [
        ("revenue", "Revenue"),
        ("collections", "Collections"),
        ("renewals", "Renewals"),
        ("mrr", "MRR"),
    ]

    model_name = models.CharField(max_length=20, choices=MODEL_CHOICES, default="prophet")
    model_version = models.CharField(max_length=50, default="v1")
    metric = models.CharField(max_length=20, choices=METRIC_CHOICES, default="revenue")
    period_month = models.CharField(max_length=7, db_index=True, help_text="YYYY-MM")
    forecasted_value = models.DecimalField(max_digits=18, decimal_places=2)
    actual_value = models.DecimalField(max_digits=18, decimal_places=2, null=True, blank=True)
    lower_bound = models.DecimalField(max_digits=18, decimal_places=2, null=True, blank=True)
    upper_bound = models.DecimalField(max_digits=18, decimal_places=2, null=True, blank=True)
    deviation_pct = models.FloatField(null=True, blank=True)
    is_anomalous = models.BooleanField(default=False, db_index=True)
    anomaly_threshold_pct = models.FloatField(default=15.0)
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "forecast_results"
        unique_together = [("model_name", "metric", "period_month", "model_version")]
        indexes = [
            models.Index(fields=["metric", "period_month"]),
            models.Index(fields=["is_anomalous"]),
        ]

    def __str__(self):
        return f"Forecast {self.metric} {self.period_month} = {self.forecasted_value}"

    def compute_deviation(self):
        if self.actual_value and self.forecasted_value:
            self.deviation_pct = float(
                (self.actual_value - self.forecasted_value) / self.forecasted_value * 100
            )
            self.is_anomalous = abs(self.deviation_pct) > self.anomaly_threshold_pct
