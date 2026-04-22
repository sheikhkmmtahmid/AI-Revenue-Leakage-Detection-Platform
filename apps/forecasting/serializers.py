from rest_framework import serializers
from apps.forecasting.models import ForecastResult


class ForecastResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastResult
        fields = [
            "id", "model_name", "model_version", "metric", "period_month",
            "forecasted_value", "actual_value", "lower_bound", "upper_bound",
            "deviation_pct", "is_anomalous", "anomaly_threshold_pct", "generated_at",
        ]
