from rest_framework import serializers
from apps.anomaly_detection.models import AnomalyScore


class AnomalyScoreSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source="customer.name", read_only=True)
    invoice_number = serializers.CharField(source="invoice.invoice_number", read_only=True)

    class Meta:
        model = AnomalyScore
        fields = [
            "id", "customer_id", "customer_name", "invoice_id", "invoice_number",
            "model_version", "score", "is_anomaly", "threshold_used",
            "scored_at", "period_month",
        ]
