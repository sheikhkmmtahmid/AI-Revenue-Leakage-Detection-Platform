from rest_framework import serializers
from apps.risk_scoring.models import RiskScore


class RiskScoreSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source="customer.name", read_only=True)
    invoice_number = serializers.CharField(source="invoice.invoice_number", read_only=True)

    class Meta:
        model = RiskScore
        fields = [
            "id", "customer_id", "customer_name", "invoice_id", "invoice_number",
            "model_name", "model_version", "leakage_probability", "risk_severity",
            "rank_percentile", "top_features", "shap_values", "scored_at", "period_month",
        ]
