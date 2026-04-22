from rest_framework import serializers
from apps.customers.models import Customer


class CustomerListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = [
            "id", "external_id", "name", "email", "segment", "status",
            "industry", "country", "acquisition_date", "churn_date",
            "risk_tier", "credit_limit", "account_manager",
        ]


class CustomerDetailSerializer(serializers.ModelSerializer):
    total_invoices = serializers.SerializerMethodField()
    total_invoiced = serializers.SerializerMethodField()
    total_paid = serializers.SerializerMethodField()
    leakage_risk = serializers.SerializerMethodField()

    class Meta:
        model = Customer
        fields = "__all__"

    def get_total_invoices(self, obj):
        return obj.invoices.filter(is_deleted=False).count()

    def get_total_invoiced(self, obj):
        from django.db.models import Sum
        result = obj.invoices.filter(is_deleted=False).aggregate(t=Sum("total_amount"))
        return float(result["t"] or 0)

    def get_total_paid(self, obj):
        from django.db.models import Sum
        result = obj.invoices.filter(is_deleted=False).aggregate(t=Sum("paid_amount"))
        return float(result["t"] or 0)

    def get_leakage_risk(self, obj):
        score = obj.risk_scores.order_by("-scored_at").first()
        if score:
            return {
                "probability": round(float(score.leakage_probability), 4),
                "severity": score.risk_severity,
                "top_features": score.top_features,
            }
        return None
