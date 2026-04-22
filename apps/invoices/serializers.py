from rest_framework import serializers
from apps.invoices.models import Invoice


class InvoiceSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source="customer.name", read_only=True)
    risk_probability = serializers.SerializerMethodField()
    shap_explanation = serializers.SerializerMethodField()

    class Meta:
        model = Invoice
        fields = [
            "id", "invoice_number", "customer_id", "customer_name",
            "status", "issue_date", "due_date", "period_start", "period_end",
            "subtotal", "tax_amount", "discount_amount", "total_amount",
            "paid_amount", "outstanding_amount", "currency",
            "risk_probability", "shap_explanation",
        ]

    def get_risk_probability(self, obj):
        score = obj.risk_scores.order_by("-scored_at").first()
        return round(float(score.leakage_probability), 4) if score else None

    def get_shap_explanation(self, obj):
        score = obj.risk_scores.order_by("-scored_at").first()
        if score and score.shap_values:
            top = score.top_features[:3] if score.top_features else []
            return "Flagged due to: " + ", ".join(
                f.replace("_", " ") for f in top
            ) if top else None
        return None
