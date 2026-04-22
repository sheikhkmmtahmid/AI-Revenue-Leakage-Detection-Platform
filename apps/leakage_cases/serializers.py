from rest_framework import serializers
from apps.leakage_cases.models import LeakageCase, RuleAlert


class RuleAlertSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source="customer.name", read_only=True)

    class Meta:
        model = RuleAlert
        fields = [
            "id", "customer_id", "customer_name", "rule_code", "severity",
            "detected_at", "description", "leakage_amount", "is_resolved",
            "resolved_at", "metadata",
        ]


class LeakageCaseSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source="customer.name", read_only=True)
    assigned_to_name = serializers.SerializerMethodField()
    alert_count = serializers.SerializerMethodField()

    class Meta:
        model = LeakageCase
        fields = [
            "id", "case_number", "customer_id", "customer_name", "title",
            "description", "status", "priority", "estimated_leakage_amount",
            "confirmed_leakage_amount", "assigned_to_id", "assigned_to_name",
            "resolution_notes", "resolved_at", "due_date", "tags",
            "alert_count", "created_at", "updated_at",
        ]

    def get_assigned_to_name(self, obj):
        return obj.assigned_to.get_full_name() if obj.assigned_to else None

    def get_alert_count(self, obj):
        return obj.rule_alerts.count()
