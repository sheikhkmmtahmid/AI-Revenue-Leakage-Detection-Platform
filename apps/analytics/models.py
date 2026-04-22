from django.db import models
from django.contrib.auth import get_user_model
from apps.common.models import TimeStampedModel

User = get_user_model()


class AuditLog(TimeStampedModel):
    """Immutable event log for all platform actions."""

    ACTION_CHOICES = [
        ("CREATE", "Create"),
        ("UPDATE", "Update"),
        ("DELETE", "Delete"),
        ("SCORE", "Score"),
        ("ALERT", "Alert"),
        ("RESOLVE", "Resolve"),
        ("INGEST", "Ingest"),
        ("TRAIN", "Train"),
        ("FORECAST", "Forecast"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="audit_logs"
    )
    action = models.CharField(max_length=20, choices=ACTION_CHOICES, db_index=True)
    entity_type = models.CharField(max_length=100, db_index=True)
    entity_id = models.CharField(max_length=128, db_index=True)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    occurred_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "audit_logs"
        indexes = [
            models.Index(fields=["entity_type", "entity_id"]),
            models.Index(fields=["occurred_at"]),
            models.Index(fields=["user", "action"]),
        ]

    def __str__(self):
        return f"{self.action} {self.entity_type}#{self.entity_id} at {self.occurred_at}"


class MonthlyRevenueSummary(TimeStampedModel):
    """Aggregated monthly revenue snapshot for trend analysis."""

    period_month = models.CharField(max_length=7, unique=True, db_index=True, help_text="YYYY-MM")
    total_invoiced = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    total_collected = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    total_refunded = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    total_discounted = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    net_revenue = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    active_subscriptions = models.IntegerField(default=0)
    new_customers = models.IntegerField(default=0)
    churned_customers = models.IntegerField(default=0)
    failed_payment_count = models.IntegerField(default=0)
    leakage_estimated = models.DecimalField(max_digits=18, decimal_places=2, default=0)
    collection_rate = models.FloatField(default=0.0)

    class Meta:
        db_table = "monthly_revenue_summaries"

    def __str__(self):
        return f"Revenue {self.period_month}: net={self.net_revenue}"


class Discount(TimeStampedModel):
    """Discount events applied to invoices or subscriptions."""

    TYPE_CHOICES = [
        ("percentage", "Percentage"),
        ("fixed", "Fixed Amount"),
        ("promotional", "Promotional"),
        ("loyalty", "Loyalty"),
        ("negotiated", "Negotiated"),
    ]

    from apps.invoices.models import Invoice
    from apps.subscriptions.models import Subscription

    invoice = models.ForeignKey(
        "invoices.Invoice", on_delete=models.CASCADE, null=True, blank=True, related_name="discounts"
    )
    subscription = models.ForeignKey(
        "subscriptions.Subscription", on_delete=models.CASCADE, null=True, blank=True, related_name="discounts"
    )
    discount_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default="percentage")
    discount_pct = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    approved_by = models.CharField(max_length=255, blank=True)
    valid_from = models.DateField()
    valid_to = models.DateField(null=True, blank=True)
    is_approved = models.BooleanField(default=True)
    reason = models.TextField(blank=True)

    class Meta:
        db_table = "discounts"

    def __str__(self):
        return f"Discount {self.discount_type} {self.discount_pct}%/{self.discount_amount}"
