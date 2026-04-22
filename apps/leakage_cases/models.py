from django.db import models
from django.contrib.auth import get_user_model
from apps.common.models import TimeStampedModel
from apps.customers.models import Customer
from apps.invoices.models import Invoice

User = get_user_model()


class RuleAlert(TimeStampedModel):
    """One alert fired by the rule-based reconciliation engine."""

    SEVERITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]

    RULE_CODE_CHOICES = [
        ("MISSING_PAYMENT", "Invoice issued but payment missing"),
        ("NO_INVOICE", "Active subscription — no invoice generated"),
        ("UNDERBILLING", "Billed amount < contracted amount"),
        ("DUPLICATE_REFUND", "Duplicate refund detected"),
        ("ABNORMAL_DISCOUNT", "Discount ratio above threshold"),
        ("MISSING_RENEWAL", "Contract renewal overdue"),
        ("PAYMENT_DELAY", "Payment received after due date"),
        ("CONTRACT_MISMATCH", "Billing does not match contract terms"),
        ("FAILED_PAYMENT_STREAK", "Multiple consecutive failed payments"),
        ("ZERO_REVENUE", "Active customer — zero revenue period"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="rule_alerts")
    invoice = models.ForeignKey(
        Invoice, on_delete=models.SET_NULL, null=True, blank=True, related_name="rule_alerts"
    )
    rule_code = models.CharField(max_length=50, choices=RULE_CODE_CHOICES, db_index=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="medium")
    detected_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField()
    leakage_amount = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    is_resolved = models.BooleanField(default=False, db_index=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "rule_alerts"
        indexes = [
            models.Index(fields=["customer", "rule_code"]),
            models.Index(fields=["severity", "is_resolved"]),
            models.Index(fields=["detected_at"]),
        ]

    def __str__(self):
        return f"{self.rule_code} | {self.customer} | {self.severity}"


class LeakageCase(TimeStampedModel):
    """Aggregated investigation case linking one or more RuleAlerts."""

    STATUS_CHOICES = [
        ("open", "Open"),
        ("investigating", "Investigating"),
        ("confirmed", "Confirmed"),
        ("resolved", "Resolved"),
        ("false_positive", "False Positive"),
        ("escalated", "Escalated"),
    ]

    PRIORITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="leakage_cases")
    case_number = models.CharField(max_length=64, unique=True, db_index=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default="medium")
    estimated_leakage_amount = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    confirmed_leakage_amount = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    assigned_to = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="assigned_cases"
    )
    rule_alerts = models.ManyToManyField(RuleAlert, blank=True, related_name="leakage_cases")
    resolution_notes = models.TextField(blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    due_date = models.DateField(null=True, blank=True)
    tags = models.JSONField(default=list, blank=True)

    class Meta:
        db_table = "leakage_cases"
        indexes = [
            models.Index(fields=["customer", "status"]),
            models.Index(fields=["priority", "status"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.case_number} — {self.title} [{self.status}]"
