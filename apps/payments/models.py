from django.db import models
from apps.common.models import SoftDeleteModel
from apps.customers.models import Customer
from apps.invoices.models import Invoice


class Payment(SoftDeleteModel):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("succeeded", "Succeeded"),
        ("failed", "Failed"),
        ("refunded", "Refunded"),
        ("partially_refunded", "Partially Refunded"),
        ("disputed", "Disputed"),
        ("cancelled", "Cancelled"),
    ]

    METHOD_CHOICES = [
        ("credit_card", "Credit Card"),
        ("bank_transfer", "Bank Transfer"),
        ("ach", "ACH"),
        ("wire", "Wire"),
        ("check", "Check"),
        ("crypto", "Crypto"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="payments")
    invoice = models.ForeignKey(
        Invoice, on_delete=models.SET_NULL, null=True, blank=True, related_name="payments"
    )
    transaction_id = models.CharField(max_length=128, unique=True, db_index=True)
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="pending")
    method = models.CharField(max_length=20, choices=METHOD_CHOICES, default="credit_card")
    amount = models.DecimalField(max_digits=14, decimal_places=2)
    currency = models.CharField(max_length=3, default="USD")
    payment_date = models.DateField(db_index=True)
    settled_at = models.DateTimeField(null=True, blank=True)
    failure_reason = models.CharField(max_length=255, blank=True)
    gateway_response = models.JSONField(default=dict, blank=True)
    attempt_count = models.PositiveSmallIntegerField(default=1)
    is_late = models.BooleanField(default=False)
    days_late = models.IntegerField(default=0)
    processor = models.CharField(max_length=100, blank=True)

    class Meta:
        db_table = "payments"
        indexes = [
            models.Index(fields=["customer", "status"]),
            models.Index(fields=["payment_date"]),
            models.Index(fields=["invoice", "status"]),
        ]

    def __str__(self):
        return f"{self.transaction_id} — {self.amount} {self.currency} ({self.status})"
