from django.db import models
from apps.common.models import SoftDeleteModel
from apps.customers.models import Customer
from apps.invoices.models import Invoice
from apps.payments.models import Payment


class Refund(SoftDeleteModel):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("processed", "Processed"),
        ("rejected", "Rejected"),
        ("reversed", "Reversed"),
    ]

    REASON_CHOICES = [
        ("duplicate_charge", "Duplicate Charge"),
        ("service_cancellation", "Service Cancellation"),
        ("billing_error", "Billing Error"),
        ("contract_termination", "Contract Termination"),
        ("customer_request", "Customer Request"),
        ("fraud", "Fraud"),
        ("other", "Other"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="refunds")
    payment = models.ForeignKey(Payment, on_delete=models.PROTECT, related_name="refunds")
    invoice = models.ForeignKey(
        Invoice, on_delete=models.SET_NULL, null=True, blank=True, related_name="refunds"
    )
    refund_number = models.CharField(max_length=64, unique=True, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    reason = models.CharField(max_length=30, choices=REASON_CHOICES, default="customer_request")
    amount = models.DecimalField(max_digits=14, decimal_places=2)
    currency = models.CharField(max_length=3, default="USD")
    requested_at = models.DateTimeField()
    processed_at = models.DateTimeField(null=True, blank=True)
    is_duplicate = models.BooleanField(default=False, db_index=True)
    notes = models.TextField(blank=True)

    class Meta:
        db_table = "refunds"
        indexes = [
            models.Index(fields=["customer", "status"]),
            models.Index(fields=["payment", "is_duplicate"]),
        ]

    def __str__(self):
        return f"{self.refund_number} — {self.amount} {self.currency} ({self.reason})"
