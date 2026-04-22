from django.db import models
from apps.common.models import SoftDeleteModel
from apps.customers.models import Customer
from apps.contracts.models import Contract
from apps.subscriptions.models import Subscription


class Invoice(SoftDeleteModel):
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("issued", "Issued"),
        ("paid", "Paid"),
        ("partially_paid", "Partially Paid"),
        ("overdue", "Overdue"),
        ("void", "Void"),
        ("written_off", "Written Off"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="invoices")
    contract = models.ForeignKey(
        Contract, on_delete=models.SET_NULL, null=True, blank=True, related_name="invoices"
    )
    subscription = models.ForeignKey(
        Subscription, on_delete=models.SET_NULL, null=True, blank=True, related_name="invoices"
    )
    invoice_number = models.CharField(max_length=64, unique=True, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="issued")
    issue_date = models.DateField(db_index=True)
    due_date = models.DateField(db_index=True)
    period_start = models.DateField()
    period_end = models.DateField()
    subtotal = models.DecimalField(max_digits=14, decimal_places=2)
    tax_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_amount = models.DecimalField(max_digits=14, decimal_places=2)
    paid_amount = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    outstanding_amount = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    currency = models.CharField(max_length=3, default="USD")
    billing_address = models.TextField(blank=True)

    class Meta:
        db_table = "invoices"
        indexes = [
            models.Index(fields=["customer", "status"]),
            models.Index(fields=["issue_date", "due_date"]),
            models.Index(fields=["status", "due_date"]),
        ]

    def __str__(self):
        return f"{self.invoice_number} — {self.customer} — {self.total_amount}"

    def save(self, *args, **kwargs):
        self.outstanding_amount = self.total_amount - self.paid_amount
        super().save(*args, **kwargs)
