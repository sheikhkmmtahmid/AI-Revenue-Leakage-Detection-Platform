from django.db import models
from apps.common.models import SoftDeleteModel
from apps.customers.models import Customer
from apps.contracts.models import Contract, Product


class Subscription(SoftDeleteModel):
    STATUS_CHOICES = [
        ("active", "Active"),
        ("cancelled", "Cancelled"),
        ("suspended", "Suspended"),
        ("trial", "Trial"),
        ("expired", "Expired"),
        ("pending", "Pending"),
    ]

    BILLING_CYCLE_CHOICES = [
        ("monthly", "Monthly"),
        ("quarterly", "Quarterly"),
        ("annual", "Annual"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="subscriptions")
    contract = models.ForeignKey(
        Contract, on_delete=models.SET_NULL, null=True, blank=True, related_name="subscriptions"
    )
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name="subscriptions")
    external_id = models.CharField(max_length=64, unique=True, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    billing_cycle = models.CharField(max_length=20, choices=BILLING_CYCLE_CHOICES, default="monthly")
    mrr = models.DecimalField(max_digits=12, decimal_places=2, help_text="Monthly Recurring Revenue")
    arr = models.DecimalField(max_digits=14, decimal_places=2, help_text="Annual Recurring Revenue")
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    discount_pct = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    trial_end_date = models.DateField(null=True, blank=True)
    next_billing_date = models.DateField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    cancellation_reason = models.CharField(max_length=255, blank=True)
    upgrade_from = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="upgrades"
    )

    class Meta:
        db_table = "subscriptions"
        indexes = [
            models.Index(fields=["customer", "status"]),
            models.Index(fields=["next_billing_date"]),
            models.Index(fields=["product", "status"]),
        ]

    def __str__(self):
        return f"Sub-{self.external_id} ({self.customer} / {self.product})"
