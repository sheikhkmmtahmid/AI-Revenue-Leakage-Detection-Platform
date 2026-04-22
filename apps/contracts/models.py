from django.db import models
from apps.common.models import SoftDeleteModel
from apps.customers.models import Customer


class Product(SoftDeleteModel):
    BILLING_CYCLE_CHOICES = [
        ("monthly", "Monthly"),
        ("quarterly", "Quarterly"),
        ("annual", "Annual"),
    ]

    code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    base_price = models.DecimalField(max_digits=12, decimal_places=2)
    billing_cycle = models.CharField(max_length=20, choices=BILLING_CYCLE_CHOICES, default="monthly")
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "products"

    def __str__(self):
        return f"{self.code} — {self.name}"


class Contract(SoftDeleteModel):
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("active", "Active"),
        ("expired", "Expired"),
        ("terminated", "Terminated"),
        ("renewed", "Renewed"),
    ]

    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="contracts")
    contract_number = models.CharField(max_length=64, unique=True, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    start_date = models.DateField()
    end_date = models.DateField()
    renewal_date = models.DateField(null=True, blank=True)
    contracted_value = models.DecimalField(max_digits=16, decimal_places=2)
    currency = models.CharField(max_length=3, default="USD")
    payment_terms_days = models.PositiveSmallIntegerField(default=30)
    auto_renew = models.BooleanField(default=True)
    discount_pct = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    signed_by = models.CharField(max_length=255, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        db_table = "contracts"
        indexes = [
            models.Index(fields=["customer", "status"]),
            models.Index(fields=["end_date"]),
            models.Index(fields=["renewal_date"]),
        ]

    def __str__(self):
        return f"{self.contract_number} ({self.customer})"
