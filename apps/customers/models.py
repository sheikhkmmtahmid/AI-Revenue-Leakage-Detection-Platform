from django.db import models
from apps.common.models import SoftDeleteModel


class Customer(SoftDeleteModel):
    SEGMENT_CHOICES = [
        ("enterprise", "Enterprise"),
        ("mid_market", "Mid-Market"),
        ("smb", "SMB"),
        ("startup", "Startup"),
    ]

    STATUS_CHOICES = [
        ("active", "Active"),
        ("churned", "Churned"),
        ("suspended", "Suspended"),
        ("prospect", "Prospect"),
    ]

    external_id = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=32, blank=True)
    industry = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, default="US")
    segment = models.CharField(max_length=20, choices=SEGMENT_CHOICES, default="smb")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    acquisition_date = models.DateField()
    churn_date = models.DateField(null=True, blank=True)
    account_manager = models.CharField(max_length=255, blank=True)
    credit_limit = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    risk_tier = models.CharField(max_length=20, default="medium")
    notes = models.TextField(blank=True)

    class Meta:
        db_table = "customers"
        indexes = [
            models.Index(fields=["status", "segment"]),
            models.Index(fields=["acquisition_date"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.external_id})"

    @property
    def is_churned(self):
        return self.status == "churned"
