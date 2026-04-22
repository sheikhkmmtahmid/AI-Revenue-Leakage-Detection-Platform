"""Generates Customer and Product rows and writes them to MySQL via Django ORM."""

import random
import sys
import os
import django

# Bootstrap Django when running standalone
if "django" not in sys.modules or not django.conf.settings.configured:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()

from faker import Faker
from apps.customers.models import Customer
from apps.contracts.models import Product
from synthetic_data.configs.data_config import (
    NUM_CUSTOMERS, SEGMENTS, PRODUCTS, START_DATE, HISTORY_MONTHS
)
from synthetic_data.generators.base_generator import (
    uid, random_date, random_decimal, weighted_choice, INDUSTRIES, COUNTRIES, ACCOUNT_MANAGERS
)
from datetime import date, timedelta

fake = Faker()
Faker.seed(42)
random.seed(42)


def generate_products() -> list[Product]:
    """Idempotent — skips if products already exist."""
    created = []
    for p in PRODUCTS:
        obj, new = Product.objects.get_or_create(
            code=p["code"],
            defaults={
                "name": p["name"],
                "base_price": p["price"],
                "billing_cycle": p["cycle"],
                "is_active": True,
            },
        )
        created.append(obj)
    print(f"  Products: {len(created)} ready")
    return created


def generate_customers(n: int = NUM_CUSTOMERS) -> list[Customer]:
    existing = Customer.objects.count()
    if existing >= n:
        print(f"  Customers: {existing} already exist, skipping")
        return list(Customer.objects.all()[:n])

    segment_weights = SEGMENTS
    status_pool = ["active"] * 75 + ["churned"] * 15 + ["suspended"] * 7 + ["prospect"] * 3
    end_date = START_DATE + timedelta(days=HISTORY_MONTHS * 30)

    batch = []
    emails_used = set(Customer.objects.values_list("email", flat=True))
    external_ids_used = set(Customer.objects.values_list("external_id", flat=True))

    for i in range(n - existing):
        segment = weighted_choice(segment_weights)
        status = random.choice(status_pool)
        acq_date = random_date(START_DATE, end_date - timedelta(days=90))

        churn_date = None
        if status == "churned":
            churn_date = random_date(acq_date + timedelta(days=30), end_date)

        credit_map = {
            "enterprise": (50_000, 500_000),
            "mid_market": (10_000, 100_000),
            "smb": (1_000, 20_000),
            "startup": (500, 10_000),
        }

        # Unique email + external_id
        email = None
        while email is None or email in emails_used:
            email = fake.unique.email()
        emails_used.add(email)

        ext_id = None
        while ext_id is None or ext_id in external_ids_used:
            ext_id = uid("CUST")
        external_ids_used.add(ext_id)

        lo, hi = credit_map[segment]
        batch.append(Customer(
            external_id=ext_id,
            name=fake.company(),
            email=email,
            phone=fake.phone_number()[:32],
            industry=random.choice(INDUSTRIES),
            country=random.choice(COUNTRIES),
            segment=segment,
            status=status,
            acquisition_date=acq_date,
            churn_date=churn_date,
            account_manager=random.choice(ACCOUNT_MANAGERS),
            credit_limit=random_decimal(lo, hi),
            risk_tier=random.choice(["low", "medium", "high"]),
        ))

        if len(batch) >= 500:
            Customer.objects.bulk_create(batch, ignore_conflicts=True)
            print(f"    Inserted {Customer.objects.count()} customers so far…")
            batch = []

    if batch:
        Customer.objects.bulk_create(batch, ignore_conflicts=True)

    total = Customer.objects.count()
    print(f"  Customers: {total} total")
    return list(Customer.objects.all())
