"""
Master synthetic data runner.
Run from project root:
    venv/Scripts/python.exe synthetic_data/generators/run_all.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from synthetic_data.generators.customer_generator import generate_products, generate_customers
from synthetic_data.generators.contract_subscription_generator import (
    generate_contracts_and_subscriptions,
)
from synthetic_data.generators.invoice_payment_generator import (
    generate_invoices,
    generate_payments,
    generate_refunds,
)


def run():
    t0 = time.time()
    print("=" * 60)
    print("AI Revenue Leakage Platform — Synthetic Data Generator")
    print("=" * 60)

    print("\n[1/6] Generating Products…")
    products = generate_products()

    print("\n[2/6] Generating Customers…")
    customers = generate_customers()

    print("\n[3/6] Generating Contracts + Subscriptions…")
    contracts, subscriptions = generate_contracts_and_subscriptions(customers, products)

    print("\n[4/6] Generating Invoices…")
    invoices = generate_invoices(subscriptions)

    print("\n[5/6] Generating Payments…")
    payments = generate_payments(invoices)

    print("\n[6/6] Generating Refunds…")
    refunds = generate_refunds(payments)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Data generation complete in {elapsed:.1f}s")
    print(f"  Products     : {len(products)}")
    print(f"  Customers    : {len(customers)}")
    print(f"  Contracts    : {len(contracts)}")
    print(f"  Subscriptions: {len(subscriptions)}")
    print(f"  Invoices     : {len(invoices)}")
    print(f"  Payments     : {len(payments)}")
    print(f"  Refunds      : {len(refunds)}")
    print("=" * 60)


if __name__ == "__main__":
    run()
