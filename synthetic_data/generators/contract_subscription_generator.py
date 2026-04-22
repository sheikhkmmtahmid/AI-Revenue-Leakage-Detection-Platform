"""Generates Contract and Subscription rows."""

import random
from datetime import date, timedelta
from decimal import Decimal

from apps.customers.models import Customer
from apps.contracts.models import Contract, Product
from apps.subscriptions.models import Subscription
from synthetic_data.configs.data_config import (
    NUM_SUBSCRIPTIONS, START_DATE, HISTORY_MONTHS, LEAKAGE_RATES
)
from synthetic_data.generators.base_generator import uid, random_date, random_decimal

random.seed(42)

END_DATE = START_DATE + timedelta(days=HISTORY_MONTHS * 30)


def _contract_value(product: Product, months: int) -> Decimal:
    if product.billing_cycle == "annual":
        return product.base_price * (months // 12 or 1)
    return product.base_price * months


def _build_pair(customer, product, contract_nums, sub_ext_ids):
    """Return (Contract instance, Subscription instance) — unsaved."""
    if customer.status == "churned" and customer.churn_date:
        max_end = customer.churn_date
    else:
        max_end = END_DATE

    max_end = max(max_end, customer.acquisition_date + timedelta(days=31))
    start = random_date(customer.acquisition_date, max_end - timedelta(days=30))

    duration_months = random.randint(1, HISTORY_MONTHS)
    end = start + timedelta(days=duration_months * 30)
    end = min(end, END_DATE)
    if customer.status == "churned" and customer.churn_date:
        end = min(end, customer.churn_date)

    status = "active"
    if end < END_DATE - timedelta(days=1):
        status = random.choices(["cancelled", "expired", "active"], weights=[40, 30, 30])[0]
    if customer.status == "churned":
        status = "cancelled"

    discount_pct = Decimal("0.00")
    if random.random() < LEAKAGE_RATES["abnormal_discount"]:
        discount_pct = random_decimal(30, 60)
    elif random.random() < 0.2:
        discount_pct = random_decimal(5, 20)

    qty = random.choices([1, 2, 5, 10], weights=[65, 20, 10, 5])[0]
    unit_price = product.base_price
    mrr_base = unit_price * Decimal(qty) * (Decimal("1") - discount_pct / Decimal("100"))
    if product.billing_cycle == "annual":
        mrr = mrr_base / Decimal("12")
        arr = mrr_base
    else:
        mrr = mrr_base
        arr = mrr * Decimal("12")

    next_billing = None
    if status == "active":
        if product.billing_cycle == "monthly":
            next_billing = (END_DATE.replace(day=1) + timedelta(days=32)).replace(day=1)
        else:
            next_billing = start.replace(year=start.year + 1)

    contract_num = None
    while contract_num is None or contract_num in contract_nums:
        contract_num = uid("CNT")
    contract_nums.add(contract_num)

    contract = Contract(
        customer=customer,
        contract_number=contract_num,
        status="active" if status == "active" else "expired",
        start_date=start,
        end_date=end,
        renewal_date=end + timedelta(days=30) if status == "active" else None,
        contracted_value=_contract_value(product, duration_months),
        currency="USD",
        payment_terms_days=random.choice([15, 30, 45, 60]),
        auto_renew=random.random() < 0.7,
        discount_pct=discount_pct,
    )

    sub_ext_id = None
    while sub_ext_id is None or sub_ext_id in sub_ext_ids:
        sub_ext_id = uid("SUB")
    sub_ext_ids.add(sub_ext_id)

    sub = Subscription(
        customer=customer,
        product=product,
        external_id=sub_ext_id,
        status=status,
        billing_cycle=product.billing_cycle,
        mrr=mrr,
        arr=arr,
        quantity=qty,
        unit_price=unit_price,
        discount_pct=discount_pct,
        start_date=start,
        end_date=end if status != "active" else None,
        next_billing_date=next_billing,
        cancellation_reason=(
            random.choice(["voluntary", "non_payment", "upgrade"])
            if status == "cancelled" else ""
        ),
    )
    # store contract_number on sub so we can look up PK after bulk_create
    sub._tmp_contract_number = contract_num

    return contract, sub


def generate_contracts_and_subscriptions(
    customers: list[Customer], products: list[Product]
) -> tuple[list[Contract], list[Subscription]]:
    existing_subs = Subscription.objects.count()
    if existing_subs >= NUM_SUBSCRIPTIONS:
        print(f"  Subscriptions: {existing_subs} already exist, skipping")
        return list(Contract.objects.all()), list(Subscription.objects.all())

    monthly_products = [p for p in products if p.billing_cycle == "monthly"]
    annual_products = [p for p in products if p.billing_cycle == "annual"]

    contract_nums = set(Contract.objects.values_list("contract_number", flat=True))
    sub_ext_ids = set(Subscription.objects.values_list("external_id", flat=True))

    target = NUM_SUBSCRIPTIONS - existing_subs
    created = 0

    all_customers = list(customers)
    random.shuffle(all_customers)

    contracts_batch = []
    subs_batch = []

    for customer in all_customers:
        if created >= target:
            break

        n_subs = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        for _ in range(n_subs):
            if created >= target:
                break

            if customer.segment in ("enterprise", "mid_market") and random.random() < 0.5:
                product = random.choice(annual_products)
            else:
                product = random.choice(monthly_products)

            c, s = _build_pair(customer, product, contract_nums, sub_ext_ids)
            contracts_batch.append(c)
            subs_batch.append(s)
            created += 1

        # Flush every 500 pairs
        if len(contracts_batch) >= 500:
            _flush(contracts_batch, subs_batch)
            print(f"    {Subscription.objects.count()} subscriptions inserted…")
            contracts_batch = []
            subs_batch = []

    if contracts_batch:
        _flush(contracts_batch, subs_batch)

    total_subs = Subscription.objects.count()
    total_contracts = Contract.objects.count()
    print(f"  Contracts: {total_contracts} | Subscriptions: {total_subs}")
    return list(Contract.objects.all()), list(Subscription.objects.all())


def _flush(contracts_batch: list, subs_batch: list):
    """
    Insert contracts in bulk, look up their PKs by contract_number,
    assign to subs, then bulk-insert subs.
    """
    # Save contracts one-by-one is too slow for 8k records.
    # Instead: bulk_create, then lookup PKs by unique contract_number.
    Contract.objects.bulk_create(contracts_batch, ignore_conflicts=True)

    nums = [c.contract_number for c in contracts_batch]
    pk_map = dict(
        Contract.objects.filter(contract_number__in=nums).values_list("contract_number", "pk")
    )

    for s in subs_batch:
        pk = pk_map.get(s._tmp_contract_number)
        if pk:
            s.contract_id = pk
        del s._tmp_contract_number

    Subscription.objects.bulk_create(subs_batch, ignore_conflicts=True)
