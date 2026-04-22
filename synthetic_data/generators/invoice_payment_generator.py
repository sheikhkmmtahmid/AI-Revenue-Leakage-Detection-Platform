"""Generates Invoice, Payment, and Refund rows with injected leakage scenarios."""

import random
import uuid
from datetime import date, timedelta, datetime
from decimal import Decimal
from django.utils import timezone

from apps.customers.models import Customer
from apps.subscriptions.models import Subscription
from apps.invoices.models import Invoice
from apps.payments.models import Payment
from apps.refunds.models import Refund
from synthetic_data.configs.data_config import (
    NUM_INVOICES, NUM_PAYMENTS, NUM_REFUNDS,
    LEAKAGE_RATES, PAYMENT_SUCCESS_RATE, MAX_RETRY_ATTEMPTS,
    LATE_PAYMENT_DAYS_RANGE, TAX_RATE, START_DATE, HISTORY_MONTHS,
)
from synthetic_data.generators.base_generator import uid, random_date, random_decimal

random.seed(42)
END_DATE = START_DATE + timedelta(days=HISTORY_MONTHS * 30)


def _invoice_number() -> str:
    return f"INV-{uuid.uuid4().hex[:10].upper()}"


def _refund_number() -> str:
    return f"REF-{uuid.uuid4().hex[:8].upper()}"


def _txn_id() -> str:
    return f"TXN-{uuid.uuid4().hex[:12].upper()}"


def generate_invoices(subscriptions: list[Subscription]) -> list[Invoice]:
    existing = Invoice.objects.count()
    if existing >= NUM_INVOICES:
        print(f"  Invoices: {existing} already exist, skipping")
        return list(Invoice.objects.all())

    target = NUM_INVOICES - existing
    inv_numbers = set(Invoice.objects.values_list("invoice_number", flat=True))
    batch = []

    active_subs = [s for s in subscriptions if s.status in ("active", "cancelled", "expired")]
    random.shuffle(active_subs)

    created = 0
    for sub in active_subs:
        if created >= target:
            break

        # How many invoices for this subscription?
        sub_start = sub.start_date
        sub_end = sub.end_date or END_DATE

        if sub.billing_cycle == "monthly":
            n_invoices = max(1, ((sub_end - sub_start).days // 30))
        else:
            n_invoices = max(1, ((sub_end - sub_start).days // 365))

        n_invoices = min(n_invoices, target - created, 24)

        current = sub_start
        for _ in range(n_invoices):
            if created >= target:
                break

            if sub.billing_cycle == "monthly":
                period_end = current + timedelta(days=30)
            else:
                period_end = current + timedelta(days=365)
            period_end = min(period_end, sub_end, END_DATE)

            issue_date = current
            due_date = issue_date + timedelta(days=sub.contract.payment_terms_days if sub.contract else 30)

            # Compute amounts
            base = float(sub.unit_price) * sub.quantity
            if sub.billing_cycle == "annual":
                base = base / 12  # monthly invoice even for annual
            discount_pct = float(sub.discount_pct) / 100
            discount_amount = Decimal(str(round(base * discount_pct, 2)))
            subtotal = Decimal(str(round(base - float(discount_amount), 2)))
            tax = Decimal(str(round(float(subtotal) * TAX_RATE, 2)))
            total = subtotal + tax

            # Leakage: underbilling — bill less than contracted
            if random.random() < LEAKAGE_RATES["underbilling"]:
                total = total * Decimal(str(round(random.uniform(0.5, 0.9), 2)))
                subtotal = total - tax

            inv_num = None
            while inv_num is None or inv_num in inv_numbers:
                inv_num = _invoice_number()
            inv_numbers.add(inv_num)

            # Missing payment scenario → mark as issued/overdue with no payment
            missing_payment = random.random() < LEAKAGE_RATES["missing_payment"]

            status = "issued"
            if due_date < END_DATE and not missing_payment:
                status = "paid"
            elif due_date < END_DATE and missing_payment:
                status = "overdue"

            batch.append(Invoice(
                customer=sub.customer,
                contract=sub.contract,
                subscription=sub,
                invoice_number=inv_num,
                status=status,
                issue_date=issue_date,
                due_date=due_date,
                period_start=current,
                period_end=period_end,
                subtotal=subtotal,
                tax_amount=tax,
                discount_amount=discount_amount,
                total_amount=total,
                paid_amount=total if status == "paid" else Decimal("0.00"),
                outstanding_amount=Decimal("0.00") if status == "paid" else total,
                currency="USD",
            ))
            created += 1
            current = period_end

        if len(batch) >= 1000:
            Invoice.objects.bulk_create(batch, ignore_conflicts=True)
            print(f"    {Invoice.objects.count()} invoices inserted…")
            batch = []

    if batch:
        Invoice.objects.bulk_create(batch, ignore_conflicts=True)

    total = Invoice.objects.count()
    print(f"  Invoices: {total}")
    return list(Invoice.objects.all())


def generate_payments(invoices: list[Invoice]) -> list[Payment]:
    existing = Payment.objects.count()
    if existing >= NUM_PAYMENTS:
        print(f"  Payments: {existing} already exist, skipping")
        return list(Payment.objects.all())

    target = NUM_PAYMENTS - existing
    txn_ids = set(Payment.objects.values_list("transaction_id", flat=True))
    batch = []
    created = 0

    payable = [inv for inv in invoices if inv.status in ("paid", "partially_paid", "overdue")]
    random.shuffle(payable)

    methods = ["credit_card", "bank_transfer", "ach", "wire", "check"]
    method_weights = [50, 20, 15, 10, 5]

    for inv in payable:
        if created >= target:
            break

        # Determine payment outcome
        success = random.random() < PAYMENT_SUCCESS_RATE
        attempts = 1 if success else random.randint(1, MAX_RETRY_ATTEMPTS)

        # Late payment injection
        is_late = random.random() < LEAKAGE_RATES["payment_delay"]
        days_late = random.randint(*LATE_PAYMENT_DAYS_RANGE) if is_late else 0
        payment_date = inv.due_date + timedelta(days=days_late)
        payment_date = min(payment_date, END_DATE)

        txn = None
        while txn is None or txn in txn_ids:
            txn = _txn_id()
        txn_ids.add(txn)

        status = "succeeded" if success else "failed"
        amount = inv.total_amount if success else Decimal("0.00")
        # Failed payment streak injection
        if random.random() < LEAKAGE_RATES["failed_payment_streak"]:
            status = "failed"
            amount = Decimal("0.00")

        batch.append(Payment(
            customer=inv.customer,
            invoice=inv,
            transaction_id=txn,
            status=status,
            method=random.choices(methods, weights=method_weights)[0],
            amount=amount,
            currency="USD",
            payment_date=payment_date,
            is_late=is_late,
            days_late=days_late,
            attempt_count=attempts,
            failure_reason="" if success else random.choice([
                "insufficient_funds", "card_declined", "bank_error",
                "timeout", "fraud_detected",
            ]),
            processor=random.choice(["Stripe", "Braintree", "Adyen", "PayPal"]),
        ))
        created += 1

        if len(batch) >= 1000:
            Payment.objects.bulk_create(batch, ignore_conflicts=True)
            print(f"    {Payment.objects.count()} payments inserted…")
            batch = []

    if batch:
        Payment.objects.bulk_create(batch, ignore_conflicts=True)

    total = Payment.objects.count()
    print(f"  Payments: {total}")
    return list(Payment.objects.all())


def generate_refunds(payments: list[Payment]) -> list[Refund]:
    existing = Refund.objects.count()
    if existing >= NUM_REFUNDS:
        print(f"  Refunds: {existing} already exist, skipping")
        return list(Refund.objects.all())

    target = NUM_REFUNDS - existing
    ref_nums = set(Refund.objects.values_list("refund_number", flat=True))
    batch = []

    succeeded = [p for p in payments if p.status == "succeeded" and p.amount > 0]
    random.shuffle(succeeded)

    reasons = [
        "duplicate_charge", "service_cancellation", "billing_error",
        "contract_termination", "customer_request", "fraud",
    ]
    reason_weights = [15, 25, 20, 15, 20, 5]

    created = 0
    for pay in succeeded:
        if created >= target:
            break

        ref_num = None
        while ref_num is None or ref_num in ref_nums:
            ref_num = _refund_number()
        ref_nums.add(ref_num)

        is_dup = random.random() < LEAKAGE_RATES["duplicate_refund"]
        amount = pay.amount if is_dup else pay.amount * Decimal(str(round(random.uniform(0.1, 1.0), 2)))

        requested_date = pay.payment_date + timedelta(days=random.randint(1, 30))
        requested_at = timezone.make_aware(datetime(requested_date.year, requested_date.month, requested_date.day))
        processed_days = random.randint(1, 7)
        processed_at = None
        if random.random() > 0.1:
            processed_date = requested_date + timedelta(days=processed_days)
            processed_at = timezone.make_aware(datetime(processed_date.year, processed_date.month, processed_date.day))

        batch.append(Refund(
            customer=pay.customer,
            payment=pay,
            invoice=pay.invoice,
            refund_number=ref_num,
            status=random.choice(["processed", "processed", "processed", "pending", "rejected"]),
            reason=random.choices(reasons, weights=reason_weights)[0],
            amount=amount,
            currency="USD",
            requested_at=requested_at,
            processed_at=processed_at,
            is_duplicate=is_dup,
        ))
        created += 1

        if len(batch) >= 500:
            Refund.objects.bulk_create(batch, ignore_conflicts=True)
            print(f"    {Refund.objects.count()} refunds inserted…")
            batch = []

    if batch:
        Refund.objects.bulk_create(batch, ignore_conflicts=True)

    total = Refund.objects.count()
    print(f"  Refunds: {total}")
    return list(Refund.objects.all())
