"""
Rule-based reconciliation engine.

Scans the database for structural anomalies and writes RuleAlert rows.
Each rule function returns a list of RuleAlert objects (unsaved).

Usage:
    venv/Scripts/python.exe ml_pipeline/rule_engine.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from decimal import Decimal
from django.db import connection
from django.utils import timezone

from apps.leakage_cases.models import RuleAlert
from ml_pipeline.utils import get_logger

log = get_logger("rule_engine")

BATCH_SIZE = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bulk_insert(alerts: list[RuleAlert]):
    if alerts:
        RuleAlert.objects.bulk_create(alerts, ignore_conflicts=True)


# ── Rule implementations ──────────────────────────────────────────────────────

def rule_missing_payment() -> list[RuleAlert]:
    """MISSING_PAYMENT — invoice issued/overdue with no completed payment."""
    with connection.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.customer_id, i.total_amount, i.outstanding_amount,
                   i.invoice_number, i.due_date
            FROM invoices i
            LEFT JOIN payments p
                ON p.invoice_id = i.id AND p.status = 'completed'
            WHERE i.is_deleted = 0
              AND i.status IN ('issued', 'overdue')
              AND i.total_amount > 0
              AND p.id IS NULL
        """)
        rows = cur.fetchall()

    alerts = []
    for inv_id, cust_id, total, outstanding, inv_num, due_date in rows:
        severity = "high" if (due_date and due_date < timezone.now().date()) else "medium"
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=inv_id,
            rule_code="MISSING_PAYMENT",
            severity=severity,
            description=f"Invoice {inv_num} (${float(total):,.2f}) has no completed payment.",
            leakage_amount=Decimal(str(outstanding or total)),
        ))
    log.info("MISSING_PAYMENT: %d alerts", len(alerts))
    return alerts


def rule_duplicate_refund() -> list[RuleAlert]:
    """DUPLICATE_REFUND — refunds flagged as duplicates."""
    with connection.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.customer_id, r.invoice_id, r.amount, r.refund_number
            FROM refunds r
            WHERE r.is_duplicate = 1
        """)
        rows = cur.fetchall()

    alerts = []
    for ref_id, cust_id, inv_id, amount, ref_num in rows:
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=inv_id,
            rule_code="DUPLICATE_REFUND",
            severity="high",
            description=f"Duplicate refund detected: {ref_num} (${float(amount):,.2f}).",
            leakage_amount=Decimal(str(amount)),
            metadata={"refund_id": ref_id, "refund_number": ref_num},
        ))
    log.info("DUPLICATE_REFUND: %d alerts", len(alerts))
    return alerts


def rule_abnormal_discount() -> list[RuleAlert]:
    """ABNORMAL_DISCOUNT — discount ratio > 35% of invoice total."""
    THRESHOLD = 0.35
    with connection.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.customer_id, i.invoice_number,
                   i.total_amount, i.discount_amount
            FROM invoices i
            WHERE i.is_deleted = 0
              AND i.total_amount > 0
              AND i.discount_amount / i.total_amount > %s
        """, [THRESHOLD])
        rows = cur.fetchall()

    alerts = []
    for inv_id, cust_id, inv_num, total, discount in rows:
        ratio = float(discount) / float(total)
        severity = "critical" if ratio > 0.60 else "high" if ratio > 0.45 else "medium"
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=inv_id,
            rule_code="ABNORMAL_DISCOUNT",
            severity=severity,
            description=(
                f"Invoice {inv_num}: discount ratio {ratio:.1%} "
                f"(${float(discount):,.2f} on ${float(total):,.2f})."
            ),
            leakage_amount=Decimal(str(discount)),
            metadata={"discount_ratio": round(ratio, 4)},
        ))
    log.info("ABNORMAL_DISCOUNT: %d alerts", len(alerts))
    return alerts


def rule_underbilling() -> list[RuleAlert]:
    """UNDERBILLING — invoice total < 85% of subscription MRR."""
    THRESHOLD = 0.85
    with connection.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.customer_id, i.invoice_number,
                   i.total_amount, s.mrr
            FROM invoices i
            JOIN subscriptions s ON i.subscription_id = s.id
            WHERE i.is_deleted = 0
              AND i.total_amount > 0
              AND i.total_amount < s.mrr * %s
              AND s.mrr > 0
        """, [THRESHOLD])
        rows = cur.fetchall()

    alerts = []
    for inv_id, cust_id, inv_num, total, mrr in rows:
        gap = float(mrr) - float(total)
        ratio = float(total) / float(mrr)
        severity = "critical" if ratio < 0.50 else "high" if ratio < 0.70 else "medium"
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=inv_id,
            rule_code="UNDERBILLING",
            severity=severity,
            description=(
                f"Invoice {inv_num}: billed ${float(total):,.2f} "
                f"vs MRR ${float(mrr):,.2f} ({ratio:.1%})."
            ),
            leakage_amount=Decimal(str(round(gap, 2))),
            metadata={"billed": float(total), "mrr": float(mrr), "ratio": round(ratio, 4)},
        ))
    log.info("UNDERBILLING: %d alerts", len(alerts))
    return alerts


def rule_payment_delay() -> list[RuleAlert]:
    """PAYMENT_DELAY — completed payment received > 30 days after due date."""
    MIN_DAYS_LATE = 30
    with connection.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.customer_id, i.invoice_number,
                   i.total_amount, p.days_late, p.payment_date
            FROM payments p
            JOIN invoices i ON p.invoice_id = i.id
            WHERE p.status = 'completed'
              AND p.is_late = 1
              AND p.days_late >= %s
              AND i.is_deleted = 0
        """, [MIN_DAYS_LATE])
        rows = cur.fetchall()

    alerts = []
    for inv_id, cust_id, inv_num, total, days_late, pay_date in rows:
        severity = "high" if days_late >= 90 else "medium"
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=inv_id,
            rule_code="PAYMENT_DELAY",
            severity=severity,
            description=(
                f"Invoice {inv_num}: payment received {days_late} days late "
                f"(paid {pay_date})."
            ),
            leakage_amount=Decimal("0"),
            metadata={"days_late": days_late, "payment_date": str(pay_date)},
        ))
    log.info("PAYMENT_DELAY: %d alerts", len(alerts))
    return alerts


def rule_failed_payment_streak() -> list[RuleAlert]:
    """FAILED_PAYMENT_STREAK — customer with 3+ failed payments."""
    MIN_FAILS = 3
    with connection.cursor() as cur:
        cur.execute("""
            SELECT p.customer_id, COUNT(*) AS fails,
                   SUM(p.amount) AS at_risk
            FROM payments p
            WHERE p.status = 'failed'
              AND p.is_deleted = 0
            GROUP BY p.customer_id
            HAVING fails >= %s
        """, [MIN_FAILS])
        rows = cur.fetchall()

    alerts = []
    for cust_id, fails, at_risk in rows:
        severity = "critical" if fails >= 10 else "high" if fails >= 6 else "medium"
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=None,
            rule_code="FAILED_PAYMENT_STREAK",
            severity=severity,
            description=f"Customer has {fails} failed payments totalling ${float(at_risk):,.2f}.",
            leakage_amount=Decimal(str(round(float(at_risk), 2))),
            metadata={"failed_count": fails, "total_at_risk": float(at_risk)},
        ))
    log.info("FAILED_PAYMENT_STREAK: %d alerts", len(alerts))
    return alerts


def rule_overdue_invoice() -> list[RuleAlert]:
    """MISSING_PAYMENT (overdue variant) — status=overdue past due_date."""
    with connection.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.customer_id, i.invoice_number,
                   i.outstanding_amount, i.due_date,
                   DATEDIFF(CURDATE(), i.due_date) AS days_overdue
            FROM invoices i
            WHERE i.is_deleted = 0
              AND i.status = 'overdue'
              AND i.due_date < CURDATE()
        """)
        rows = cur.fetchall()

    alerts = []
    for inv_id, cust_id, inv_num, outstanding, due_date, days_over in rows:
        severity = "critical" if days_over >= 90 else "high" if days_over >= 30 else "medium"
        alerts.append(RuleAlert(
            customer_id=cust_id,
            invoice_id=inv_id,
            rule_code="MISSING_RENEWAL",
            severity=severity,
            description=(
                f"Invoice {inv_num} is {days_over} days overdue "
                f"(due {due_date}). Outstanding: ${float(outstanding):,.2f}."
            ),
            leakage_amount=Decimal(str(outstanding or 0)),
            metadata={"days_overdue": days_over, "due_date": str(due_date)},
        ))
    log.info("MISSING_RENEWAL (overdue): %d alerts", len(alerts))
    return alerts


# ── Master runner ─────────────────────────────────────────────────────────────

RULES = [
    rule_missing_payment,
    rule_duplicate_refund,
    rule_abnormal_discount,
    rule_underbilling,
    rule_payment_delay,
    rule_failed_payment_streak,
    rule_overdue_invoice,
]


def run_rule_engine(clear_existing: bool = False):
    """Run all rules and bulk-insert RuleAlert rows."""
    if clear_existing:
        deleted, _ = RuleAlert.objects.all().delete()
        log.info("Cleared %d existing RuleAlert rows", deleted)

    existing = RuleAlert.objects.count()
    if existing > 0 and not clear_existing:
        log.info("RuleAlert table already has %d rows — skipping (use --clear to re-run)", existing)
        return

    t0 = time.time()
    log.info("=" * 55)
    log.info("Running rule-based reconciliation engine…")
    log.info("=" * 55)

    total = 0
    for rule_fn in RULES:
        alerts = rule_fn()
        # Batch-insert
        for i in range(0, len(alerts), BATCH_SIZE):
            _bulk_insert(alerts[i : i + BATCH_SIZE])
        total += len(alerts)

    elapsed = time.time() - t0
    final_count = RuleAlert.objects.count()
    log.info("=" * 55)
    log.info("Rule engine complete in %.1fs", elapsed)
    log.info("Total RuleAlerts inserted: %d (table total: %d)", total, final_count)
    log.info("=" * 55)

    # Per-rule summary
    from django.db.models import Count, Sum
    summary = (
        RuleAlert.objects
        .values("rule_code", "severity")
        .annotate(cnt=Count("id"), leakage=Sum("leakage_amount"))
        .order_by("rule_code", "severity")
    )
    for row in summary:
        leakage = float(row["leakage"] or 0)
        log.info("  %-30s  %-8s  %5d alerts  $%s leakage",
                 row["rule_code"], row["severity"], row["cnt"],
                 f"{leakage:,.0f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Revenue Leakage Rule Engine")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing RuleAlerts before re-running")
    args = parser.parse_args()
    run_rule_engine(clear_existing=args.clear)
