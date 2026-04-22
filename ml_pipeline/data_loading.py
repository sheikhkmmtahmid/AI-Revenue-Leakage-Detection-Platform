"""
Loads raw data from MySQL into pandas DataFrames for the ML pipeline.
All queries use Django ORM so connection settings stay centralised.
"""

import os
import sys
import django
import pandas as pd

# Bootstrap Django when running standalone
if not django.conf.settings.configured:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()

from django.db import connection


def _query(sql: str, params=None) -> pd.DataFrame:
    with connection.cursor() as cur:
        cur.execute(sql, params or [])
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_invoice_features() -> pd.DataFrame:
    """
    Invoice-level feature table joining invoices, payments, refunds,
    subscriptions, and contracts.
    """
    sql = """
        SELECT
            i.id                                        AS invoice_id,
            i.customer_id,
            i.subscription_id,
            i.invoice_number,
            i.status                                    AS invoice_status,
            i.issue_date,
            i.due_date,
            i.period_start,
            i.period_end,
            i.subtotal,
            i.tax_amount,
            i.discount_amount,
            i.total_amount,
            i.paid_amount,
            i.outstanding_amount,

            -- Payment aggregates for this invoice
            COALESCE(p_agg.total_paid, 0)               AS total_paid,
            COALESCE(p_agg.payment_count, 0)            AS payment_count,
            COALESCE(p_agg.failed_count, 0)             AS failed_payment_count,
            COALESCE(p_agg.max_days_late, 0)            AS max_days_late,
            COALESCE(p_agg.avg_days_late, 0)            AS avg_days_late,
            COALESCE(p_agg.attempt_sum, 0)              AS total_attempts,

            -- Refund aggregates
            COALESCE(r_agg.refund_count, 0)             AS refund_count,
            COALESCE(r_agg.refund_amount, 0)            AS refund_amount,
            COALESCE(r_agg.dup_refund_count, 0)         AS duplicate_refund_count,

            -- Subscription context
            s.billing_cycle,
            s.mrr,
            s.discount_pct                              AS sub_discount_pct,
            s.quantity,

            -- Contract context
            c.contracted_value,
            c.payment_terms_days,
            c.discount_pct                              AS contract_discount_pct

        FROM invoices i

        LEFT JOIN (
            SELECT
                invoice_id,
                SUM(CASE WHEN status = 'succeeded' THEN amount ELSE 0 END) AS total_paid,
                COUNT(*)                                                      AS payment_count,
                SUM(CASE WHEN status = 'failed'    THEN 1     ELSE 0 END)  AS failed_count,
                MAX(days_late)                                                AS max_days_late,
                AVG(days_late)                                                AS avg_days_late,
                SUM(attempt_count)                                            AS attempt_sum
            FROM payments
            GROUP BY invoice_id
        ) p_agg ON p_agg.invoice_id = i.id

        LEFT JOIN (
            SELECT
                invoice_id,
                COUNT(*)                                     AS refund_count,
                SUM(amount)                                  AS refund_amount,
                SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) AS dup_refund_count
            FROM refunds
            GROUP BY invoice_id
        ) r_agg ON r_agg.invoice_id = i.id

        LEFT JOIN subscriptions s ON s.id = i.subscription_id
        LEFT JOIN contracts    c ON c.id = i.contract_id

        WHERE i.is_deleted = 0
    """
    return _query(sql)


def load_customer_features() -> pd.DataFrame:
    """Account-level aggregated features per customer."""
    sql = """
        SELECT
            c.id                                            AS customer_id,
            c.segment,
            c.status                                        AS customer_status,
            c.acquisition_date,
            c.churn_date,
            c.risk_tier,
            DATEDIFF(CURDATE(), c.acquisition_date)        AS tenure_days,

            -- Subscription stats
            COALESCE(s_agg.active_subs, 0)                 AS active_subscriptions,
            COALESCE(s_agg.total_subs, 0)                  AS total_subscriptions,
            COALESCE(s_agg.avg_mrr, 0)                     AS avg_mrr,
            COALESCE(s_agg.total_arr, 0)                   AS total_arr,
            COALESCE(s_agg.avg_discount_pct, 0)            AS avg_discount_pct,

            -- Invoice stats
            COALESCE(i_agg.invoice_count, 0)               AS invoice_count,
            COALESCE(i_agg.total_invoiced, 0)              AS total_invoiced,
            COALESCE(i_agg.total_outstanding, 0)           AS total_outstanding,
            COALESCE(i_agg.overdue_count, 0)               AS overdue_invoice_count,

            -- Payment stats
            COALESCE(p_agg.payment_count, 0)               AS payment_count,
            COALESCE(p_agg.failed_count, 0)                AS failed_payment_count,
            COALESCE(p_agg.total_paid, 0)                  AS total_paid,
            COALESCE(p_agg.avg_days_late, 0)               AS avg_days_late,

            -- Refund stats
            COALESCE(r_agg.refund_count, 0)                AS refund_count,
            COALESCE(r_agg.total_refunded, 0)              AS total_refunded,
            COALESCE(r_agg.dup_count, 0)                   AS duplicate_refund_count

        FROM customers c

        LEFT JOIN (
            SELECT
                customer_id,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_subs,
                COUNT(*)                                             AS total_subs,
                AVG(mrr)                                             AS avg_mrr,
                SUM(arr)                                             AS total_arr,
                AVG(discount_pct)                                    AS avg_discount_pct
            FROM subscriptions
            WHERE is_deleted = 0
            GROUP BY customer_id
        ) s_agg ON s_agg.customer_id = c.id

        LEFT JOIN (
            SELECT
                customer_id,
                COUNT(*)                                                     AS invoice_count,
                SUM(total_amount)                                            AS total_invoiced,
                SUM(outstanding_amount)                                      AS total_outstanding,
                SUM(CASE WHEN status = 'overdue' THEN 1 ELSE 0 END)         AS overdue_count
            FROM invoices
            WHERE is_deleted = 0
            GROUP BY customer_id
        ) i_agg ON i_agg.customer_id = c.id

        LEFT JOIN (
            SELECT
                customer_id,
                COUNT(*)                                                      AS payment_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)           AS failed_count,
                SUM(CASE WHEN status = 'succeeded' THEN amount ELSE 0 END)   AS total_paid,
                AVG(days_late)                                                 AS avg_days_late
            FROM payments
            WHERE is_deleted = 0
            GROUP BY customer_id
        ) p_agg ON p_agg.customer_id = c.id

        LEFT JOIN (
            SELECT
                customer_id,
                COUNT(*)                                            AS refund_count,
                SUM(amount)                                         AS total_refunded,
                SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END)      AS dup_count
            FROM refunds
            WHERE is_deleted = 0
            GROUP BY customer_id
        ) r_agg ON r_agg.customer_id = c.id

        WHERE c.is_deleted = 0
    """
    return _query(sql)


def load_monthly_revenue() -> pd.DataFrame:
    """Time-series monthly revenue aggregation for forecasting."""
    sql = """
        SELECT
            DATE_FORMAT(i.issue_date, '%%Y-%%m')        AS period_month,
            SUM(i.total_amount)                          AS total_invoiced,
            SUM(i.paid_amount)                           AS total_collected,
            COUNT(i.id)                                  AS invoice_count,
            SUM(i.discount_amount)                       AS total_discounted,
            SUM(i.tax_amount)                            AS total_tax,
            SUM(i.outstanding_amount)                    AS total_outstanding,
            SUM(CASE WHEN i.status = 'overdue' THEN i.outstanding_amount ELSE 0 END) AS overdue_amount
        FROM invoices i
        WHERE i.is_deleted = 0
        GROUP BY DATE_FORMAT(i.issue_date, '%%Y-%%m')
        ORDER BY period_month
    """
    return _query(sql)


if __name__ == "__main__":
    print("Loading invoice features…")
    df_inv = load_invoice_features()
    print(f"  Invoice features shape: {df_inv.shape}")

    print("Loading customer features…")
    df_cust = load_customer_features()
    print(f"  Customer features shape: {df_cust.shape}")

    print("Loading monthly revenue…")
    df_rev = load_monthly_revenue()
    print(f"  Monthly revenue shape: {df_rev.shape}")
    print(df_rev.head())
