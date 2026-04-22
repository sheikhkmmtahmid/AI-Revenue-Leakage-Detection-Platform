"""
Synthetic data generation configuration.
All tunable parameters live here so the generator is config-driven.
"""

from datetime import date

# ── Scale ────────────────────────────────────────────────────────────────────
NUM_CUSTOMERS = 5_000
NUM_SUBSCRIPTIONS = 8_000
NUM_INVOICES = 50_000
NUM_PAYMENTS = 45_000
NUM_REFUNDS = 5_000
HISTORY_MONTHS = 36  # 3 years

START_DATE = date(2022, 1, 1)

# ── Customer segments ────────────────────────────────────────────────────────
SEGMENTS = {
    "enterprise": 0.10,
    "mid_market": 0.20,
    "smb": 0.50,
    "startup": 0.20,
}

# ── Products / plans ─────────────────────────────────────────────────────────
PRODUCTS = [
    {"code": "STARTER_M", "name": "Starter Monthly", "price": 49.00, "cycle": "monthly"},
    {"code": "PRO_M", "name": "Pro Monthly", "price": 199.00, "cycle": "monthly"},
    {"code": "BUSINESS_M", "name": "Business Monthly", "price": 499.00, "cycle": "monthly"},
    {"code": "ENTERPRISE_M", "name": "Enterprise Monthly", "price": 1_499.00, "cycle": "monthly"},
    {"code": "STARTER_A", "name": "Starter Annual", "price": 470.00, "cycle": "annual"},
    {"code": "PRO_A", "name": "Pro Annual", "price": 1_990.00, "cycle": "annual"},
    {"code": "BUSINESS_A", "name": "Business Annual", "price": 4_990.00, "cycle": "annual"},
    {"code": "ENTERPRISE_A", "name": "Enterprise Annual", "price": 14_900.00, "cycle": "annual"},
]

# ── Leakage scenario injection rates ────────────────────────────────────────
LEAKAGE_RATES = {
    "missing_payment": 0.06,        # 6% of invoices → no payment
    "underbilling": 0.04,           # 4% billed below contract
    "duplicate_refund": 0.03,       # 3% of refunds are duplicates
    "abnormal_discount": 0.05,      # 5% have excess discounts
    "missing_renewal": 0.04,        # 4% of expired contracts not renewed
    "payment_delay": 0.12,          # 12% of payments are late
    "failed_payment_streak": 0.05,  # 5% customers have ≥3 consecutive fails
    "zero_revenue_period": 0.02,    # 2% of active customers have a zero-revenue month
}

# ── Payment behaviour params ─────────────────────────────────────────────────
PAYMENT_SUCCESS_RATE = 0.87
MAX_RETRY_ATTEMPTS = 3
LATE_PAYMENT_DAYS_RANGE = (1, 60)
DISCOUNT_NORMAL_MAX_PCT = 20.0
DISCOUNT_ABUSE_MAX_PCT = 60.0
TAX_RATE = 0.08
