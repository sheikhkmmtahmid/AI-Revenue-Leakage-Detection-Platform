# AI Revenue Leakage Detection Platform

A production-grade Django + MySQL SaaS application that detects revenue leakage through three complementary approaches: **rule-based reconciliation**, **ML anomaly detection** (Isolation Forest + XGBoost ensemble), and **time-series forecasting** (Prophet/SARIMA). Includes a full-featured interactive dashboard.

---

## What This Project Does

Revenue leakage occurs when a business is owed money it never collects — through missed invoices, duplicate refunds, underbilling, abnormal discounts, or failed payments. This platform:

1. **Ingests** synthetic B2B SaaS billing data (5,000 customers, 50,000 invoices, 45,000 payments)
2. **Engineers features** (47 invoice-level, 30 customer-level features)
3. **Trains three ML models** to score every invoice for leakage risk:
   - Isolation Forest — unsupervised anomaly detection
   - Logistic Regression — interpretable baseline classifier
   - XGBoost — main ensemble model (70% XGB + 30% LR weighted average)
4. **Runs a rule engine** that fires alerts for structural anomalies (missing payments, duplicate refunds, underbilling, overdue invoices, etc.)
5. **Forecasts** monthly revenue with Prophet and flags months where actuals deviate significantly from predictions
6. **Serves a dashboard** at `http://localhost:8000` with live charts, clickable KPI cards, and per-invoice ML explanations

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 · Django 5.2 · Django REST Framework |
| Database | MySQL 8.0 |
| ML | scikit-learn 1.8 · XGBoost 3.2 · SHAP 0.51 |
| Forecasting | Prophet 1.3 · statsmodels 0.14 |
| Data | pandas 3.0 · numpy 2.4 · Faker 40 |
| Task Queue | Celery 5.6 · Redis (optional, for async tasks) |
| Frontend | Chart.js 4.4 · Bootstrap 5.3 · Bootstrap Icons |

---

## Project Structure

```
AI Revenue Leakage Detection Platform/
│
├── config/                   # Django project settings
│   ├── settings.py           # DB, apps, static files config
│   ├── urls.py               # Root URL routing
│   └── wsgi.py
│
├── apps/                     # Django applications
│   ├── customers/            # Customer model & API
│   ├── invoices/             # Invoice model & API
│   ├── payments/             # Payment model
│   ├── refunds/              # Refund model (duplicate detection)
│   ├── contracts/            # Contract model
│   ├── subscriptions/        # Subscription model (MRR/ARR)
│   ├── risk_scoring/         # RiskScore model & API (XGBoost output)
│   ├── anomaly_detection/    # AnomalyScore model & API (IF output)
│   ├── leakage_cases/        # RuleAlert & LeakageCase models
│   ├── forecasting/          # ForecastResult model & API
│   ├── analytics/            # Dashboard views & all API endpoints
│   └── common/               # Shared base models (TimeStampedModel)
│
├── ml_pipeline/              # ML code (runs outside Django request cycle)
│   ├── run_pipeline.py       # Master runner: train → evaluate → infer → write DB
│   ├── data_loading.py       # Loads invoice data from MySQL into DataFrames
│   ├── feature_engineering.py# Engineers 47 ML features per invoice
│   ├── train_anomaly.py      # Isolation Forest training & scoring
│   ├── train_baseline.py     # Logistic Regression training & scoring
│   ├── train_xgboost.py      # XGBoost training & scoring
│   ├── evaluation.py         # Precision/Recall/AUC metrics & comparison table
│   ├── inference.py          # Batch scoring & DB writes (anomaly_scores, risk_scores)
│   ├── explainability.py     # SHAP value computation
│   ├── forecasting.py        # Prophet/SARIMA forecast & DB writes
│   ├── rule_engine.py        # Rule-based reconciliation → rule_alerts table
│   └── utils.py              # Logger, model save/load, time-split
│
├── synthetic_data/           # Synthetic data generators
│   └── generators/
│       └── run_all.py        # Master runner (idempotent — safe to re-run)
│
├── templates/
│   └── dashboard/
│       └── index.html        # Full-featured SPA dashboard
│
├── artifacts/                # Generated files (git-ignored in production)
│   ├── models/               # Saved ML models (.pkl)
│   ├── shap/                 # SHAP value exports
│   ├── forecasts/            # Forecast CSVs
│   └── reports/              # Evaluation metrics & feature importance
│
└── venv/                     # Python virtual environment (Windows)
```

---

## Prerequisites

Before running on a **new machine**, install:

1. **Python 3.11** — https://python.org/downloads  
   (Do NOT use Python 3.12+ — Prophet/SHAP require 3.11)
2. **MySQL 8.0** — https://dev.mysql.com/downloads/mysql/
3. **MySQL C connector** (required by `mysqlclient`):
   - Windows: included with MySQL installer
   - Ubuntu/Debian: `sudo apt install libmysqlclient-dev`
   - macOS: `brew install mysql-client`
4. **Git** (optional, for cloning)

---

## Step 1 — MySQL Setup (first time only)

Open MySQL shell (or MySQL Workbench):

```sql
-- Create the database
CREATE DATABASE revenue_leakage_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create a user (or use root)
-- If using root with password AhzUttara@2025, skip this block
CREATE USER 'rl_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON revenue_leakage_db.* TO 'rl_user'@'localhost';
FLUSH PRIVILEGES;
```

The project is **pre-configured** in `config/settings.py` to use:
```
Database: revenue_leakage_db
User:     root
Password: AhzUttara@2025
Host:     localhost
Port:     3306
```

To change these, edit `config/settings.py` → `DATABASES` section.

---

## Step 2 — Clone & Create Virtual Environment

```bash
# Clone (or unzip) the project
cd "d:/AI Revenue Leakage Detection Platform"

# Create virtual environment using Python 3.11
# Windows (where python3.11 is the 3.11 executable):
"C:\Users\<YourName>\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv

# Or if python3.11 is on PATH:
python3.11 -m venv venv
```

---

## Step 3 — Install Dependencies

**Windows (Git Bash or Command Prompt):**
```bash
# Activate virtual environment
venv\Scripts\activate        # Command Prompt
# OR
source venv/Scripts/activate # Git Bash

# Install all packages
pip install django==5.2.13 djangorestframework==3.17.1 django-filter==25.2 django-extensions==4.1
pip install mysqlclient==2.2.8
pip install pandas numpy scikit-learn xgboost shap
pip install prophet statsmodels
pip install celery redis faker
pip install matplotlib joblib
```

**macOS / Linux:**
```bash
source venv/bin/activate
pip install django djangorestframework django-filter django-extensions
pip install mysqlclient
pip install pandas numpy scikit-learn xgboost shap
pip install prophet statsmodels
pip install celery redis faker matplotlib joblib
```

> **Windows note:** All `python` commands in this project use the full path  
> `venv/Scripts/python.exe` to avoid conflicts with any system Python.

---

## Step 4 — Run Django Migrations (first time only)

This creates all 26 tables in MySQL:

```bash
venv/Scripts/python.exe manage.py makemigrations
venv/Scripts/python.exe manage.py migrate
```

Verify tables were created:
```sql
-- In MySQL shell
USE revenue_leakage_db;
SHOW TABLES;
-- Should show 26 tables: customers, invoices, payments, refunds, contracts,
-- subscriptions, products, risk_scores, anomaly_scores, rule_alerts,
-- leakage_cases, forecast_results, ...
```

---

## Step 5 — Generate Synthetic Data (first time only)

Populates ~100,000 rows across all tables. Takes 2–5 minutes.

```bash
venv/Scripts/python.exe synthetic_data/generators/run_all.py
```

What this creates:
- 5,000 customers (enterprise / mid-market / SMB / startup)
- ~8,000 contracts
- ~7,500 subscriptions
- **50,000 invoices** (2022-01 to 2024-11)
- ~45,000 payments (with 30% late, 17% failed)
- ~5,000 refunds (147 marked as duplicates)

> This script is **idempotent** — safe to re-run, it checks for existing data first.

---

## Step 6 — Run the ML Pipeline (first time only)

Trains all models, scores all 50,000 invoices, and writes results to MySQL.  
Takes 3–8 minutes depending on your machine.

```bash
venv/Scripts/python.exe ml_pipeline/run_pipeline.py
```

What this does in sequence:
1. Loads invoice features from MySQL
2. Engineers 47 features per invoice
3. Splits into train (70%) / val (15%) / test (15%) by date
4. Trains **Isolation Forest** (unsupervised anomaly detection)
5. Trains **Logistic Regression** (baseline)
6. Trains **XGBoost** (main model, with early stopping on val set)
7. Evaluates all models on held-out test set, prints comparison table
8. Runs inference on ALL 50,000 invoices
9. Writes 50,000 rows to `anomaly_scores` table
10. Writes 50,000 rows to `risk_scores` table (with rank percentiles)
11. Saves trained models to `artifacts/models/`

After the pipeline, check DB state:
```bash
venv/Scripts/python.exe -c "
import django, os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings'); django.setup()
from apps.anomaly_detection.models import AnomalyScore
from apps.risk_scoring.models import RiskScore
print('AnomalyScore rows:', AnomalyScore.objects.count())  # expect 50000
print('RiskScore rows:', RiskScore.objects.count())         # expect 50000
"
```

---

## Step 7 — Run the Forecasting Pipeline (first time only)

Fits Prophet on historical revenue and forecasts 6 months forward.

```bash
venv/Scripts/python.exe -c "
import django, os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings'); django.setup()
from ml_pipeline.forecasting import run_forecasting_pipeline
run_forecasting_pipeline()
"
```

Writes ~82 rows to the `forecast_results` table.

---

## Step 8 — Run the Rule Engine (first time only)

Detects structural billing anomalies and fires rule alerts.

```bash
venv/Scripts/python.exe ml_pipeline/rule_engine.py
```

Writes ~13,000 rows to `rule_alerts` table covering:
- `MISSING_PAYMENT` — invoice issued/overdue with no completed payment
- `DUPLICATE_REFUND` — duplicate refunds flagged
- `ABNORMAL_DISCOUNT` — discount > 35% of invoice total
- `UNDERBILLING` — billed < 85% of subscription MRR
- `FAILED_PAYMENT_STREAK` — customer with 3+ consecutive failed payments
- `MISSING_RENEWAL` — invoice overdue past due date

Re-run with `--clear` to reset and regenerate:
```bash
venv/Scripts/python.exe ml_pipeline/rule_engine.py --clear
```

---

## Step 9 — Start the Django Server

```bash
venv/Scripts/python.exe manage.py runserver 8000
```

Open your browser at: **http://localhost:8000**

The dashboard loads all data from the REST APIs automatically.

---

## Running the Project Again (Subsequent Sessions)

When you close and reopen the project, you only need **one command**:

```bash
cd "d:/AI Revenue Leakage Detection Platform"
venv/Scripts/python.exe manage.py runserver 8000
```

All data is persisted in MySQL — no need to re-run the ML pipeline or synthetic data generators.

---

## Full Command Reference (all in one place)

```bash
# ── Navigate to project ──────────────────────────────────────────────
cd "d:/AI Revenue Leakage Detection Platform"

# ── First-time setup (run once, in order) ───────────────────────────
venv/Scripts/python.exe manage.py makemigrations
venv/Scripts/python.exe manage.py migrate
venv/Scripts/python.exe synthetic_data/generators/run_all.py
venv/Scripts/python.exe ml_pipeline/run_pipeline.py
venv/Scripts/python.exe -c "
import django,os; os.environ['DJANGO_SETTINGS_MODULE']='config.settings'; django.setup()
from ml_pipeline.forecasting import run_forecasting_pipeline; run_forecasting_pipeline()
"
venv/Scripts/python.exe ml_pipeline/rule_engine.py

# ── Every session (just this one command) ───────────────────────────
venv/Scripts/python.exe manage.py runserver 8000

# ── Optional: retrain ML models from scratch ────────────────────────
venv/Scripts/python.exe ml_pipeline/run_pipeline.py

# ── Optional: re-run rule engine (resets all alerts) ────────────────
venv/Scripts/python.exe ml_pipeline/rule_engine.py --clear

# ── Optional: skip training, use saved models ────────────────────────
venv/Scripts/python.exe ml_pipeline/run_pipeline.py --skip-train

# ── Optional: score without writing to DB ───────────────────────────
venv/Scripts/python.exe ml_pipeline/run_pipeline.py --no-db-write

# ── Django admin (create superuser first) ───────────────────────────
venv/Scripts/python.exe manage.py createsuperuser
# Then visit: http://localhost:8000/admin/

# ── Check system health ──────────────────────────────────────────────
venv/Scripts/python.exe manage.py check
```

---

## How Django Connects to MySQL

The database connection is configured in `config/settings.py`:

```python
DATABASES = {
    'default': {
        'ENGINE':   'django.db.backends.mysql',
        'NAME':     'revenue_leakage_db',
        'USER':     'root',
        'PASSWORD': 'AhzUttara@2025',
        'HOST':     'localhost',
        'PORT':     '3306',
        'OPTIONS':  {'charset': 'utf8mb4'},
    }
}
```

Django uses the `mysqlclient` library (a C extension) to talk to MySQL. This is faster than PyMySQL. The connection is established when Django starts — if MySQL isn't running, the server will fail immediately with `OperationalError: (2003, "Can't connect to MySQL server")`.

**Important escaping note:** In raw SQL queries, `DATE_FORMAT(col, '%Y-%m')` uses single `%` when the `cursor.execute()` call has no parameters. When parameters are present, use `%%Y-%%m` to avoid mysqlclient treating `%` as a Python format character.

---

## REST API Endpoints

All endpoints return JSON. Base URL: `http://localhost:8000/api/`

### Dashboard Endpoints
| Endpoint | Description |
|---|---|
| `GET /api/dashboard/kpis/` | Revenue, risk, and customer KPI summary |
| `GET /api/dashboard/revenue-trend/` | Monthly actuals + Prophet forecast |
| `GET /api/dashboard/top-risks/?limit=15` | Top N high-risk invoices |
| `GET /api/dashboard/risk-distribution/` | Risk severity breakdown |
| `GET /api/dashboard/leakage-by-rule/` | Leakage grouped by rule code |
| `GET /api/dashboard/anomaly-timeline/` | Monthly anomaly counts |
| `GET /api/dashboard/invoice/<id>/` | Full ML + rule detail for one invoice |

### KPI Detail Endpoints (used by clickable cards)
| Endpoint | Description |
|---|---|
| `GET /api/dashboard/kpi/revenue-breakdown/` | Monthly invoiced/collected/rate table |
| `GET /api/dashboard/kpi/outstanding/` | Top unpaid invoices |
| `GET /api/dashboard/kpi/overdue/` | All overdue invoices with days past due |
| `GET /api/dashboard/kpi/high-risk/` | Critical + High severity invoices |
| `GET /api/dashboard/kpi/anomalies/` | Isolation Forest anomaly list |
| `GET /api/dashboard/kpi/duplicate-refunds/` | All 147 duplicate refunds |

### Entity Endpoints (DRF ViewSets)
| Endpoint | Description |
|---|---|
| `GET /api/customers/` | Customer list with filtering |
| `GET /api/invoices/` | Invoice list with filtering |
| `GET /api/risk-scores/` | Risk score list |
| `GET /api/anomaly-scores/` | Anomaly score list |
| `GET /api/forecasts/` | Forecast results with `?is_anomalous=true&metric=revenue` |
| `GET /admin/` | Django admin interface |

---

## ML Pipeline Architecture

```
MySQL invoices table
        │
        ▼
data_loading.py          ← SQL query → pandas DataFrame (50,000 rows)
        │
        ▼
feature_engineering.py   ← 47 features: discount_ratio, payment_delay_ratio,
        │                   failed_payment_rate, billed_vs_expected_ratio, etc.
        ▼
   Time Split (70/15/15 by date)
   ┌──────────────────────┐
   │ Train │  Val  │ Test │
   │ 35k   │  7.5k │ 7.5k │
   └──────────────────────┘
        │
        ├── train_anomaly.py    → Isolation Forest (unsupervised, no labels)
        ├── train_baseline.py   → Logistic Regression (with leakage_label)
        └── train_xgboost.py   → XGBoost (main model, hypertuned on val set)
        │
        ▼
   evaluation.py           ← ROC-AUC, Precision, Recall, F1 on test set
        │
        ▼
   inference.py            ← Score all 50,000 invoices
        │
        ├── anomaly_scores table  (IF score + is_anomaly)
        └── risk_scores table     (XGB prob + ensemble prob + rank percentile)
```

The **leakage label** is synthetic: an invoice is marked as leakage if it has any of: `discount_ratio > 0.4`, `outstanding_ratio > 0.5`, `failed_payment_rate > 0.3`, or `duplicate_refund_count > 0`. This gives a ~28% positive rate, producing an imbalanced but realistic classification problem.

---

## Forecasting Architecture

```
Monthly aggregated revenue (from invoices table)
        │
        ▼
   Prophet model          ← Fits on historical months 2022-01 to 2024-05
        │                   Predicts 6 months forward (2024-06 to 2024-11)
        ▼
   Anomaly detection      ← Months where actual > forecast + 2σ  flagged
        │
        ▼
   forecast_results table ← period_month, forecasted_value, lower/upper bound,
                             is_anomalous, deviation_pct
```

Detected anomalies (2024-08: +67.6%, 2024-09: +21.6%, 2024-11: −32.2%) appear in the "Forecast Anomalies Detected" table on the dashboard.

---

## Dashboard Features

- **8 clickable KPI cards** — each opens a modal with full detail tables
- **Revenue vs Forecast chart** — dual-line (actual=scarlet, forecast=gold) with 90% CI band
- **Risk Severity donut** — live distribution of Critical/High/Medium/Low invoices
- **Top 15 High-Risk Invoices table** — click any row for full ML breakdown modal showing:
  - XGBoost probability, LR probability, ensemble score with progress bars
  - Isolation Forest anomaly flag and score
  - Top 3 risk-driving features
  - All rule alerts triggered for that invoice
- **Forecast Anomalies table** — months where revenue deviated from model prediction
- **Column tooltips** — hover the ⓘ icons in table headers for plain-English explanations

---

## Troubleshooting

**`django.db.utils.OperationalError: (2003, Can't connect to MySQL server)`**  
→ MySQL is not running. Start it: Windows Services → MySQL80 → Start

**`ModuleNotFoundError: No module named 'MySQLdb'`**  
→ `mysqlclient` not installed or C libs missing. Run: `pip install mysqlclient`

**`django.db.utils.ProgrammingError: Table doesn't exist`**  
→ Migrations not applied. Run: `venv/Scripts/python.exe manage.py migrate`

**Server returns old data after code changes**  
→ Django dev server with `--noreload` doesn't pick up changes. Kill the process (`Ctrl+C`) and restart.

**Chart shows `%Y-%m` as a label**  
→ Old server process still running. Use `netstat -ano | findstr :8000` to find PID and kill it, then restart.

**ML pipeline fails with `Prophet` import error**  
→ Prophet requires Python 3.11. Confirm: `venv/Scripts/python.exe --version` should show 3.11.x

**`bulk_create` fails with duplicate key error**  
→ Data already exists. The rule engine accepts `--clear`. The synthetic data generator is idempotent and checks for existing rows.
