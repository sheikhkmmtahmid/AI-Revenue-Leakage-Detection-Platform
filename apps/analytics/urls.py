from django.urls import path
from apps.analytics import views

urlpatterns = [
    path("dashboard/kpis/",                          views.kpis,             name="dashboard-kpis"),
    path("dashboard/revenue-trend/",                 views.revenue_trend,    name="dashboard-revenue-trend"),
    path("dashboard/top-risks/",                     views.top_risks,        name="dashboard-top-risks"),
    path("dashboard/risk-distribution/",             views.risk_distribution,name="dashboard-risk-dist"),
    path("dashboard/leakage-by-rule/",               views.leakage_by_rule,  name="dashboard-leakage-rule"),
    path("dashboard/anomaly-timeline/",              views.anomaly_timeline, name="dashboard-anomaly-timeline"),
    path("dashboard/invoice/<int:invoice_id>/",      views.invoice_detail,          name="dashboard-invoice-detail"),
    path("dashboard/kpi/revenue-breakdown/",          views.kpi_revenue_breakdown,   name="kpi-revenue-breakdown"),
    path("dashboard/kpi/outstanding/",                views.kpi_outstanding,         name="kpi-outstanding"),
    path("dashboard/kpi/overdue/",                    views.kpi_overdue,             name="kpi-overdue"),
    path("dashboard/kpi/high-risk/",                  views.kpi_high_risk_detail,    name="kpi-high-risk"),
    path("dashboard/kpi/anomalies/",                  views.kpi_anomalies_detail,    name="kpi-anomalies"),
    path("dashboard/kpi/duplicate-refunds/",          views.kpi_duplicate_refunds_detail, name="kpi-duplicate-refunds"),
]
