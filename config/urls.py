from django.contrib import admin
from django.urls import path, include
from apps.analytics.views import dashboard

urlpatterns = [
    # Dashboard HTML
    path("", dashboard, name="dashboard"),

    # Django admin
    path("admin/", admin.site.urls),

    # REST API — entity endpoints
    path("api/", include("apps.customers.urls")),
    path("api/", include("apps.invoices.urls")),
    path("api/", include("apps.leakage_cases.urls")),
    path("api/", include("apps.risk_scoring.urls")),
    path("api/", include("apps.anomaly_detection.urls")),
    path("api/", include("apps.forecasting.urls")),

    # REST API — dashboard endpoints
    path("api/", include("apps.analytics.urls")),
]
