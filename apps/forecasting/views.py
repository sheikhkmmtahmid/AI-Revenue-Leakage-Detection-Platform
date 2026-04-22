from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from apps.forecasting.models import ForecastResult
from apps.forecasting.serializers import ForecastResultSerializer


class ForecastResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ForecastResult.objects.order_by("metric", "period_month")
    serializer_class = ForecastResultSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["metric", "model_name", "model_version", "is_anomalous"]
    ordering_fields = ["period_month", "deviation_pct"]
