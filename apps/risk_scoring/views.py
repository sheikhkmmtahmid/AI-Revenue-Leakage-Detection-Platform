from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from apps.risk_scoring.models import RiskScore
from apps.risk_scoring.serializers import RiskScoreSerializer


class RiskScoreViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = (
        RiskScore.objects
        .select_related("customer", "invoice")
        .order_by("-leakage_probability")
    )
    serializer_class = RiskScoreSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["risk_severity", "model_name", "model_version", "customer"]
    search_fields = ["customer__name"]
    ordering_fields = ["leakage_probability", "rank_percentile", "scored_at"]
