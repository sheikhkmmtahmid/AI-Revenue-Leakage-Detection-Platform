from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from apps.anomaly_detection.models import AnomalyScore
from apps.anomaly_detection.serializers import AnomalyScoreSerializer


class AnomalyScoreViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = (
        AnomalyScore.objects
        .select_related("customer", "invoice")
        .order_by("score")   # most anomalous first (lowest score)
    )
    serializer_class = AnomalyScoreSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["is_anomaly", "model_version", "customer"]
    search_fields = ["customer__name"]
    ordering_fields = ["score", "scored_at"]
