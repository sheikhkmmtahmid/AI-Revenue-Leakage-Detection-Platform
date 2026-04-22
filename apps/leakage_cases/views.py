from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from apps.leakage_cases.models import LeakageCase, RuleAlert
from apps.leakage_cases.serializers import LeakageCaseSerializer, RuleAlertSerializer


class LeakageCaseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = (
        LeakageCase.objects
        .select_related("customer", "assigned_to")
        .order_by("-created_at")
    )
    serializer_class = LeakageCaseSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["status", "priority", "customer"]
    search_fields = ["case_number", "title", "customer__name"]
    ordering_fields = ["created_at", "estimated_leakage_amount", "priority"]


class RuleAlertViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = (
        RuleAlert.objects
        .select_related("customer", "invoice")
        .order_by("-detected_at")
    )
    serializer_class = RuleAlertSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["rule_code", "severity", "is_resolved", "customer"]
    search_fields = ["customer__name", "description"]
    ordering_fields = ["detected_at", "leakage_amount", "severity"]
