from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from apps.invoices.models import Invoice
from apps.invoices.serializers import InvoiceSerializer


class InvoiceViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = (
        Invoice.objects
        .filter(is_deleted=False)
        .select_related("customer")
        .prefetch_related("risk_scores")
        .order_by("-issue_date")
    )
    serializer_class = InvoiceSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["status", "customer", "customer__segment"]
    search_fields = ["invoice_number", "customer__name"]
    ordering_fields = ["issue_date", "due_date", "total_amount", "outstanding_amount"]
