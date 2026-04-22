from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from apps.customers.models import Customer
from apps.customers.serializers import CustomerListSerializer, CustomerDetailSerializer


class CustomerViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Customer.objects.filter(is_deleted=False).order_by("-acquisition_date")
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["segment", "status", "risk_tier", "country", "industry"]
    search_fields = ["name", "email", "external_id"]
    ordering_fields = ["acquisition_date", "name", "credit_limit"]

    def get_serializer_class(self):
        if self.action == "retrieve":
            return CustomerDetailSerializer
        return CustomerListSerializer
