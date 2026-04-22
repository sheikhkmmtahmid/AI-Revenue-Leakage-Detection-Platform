from rest_framework.routers import DefaultRouter
from apps.invoices.views import InvoiceViewSet

router = DefaultRouter()
router.register(r"invoices", InvoiceViewSet, basename="invoice")
urlpatterns = router.urls
