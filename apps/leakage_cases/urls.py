from rest_framework.routers import DefaultRouter
from apps.leakage_cases.views import LeakageCaseViewSet, RuleAlertViewSet

router = DefaultRouter()
router.register(r"leakage-cases", LeakageCaseViewSet, basename="leakagecase")
router.register(r"rule-alerts", RuleAlertViewSet, basename="rulealert")
urlpatterns = router.urls
