from rest_framework.routers import DefaultRouter
from apps.risk_scoring.views import RiskScoreViewSet

router = DefaultRouter()
router.register(r"risk-scores", RiskScoreViewSet, basename="riskscore")
urlpatterns = router.urls
