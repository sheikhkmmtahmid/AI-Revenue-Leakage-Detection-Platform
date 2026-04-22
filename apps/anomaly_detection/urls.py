from rest_framework.routers import DefaultRouter
from apps.anomaly_detection.views import AnomalyScoreViewSet

router = DefaultRouter()
router.register(r"anomaly-scores", AnomalyScoreViewSet, basename="anomalyscore")
urlpatterns = router.urls
