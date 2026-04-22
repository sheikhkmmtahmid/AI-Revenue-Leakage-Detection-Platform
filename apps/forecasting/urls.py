from rest_framework.routers import DefaultRouter
from apps.forecasting.views import ForecastResultViewSet

router = DefaultRouter()
router.register(r"forecasts", ForecastResultViewSet, basename="forecast")
urlpatterns = router.urls
