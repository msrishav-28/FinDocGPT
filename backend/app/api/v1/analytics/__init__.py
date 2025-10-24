"""
Analytics API endpoints.

This module contains all analytics-related API endpoints including
sentiment analysis, anomaly detection, forecasting, and investment advisory.
"""

from fastapi import APIRouter
from ....routes.sentiment_routes import router as sentiment_routes
from ....routes.anomaly_routes import router as anomaly_routes
from ....routes.ensemble_forecast_routes import router as forecast_routes
from ....routes.investment_advisory import router as investment_routes
from ....routes.market_data_routes import router as market_data_routes
from ....routes.explainability import router as explainability_routes

router = APIRouter()

# Include analytics-related routes
router.include_router(sentiment_routes)
router.include_router(anomaly_routes)
router.include_router(forecast_routes)
router.include_router(investment_routes)
router.include_router(market_data_routes)
router.include_router(explainability_routes)

__all__ = ["router"]