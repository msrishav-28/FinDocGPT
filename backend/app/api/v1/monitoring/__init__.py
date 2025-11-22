"""
Monitoring API endpoints.

This module contains all monitoring-related API endpoints including
health checks, metrics, alerts, and system monitoring.
"""

from fastapi import APIRouter
from ....routes.health_routes import router as health_routes
from ....routes.monitoring import router as monitoring_routes
from ....routes.alert_routes import router as alert_routes
from ....routes.websocket_routes import router as websocket_routes

router = APIRouter()

# Include monitoring-related routes
router.include_router(health_routes)
router.include_router(monitoring_routes)
router.include_router(alert_routes)
router.include_router(websocket_routes)

__all__ = ["router"]