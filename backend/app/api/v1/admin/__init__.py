"""
Administration API endpoints.

This module contains all administration-related API endpoints including
audit logs, compliance, cache management, database optimization, and background tasks.
"""

from fastapi import APIRouter
from ....routes.audit import router as audit_routes
from ....routes.compliance import router as compliance_routes
from ....routes.cache_routes import router as cache_routes
from ....routes.database_optimization_routes import router as db_optimization_routes
from ....routes.background_tasks import router as background_tasks_routes

router = APIRouter()

# Include administration-related routes
router.include_router(audit_routes)
router.include_router(compliance_routes)
router.include_router(cache_routes)
router.include_router(db_optimization_routes)
router.include_router(background_tasks_routes)

__all__ = ["router"]