"""
API v1 endpoints for the Financial Intelligence System.

This module aggregates all v1 API routes organized by domain.
"""

from fastapi import APIRouter

from .auth import router as auth_router
from .documents import router as documents_router
from .analytics import router as analytics_router
from .monitoring import router as monitoring_router
from .admin import router as admin_router

api_router = APIRouter(prefix="/v1")

# Include domain-specific routers
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
api_router.include_router(admin_router, prefix="/admin", tags=["Administration"])

__all__ = ["api_router"]