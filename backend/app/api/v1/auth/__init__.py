"""
Authentication API endpoints.

This module contains all authentication-related API endpoints including
login, logout, token management, and user authentication.
"""

from fastapi import APIRouter
from ....routes.auth import router as auth_routes

router = APIRouter()

# Include existing auth routes
router.include_router(auth_routes)

__all__ = ["router"]