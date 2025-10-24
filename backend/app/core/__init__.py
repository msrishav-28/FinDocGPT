"""
Core functionality for the Financial Intelligence System.

This module contains core configuration, security, and utility components
that are used throughout the application.
"""

from .config import settings, get_settings
from .security import get_current_user, create_access_token, verify_token
from .exceptions import (
    FinancialIntelligenceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ExternalAPIError
)

__all__ = [
    "settings",
    "get_settings",
    "get_current_user",
    "create_access_token",
    "verify_token",
    "FinancialIntelligenceException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ExternalAPIError"
]