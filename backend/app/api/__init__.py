"""
API package for the Financial Intelligence System.

This package contains all API endpoints organized by domain and version.
"""

from .v1 import api_router

__all__ = ["api_router"]