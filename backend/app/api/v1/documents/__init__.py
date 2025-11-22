"""
Document processing API endpoints.

This module contains all document-related API endpoints including
document upload, processing, analysis, and retrieval.
"""

from fastapi import APIRouter
from ....routes.document_routes import router as document_routes

router = APIRouter()

# Include existing document routes
router.include_router(document_routes)

__all__ = ["router"]