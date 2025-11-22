"""
Error handling middleware for centralized error processing
"""

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError as PydanticValidationError
from ..models.errors import ServiceError
from ..services.error_handler import error_handler

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling and response formatting"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        
        except ServiceError as e:
            # Handle our custom service errors
            error_response = error_handler.create_error_response(e, request)
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.dict()
            )
        
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            error_response = error_handler.create_error_response(e, request)
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.dict()
            )
        
        except PydanticValidationError as e:
            # Handle Pydantic validation errors
            error_response = error_handler.create_error_response(e, request)
            return JSONResponse(
                status_code=422,
                content=error_response.dict()
            )
        
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error occurred: {str(e)}")
            error_response = error_handler.create_error_response(e, request)
            return JSONResponse(
                status_code=500,
                content=error_response.dict()
            )