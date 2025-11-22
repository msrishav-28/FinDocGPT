"""
Error handling service for standardized error responses and graceful degradation
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from ..models.errors import (
    ErrorResponse, ErrorDetail, ServiceError, ErrorCode,
    ValidationError, AuthenticationError, AuthorizationError,
    RateLimitError, DocumentError, AnalysisError, ExternalAPIError,
    DatabaseError
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling service"""
    
    def __init__(self):
        self.error_stats: Dict[str, int] = {}
        self.service_health: Dict[str, bool] = {
            "document_processor": True,
            "sentiment_analyzer": True,
            "anomaly_detector": True,
            "forecasting_engine": True,
            "investment_advisor": True,
            "external_apis": True,
            "database": True
        }
    
    def generate_request_id(self) -> str:
        """Generate unique request ID for error tracking"""
        return str(uuid.uuid4())[:8]
    
    def log_error(
        self, 
        error: Exception, 
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error with context information"""
        request_id = self.generate_request_id()
        
        log_data = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if request:
            log_data.update({
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            })
        
        if context:
            log_data["context"] = context
        
        logger.error(f"Error occurred: {log_data}")
        
        # Update error statistics
        error_type = type(error).__name__
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        
        return request_id
    
    def create_error_response(
        self,
        error: Exception,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorResponse:
        """Create standardized error response"""
        request_id = self.log_error(error, request, context)
        
        if isinstance(error, ServiceError):
            return ErrorResponse(
                error_code=error.error_code.value,
                message=error.message,
                details=error.details,
                context=error.context,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id
            )
        elif isinstance(error, HTTPException):
            # Map HTTP exceptions to our error codes
            error_code = self._map_http_status_to_error_code(error.status_code)
            return ErrorResponse(
                error_code=error_code.value,
                message=error.detail,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id
            )
        elif isinstance(error, PydanticValidationError):
            # Handle Pydantic validation errors
            details = []
            for error_detail in error.errors():
                field = ".".join(str(loc) for loc in error_detail["loc"])
                details.append(ErrorDetail(
                    field=field,
                    message=error_detail["msg"],
                    code=error_detail["type"]
                ))
            
            return ErrorResponse(
                error_code=ErrorCode.VALIDATION_ERROR.value,
                message="Validation failed",
                details=details,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id
            )
        else:
            # Generic error handling
            return ErrorResponse(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR.value,
                message="An unexpected error occurred",
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id
            )
    
    def _map_http_status_to_error_code(self, status_code: int) -> ErrorCode:
        """Map HTTP status codes to error codes"""
        mapping = {
            400: ErrorCode.INVALID_INPUT,
            401: ErrorCode.AUTHENTICATION_REQUIRED,
            403: ErrorCode.INSUFFICIENT_PERMISSIONS,
            404: ErrorCode.RECORD_NOT_FOUND,
            422: ErrorCode.VALIDATION_ERROR,
            429: ErrorCode.RATE_LIMIT_EXCEEDED,
            500: ErrorCode.INTERNAL_SERVER_ERROR,
            502: ErrorCode.EXTERNAL_API_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE,
            504: ErrorCode.TIMEOUT_ERROR
        }
        return mapping.get(status_code, ErrorCode.INTERNAL_SERVER_ERROR)
    
    def handle_service_degradation(self, service_name: str, error: Exception) -> Dict[str, Any]:
        """Handle service degradation and provide fallback responses"""
        self.service_health[service_name] = False
        
        fallback_responses = {
            "document_processor": {
                "message": "Document processing temporarily unavailable. Please try again later.",
                "fallback_action": "Use cached results or manual processing"
            },
            "sentiment_analyzer": {
                "message": "Sentiment analysis temporarily unavailable.",
                "fallback_action": "Using rule-based sentiment analysis",
                "sentiment_score": 0.0,
                "confidence": 0.1
            },
            "anomaly_detector": {
                "message": "Anomaly detection temporarily unavailable.",
                "fallback_action": "Using basic statistical checks",
                "anomalies": []
            },
            "forecasting_engine": {
                "message": "Forecasting models temporarily unavailable.",
                "fallback_action": "Using historical averages",
                "forecast_confidence": 0.1
            },
            "investment_advisor": {
                "message": "Investment advisory temporarily unavailable.",
                "fallback_action": "Using conservative recommendations",
                "recommendation": "HOLD",
                "confidence": 0.1
            },
            "external_apis": {
                "message": "External data sources temporarily unavailable.",
                "fallback_action": "Using cached data",
                "data_freshness": "stale"
            },
            "database": {
                "message": "Database temporarily unavailable.",
                "fallback_action": "Using in-memory cache",
                "data_persistence": False
            }
        }
        
        return fallback_responses.get(service_name, {
            "message": f"{service_name} temporarily unavailable",
            "fallback_action": "Limited functionality available"
        })
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        return self.service_health.get(service_name, True)
    
    def restore_service_health(self, service_name: str):
        """Mark service as healthy again"""
        self.service_health[service_name] = True
        logger.info(f"Service {service_name} restored to healthy state")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "error_counts": self.error_stats.copy(),
            "service_health": self.service_health.copy(),
            "total_errors": sum(self.error_stats.values())
        }
    
    def create_validation_error(self, field: str, message: str, value: Any = None) -> ValidationError:
        """Create a validation error for a specific field"""
        details = [ErrorDetail(field=field, message=message)]
        context = {"invalid_value": value} if value is not None else None
        
        return ValidationError(
            message=f"Validation failed for field: {field}",
            details=details,
            error_code=ErrorCode.VALIDATION_ERROR
        )
    
    def create_business_logic_error(
        self, 
        message: str, 
        error_code: ErrorCode,
        context: Optional[Dict[str, Any]] = None
    ) -> ServiceError:
        """Create a business logic error"""
        return ServiceError(
            message=message,
            error_code=error_code,
            context=context,
            status_code=400
        )


# Global error handler instance
error_handler = ErrorHandler()