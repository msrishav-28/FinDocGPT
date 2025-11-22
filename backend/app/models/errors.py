"""
Error models and standardized error responses
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes"""
    # Authentication errors (1000-1099)
    AUTHENTICATION_REQUIRED = "AUTH_001"
    INVALID_CREDENTIALS = "AUTH_002"
    TOKEN_EXPIRED = "AUTH_003"
    INVALID_TOKEN = "AUTH_004"
    INSUFFICIENT_PERMISSIONS = "AUTH_005"
    
    # Validation errors (1100-1199)
    VALIDATION_ERROR = "VAL_001"
    INVALID_INPUT = "VAL_002"
    MISSING_REQUIRED_FIELD = "VAL_003"
    INVALID_FORMAT = "VAL_004"
    VALUE_OUT_OF_RANGE = "VAL_005"
    
    # Rate limiting errors (1200-1299)
    RATE_LIMIT_EXCEEDED = "RATE_001"
    QUOTA_EXCEEDED = "RATE_002"
    
    # Document processing errors (2000-2099)
    DOCUMENT_NOT_FOUND = "DOC_001"
    DOCUMENT_PROCESSING_FAILED = "DOC_002"
    UNSUPPORTED_DOCUMENT_FORMAT = "DOC_003"
    DOCUMENT_TOO_LARGE = "DOC_004"
    DOCUMENT_UPLOAD_FAILED = "DOC_005"
    
    # Analysis errors (2100-2199)
    ANALYSIS_FAILED = "ANA_001"
    INSUFFICIENT_DATA = "ANA_002"
    MODEL_UNAVAILABLE = "ANA_003"
    ANALYSIS_TIMEOUT = "ANA_004"
    
    # External API errors (2200-2299)
    EXTERNAL_API_ERROR = "EXT_001"
    EXTERNAL_API_UNAVAILABLE = "EXT_002"
    EXTERNAL_API_RATE_LIMITED = "EXT_003"
    INVALID_API_KEY = "EXT_004"
    
    # Database errors (3000-3099)
    DATABASE_ERROR = "DB_001"
    DATABASE_CONNECTION_FAILED = "DB_002"
    RECORD_NOT_FOUND = "DB_003"
    DUPLICATE_RECORD = "DB_004"
    
    # System errors (9000-9099)
    INTERNAL_SERVER_ERROR = "SYS_001"
    SERVICE_UNAVAILABLE = "SYS_002"
    TIMEOUT_ERROR = "SYS_003"
    CONFIGURATION_ERROR = "SYS_004"


class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error: bool = True
    error_code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    request_id: Optional[str] = None


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific details"""
    error_code: str = ErrorCode.VALIDATION_ERROR
    validation_errors: List[ErrorDetail]


class ServiceError(Exception):
    """Base service error with structured information"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[List[ErrorDetail]] = None,
        context: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or []
        self.context = context or {}
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(ServiceError):
    """Authentication-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.AUTHENTICATION_REQUIRED):
        super().__init__(message, error_code, status_code=401)


class AuthorizationError(ServiceError):
    """Authorization-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INSUFFICIENT_PERMISSIONS):
        super().__init__(message, error_code, status_code=403)


class ValidationError(ServiceError):
    """Validation-related errors"""
    
    def __init__(
        self, 
        message: str, 
        details: List[ErrorDetail],
        error_code: ErrorCode = ErrorCode.VALIDATION_ERROR
    ):
        super().__init__(message, error_code, details=details, status_code=422)


class RateLimitError(ServiceError):
    """Rate limiting errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(
            message, 
            ErrorCode.RATE_LIMIT_EXCEEDED, 
            context=context,
            status_code=429
        )


class DocumentError(ServiceError):
    """Document processing errors"""
    
    def __init__(self, message: str, error_code: ErrorCode, context: Dict[str, Any] = None):
        super().__init__(message, error_code, context=context, status_code=400)


class AnalysisError(ServiceError):
    """Analysis-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode, context: Dict[str, Any] = None):
        super().__init__(message, error_code, context=context, status_code=500)


class ExternalAPIError(ServiceError):
    """External API errors"""
    
    def __init__(self, message: str, error_code: ErrorCode, context: Dict[str, Any] = None):
        super().__init__(message, error_code, context=context, status_code=502)


class DatabaseError(ServiceError):
    """Database-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode, context: Dict[str, Any] = None):
        super().__init__(message, error_code, context=context, status_code=500)