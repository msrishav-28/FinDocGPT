"""
Custom exceptions for the Financial Intelligence System.

This module defines application-specific exceptions that provide
clear error handling and messaging throughout the system.
"""

from typing import Optional, Dict, Any


class FinancialIntelligenceException(Exception):
    """Base exception for Financial Intelligence System."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(FinancialIntelligenceException):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationError(FinancialIntelligenceException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            **kwargs
        )


class AuthorizationError(FinancialIntelligenceException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            **kwargs
        )


class NotFoundError(FinancialIntelligenceException):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        
        super().__init__(
            message=message,
            error_code="NOT_FOUND_ERROR",
            details=details
        )


class ExternalAPIError(FinancialIntelligenceException):
    """Raised when external API calls fail."""
    
    def __init__(
        self,
        message: str = "External API error",
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if api_name:
            details['api_name'] = api_name
        if status_code:
            details['status_code'] = status_code
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_API_ERROR",
            details=details
        )


class DatabaseError(FinancialIntelligenceException):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details
        )


class ProcessingError(FinancialIntelligenceException):
    """Raised when document or data processing fails."""
    
    def __init__(
        self,
        message: str = "Processing failed",
        processing_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if processing_type:
            details['processing_type'] = processing_type
        
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details=details
        )


class ConfigurationError(FinancialIntelligenceException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )