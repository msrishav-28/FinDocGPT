"""
Validation decorators and utilities for request validation
"""

from functools import wraps
from typing import Callable, Any, List, Optional
from fastapi import Request, HTTPException, status
from ..models.errors import ValidationError, ErrorDetail
from ..services.validation_service import validation_service


def validate_ticker(func: Callable) -> Callable:
    """Decorator to validate ticker parameter"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ticker = kwargs.get('ticker')
        if ticker:
            errors = validation_service.validate_ticker(ticker)
            if errors:
                raise ValidationError("Invalid ticker", errors)
        return await func(*args, **kwargs)
    return wrapper


def validate_file_upload(func: Callable) -> Callable:
    """Decorator to validate file upload parameters"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        file = kwargs.get('file')
        if file:
            errors = validation_service.validate_file_upload(
                file.filename, 
                file.size if hasattr(file, 'size') else 0,
                file.content_type
            )
            if errors:
                raise ValidationError("Invalid file upload", errors)
        return await func(*args, **kwargs)
    return wrapper


def validate_pagination(func: Callable) -> Callable:
    """Decorator to validate pagination parameters"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        page = kwargs.get('page', 1)
        page_size = kwargs.get('page_size', 20)
        
        errors = validation_service.validate_pagination(page, page_size)
        if errors:
            raise ValidationError("Invalid pagination parameters", errors)
        
        return await func(*args, **kwargs)
    return wrapper


def validate_date_range(func: Callable) -> Callable:
    """Decorator to validate date range parameters"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        
        if start_date and end_date:
            errors = validation_service.validate_date_range(start_date, end_date)
            if errors:
                raise ValidationError("Invalid date range", errors)
        
        return await func(*args, **kwargs)
    return wrapper


def validate_required_fields(required_fields: List[str]):
    """Decorator factory to validate required fields"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            errors = []
            
            for field in required_fields:
                value = kwargs.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    errors.append(ErrorDetail(
                        field=field,
                        message=f"{field} is required",
                        code="required"
                    ))
            
            if errors:
                raise ValidationError("Missing required fields", errors)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_business_rules(func: Callable) -> Callable:
    """Decorator to validate business rules"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Add custom business rule validations here
        # This is a placeholder for business-specific validation logic
        return await func(*args, **kwargs)
    return wrapper


class RequestValidator:
    """Utility class for request validation"""
    
    @staticmethod
    def validate_json_payload(payload: dict, required_fields: List[str]) -> List[ErrorDetail]:
        """Validate JSON payload for required fields"""
        errors = []
        
        for field in required_fields:
            if field not in payload:
                errors.append(ErrorDetail(
                    field=field,
                    message=f"Field '{field}' is required",
                    code="required"
                ))
            elif payload[field] is None:
                errors.append(ErrorDetail(
                    field=field,
                    message=f"Field '{field}' cannot be null",
                    code="null_value"
                ))
            elif isinstance(payload[field], str) and not payload[field].strip():
                errors.append(ErrorDetail(
                    field=field,
                    message=f"Field '{field}' cannot be empty",
                    code="empty_value"
                ))
        
        return errors
    
    @staticmethod
    def validate_query_params(request: Request, allowed_params: List[str]) -> List[ErrorDetail]:
        """Validate query parameters"""
        errors = []
        
        for param in request.query_params:
            if param not in allowed_params:
                errors.append(ErrorDetail(
                    field=param,
                    message=f"Unknown query parameter: {param}",
                    code="unknown_parameter"
                ))
        
        return errors
    
    @staticmethod
    def sanitize_input(value: str, max_length: Optional[int] = None) -> str:
        """Sanitize input string"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially dangerous characters
        sanitized = value.strip()
        
        # Truncate if necessary
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    @staticmethod
    def validate_content_type(request: Request, allowed_types: List[str]) -> Optional[ErrorDetail]:
        """Validate request content type"""
        content_type = request.headers.get("content-type", "").split(";")[0]
        
        if content_type not in allowed_types:
            return ErrorDetail(
                field="content_type",
                message=f"Unsupported content type: {content_type}. Allowed: {', '.join(allowed_types)}",
                code="unsupported_content_type"
            )
        
        return None