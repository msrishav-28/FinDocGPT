"""
Input validation service for comprehensive request validation
"""

import re
from typing import Any, List, Optional, Dict, Union
from datetime import datetime, date
from pydantic import BaseModel, validator, ValidationError as PydanticValidationError
from ..models.errors import ValidationError, ErrorDetail, ErrorCode
from ..services.error_handler import error_handler


class ValidationService:
    """Service for comprehensive input validation"""
    
    def __init__(self):
        self.validation_rules = {
            "ticker": {
                "pattern": r"^[A-Z]{1,5}$",
                "max_length": 5,
                "description": "Stock ticker symbol (1-5 uppercase letters)"
            },
            "email": {
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "max_length": 254,
                "description": "Valid email address"
            },
            "username": {
                "pattern": r"^[a-zA-Z0-9_]{3,30}$",
                "min_length": 3,
                "max_length": 30,
                "description": "Username (3-30 alphanumeric characters and underscores)"
            },
            "password": {
                "min_length": 8,
                "max_length": 128,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_digit": True,
                "description": "Password (8-128 characters with uppercase, lowercase, and digit)"
            },
            "company_name": {
                "min_length": 1,
                "max_length": 100,
                "description": "Company name (1-100 characters)"
            },
            "document_type": {
                "allowed_values": [
                    "earnings_report", "sec_filing", "press_release", 
                    "analyst_report", "financial_statement", "other"
                ],
                "description": "Document type"
            },
            "file_size": {
                "max_size": 50 * 1024 * 1024,  # 50MB
                "description": "File size limit"
            },
            "question": {
                "min_length": 5,
                "max_length": 1000,
                "description": "Question text (5-1000 characters)"
            },
            "forecast_horizon": {
                "allowed_values": [1, 3, 6, 12, 24],
                "description": "Forecast horizon in months"
            },
            "confidence_threshold": {
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Confidence threshold (0.0-1.0)"
            }
        }
    
    def validate_field(self, field_name: str, value: Any, rule_name: str) -> List[ErrorDetail]:
        """Validate a single field against a specific rule"""
        errors = []
        
        if rule_name not in self.validation_rules:
            return errors
        
        rule = self.validation_rules[rule_name]
        
        # Check if value is None or empty when required
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(ErrorDetail(
                field=field_name,
                message=f"{field_name} is required",
                code="required"
            ))
            return errors
        
        # String validations
        if isinstance(value, str):
            # Pattern validation
            if "pattern" in rule:
                if not re.match(rule["pattern"], value):
                    errors.append(ErrorDetail(
                        field=field_name,
                        message=f"{field_name} format is invalid. {rule['description']}",
                        code="invalid_format"
                    ))
            
            # Length validations
            if "min_length" in rule and len(value) < rule["min_length"]:
                errors.append(ErrorDetail(
                    field=field_name,
                    message=f"{field_name} must be at least {rule['min_length']} characters long",
                    code="too_short"
                ))
            
            if "max_length" in rule and len(value) > rule["max_length"]:
                errors.append(ErrorDetail(
                    field=field_name,
                    message=f"{field_name} must be at most {rule['max_length']} characters long",
                    code="too_long"
                ))
            
            # Password specific validations
            if rule_name == "password":
                if rule.get("require_uppercase") and not re.search(r"[A-Z]", value):
                    errors.append(ErrorDetail(
                        field=field_name,
                        message="Password must contain at least one uppercase letter",
                        code="missing_uppercase"
                    ))
                
                if rule.get("require_lowercase") and not re.search(r"[a-z]", value):
                    errors.append(ErrorDetail(
                        field=field_name,
                        message="Password must contain at least one lowercase letter",
                        code="missing_lowercase"
                    ))
                
                if rule.get("require_digit") and not re.search(r"\d", value):
                    errors.append(ErrorDetail(
                        field=field_name,
                        message="Password must contain at least one digit",
                        code="missing_digit"
                    ))
        
        # Numeric validations
        if isinstance(value, (int, float)):
            if "min_value" in rule and value < rule["min_value"]:
                errors.append(ErrorDetail(
                    field=field_name,
                    message=f"{field_name} must be at least {rule['min_value']}",
                    code="too_small"
                ))
            
            if "max_value" in rule and value > rule["max_value"]:
                errors.append(ErrorDetail(
                    field=field_name,
                    message=f"{field_name} must be at most {rule['max_value']}",
                    code="too_large"
                ))
        
        # File size validation
        if rule_name == "file_size" and isinstance(value, int):
            if value > rule["max_size"]:
                errors.append(ErrorDetail(
                    field=field_name,
                    message=f"File size must be less than {rule['max_size'] // (1024*1024)}MB",
                    code="file_too_large"
                ))
        
        # Allowed values validation
        if "allowed_values" in rule and value not in rule["allowed_values"]:
            errors.append(ErrorDetail(
                field=field_name,
                message=f"{field_name} must be one of: {', '.join(map(str, rule['allowed_values']))}",
                code="invalid_choice"
            ))
        
        return errors
    
    def validate_ticker(self, ticker: str) -> List[ErrorDetail]:
        """Validate stock ticker symbol"""
        return self.validate_field("ticker", ticker, "ticker")
    
    def validate_email(self, email: str) -> List[ErrorDetail]:
        """Validate email address"""
        return self.validate_field("email", email, "email")
    
    def validate_username(self, username: str) -> List[ErrorDetail]:
        """Validate username"""
        return self.validate_field("username", username, "username")
    
    def validate_password(self, password: str) -> List[ErrorDetail]:
        """Validate password"""
        return self.validate_field("password", password, "password")
    
    def validate_file_upload(self, filename: str, file_size: int, content_type: str) -> List[ErrorDetail]:
        """Validate file upload parameters"""
        errors = []
        
        # Validate file size
        errors.extend(self.validate_field("file_size", file_size, "file_size"))
        
        # Validate file extension
        allowed_extensions = [".pdf", ".txt", ".html", ".json", ".csv", ".xlsx"]
        file_extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
        
        if file_extension not in allowed_extensions:
            errors.append(ErrorDetail(
                field="filename",
                message=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}",
                code="unsupported_file_type"
            ))
        
        # Validate content type
        allowed_content_types = [
            "application/pdf", "text/plain", "text/html", "application/json",
            "text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ]
        
        if content_type not in allowed_content_types:
            errors.append(ErrorDetail(
                field="content_type",
                message=f"Content type not supported: {content_type}",
                code="unsupported_content_type"
            ))
        
        return errors
    
    def validate_date_range(self, start_date: Union[str, date], end_date: Union[str, date]) -> List[ErrorDetail]:
        """Validate date range"""
        errors = []
        
        try:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00")).date()
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
            
            if start_date > end_date:
                errors.append(ErrorDetail(
                    field="date_range",
                    message="Start date must be before end date",
                    code="invalid_date_range"
                ))
            
            # Check if date range is reasonable (not more than 10 years)
            if (end_date - start_date).days > 3650:
                errors.append(ErrorDetail(
                    field="date_range",
                    message="Date range cannot exceed 10 years",
                    code="date_range_too_large"
                ))
        
        except (ValueError, TypeError) as e:
            errors.append(ErrorDetail(
                field="date_format",
                message="Invalid date format. Use ISO format (YYYY-MM-DD)",
                code="invalid_date_format"
            ))
        
        return errors
    
    def validate_pagination(self, page: int, page_size: int) -> List[ErrorDetail]:
        """Validate pagination parameters"""
        errors = []
        
        if page < 1:
            errors.append(ErrorDetail(
                field="page",
                message="Page number must be at least 1",
                code="invalid_page"
            ))
        
        if page_size < 1 or page_size > 100:
            errors.append(ErrorDetail(
                field="page_size",
                message="Page size must be between 1 and 100",
                code="invalid_page_size"
            ))
        
        return errors
    
    def validate_model_input(self, model: BaseModel) -> Optional[ValidationError]:
        """Validate Pydantic model and convert to our error format"""
        try:
            # The model should already be validated by Pydantic
            return None
        except PydanticValidationError as e:
            details = []
            for error_detail in e.errors():
                field = ".".join(str(loc) for loc in error_detail["loc"])
                details.append(ErrorDetail(
                    field=field,
                    message=error_detail["msg"],
                    code=error_detail["type"]
                ))
            
            return ValidationError(
                message="Model validation failed",
                details=details
            )
    
    def create_validation_error(self, errors: List[ErrorDetail]) -> ValidationError:
        """Create a validation error from a list of error details"""
        if not errors:
            return None
        
        return ValidationError(
            message="Validation failed",
            details=errors
        )


# Global validation service instance
validation_service = ValidationService()