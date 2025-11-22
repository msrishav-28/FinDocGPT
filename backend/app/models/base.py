"""
Base models and mixins for the Financial Intelligence System
"""

from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4


class BaseModel(PydanticBaseModel):
    """Base model with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp tracking"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class UUIDMixin(BaseModel):
    """Mixin for models that use UUID as primary key"""
    id: UUID = Field(default_factory=uuid4)


class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


class TimeRange(BaseModel):
    """Time range specification"""
    start_date: datetime
    end_date: datetime
    
    def validate_range(self) -> bool:
        """Validate that start_date is before end_date"""
        return self.start_date < self.end_date


class ErrorResponse(BaseModel):
    """Standard error response format"""
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseModel):
    """Standard success response format"""
    message: str
    data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)