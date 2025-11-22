"""
User and authentication data models
"""

from pydantic import Field, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin


class UserRole(str, Enum):
    """User role types"""
    ADMIN = "admin"
    ANALYST = "analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    VIEWER = "viewer"


class SubscriptionTier(str, Enum):
    """Subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class User(UUIDMixin, TimestampMixin):
    """User account model"""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    
    # Authentication
    hashed_password: str = Field(..., description="Hashed password")
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    last_login: Optional[datetime] = None
    
    # Authorization
    role: UserRole = Field(default=UserRole.VIEWER)
    permissions: List[str] = Field(default_factory=list)
    
    # Subscription
    subscription_tier: SubscriptionTier = Field(default=SubscriptionTier.FREE)
    subscription_expires: Optional[datetime] = None
    
    # Profile
    timezone: str = Field(default="UTC")
    language: str = Field(default="en")
    avatar_url: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v.lower()


class UserPreferences(UUIDMixin, TimestampMixin):
    """User preferences and settings"""
    user_id: UUID = Field(..., description="Reference to user")
    
    # Dashboard preferences
    default_dashboard: Optional[str] = None
    dashboard_layout: Dict[str, Any] = Field(default_factory=dict)
    chart_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Notification preferences
    email_notifications: bool = Field(default=True)
    push_notifications: bool = Field(default=True)
    alert_frequency: str = Field(default="immediate", description="immediate, daily, weekly")
    
    # Analysis preferences
    default_time_horizon: str = Field(default="medium_term")
    risk_tolerance: str = Field(default="medium")
    preferred_analysis_depth: str = Field(default="standard")
    
    # Data preferences
    preferred_data_sources: List[str] = Field(default_factory=list)
    currency: str = Field(default="USD")
    date_format: str = Field(default="YYYY-MM-DD")
    
    # Privacy settings
    data_sharing_consent: bool = Field(default=False)
    analytics_tracking: bool = Field(default=True)
    marketing_consent: bool = Field(default=False)


class Watchlist(UUIDMixin, TimestampMixin):
    """User watchlist for tracking stocks"""
    user_id: UUID = Field(..., description="Reference to user")
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    
    # Watchlist content
    tickers: List[str] = Field(default_factory=list, description="List of stock tickers")
    is_default: bool = Field(default=False)
    is_public: bool = Field(default=False)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    color: Optional[str] = Field(None, description="UI color for the watchlist")
    sort_order: int = Field(default=0)
    
    @validator('tickers')
    def validate_tickers(cls, v):
        return [ticker.upper().strip() for ticker in v if ticker.strip()]


class UserSession(UUIDMixin, TimestampMixin):
    """User session tracking"""
    user_id: UUID = Field(..., description="Reference to user")
    session_token: str = Field(..., description="Session token")
    refresh_token: Optional[str] = None
    
    # Session details
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_type: Optional[str] = None
    
    # Session lifecycle
    expires_at: datetime
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    # Security
    login_method: str = Field(default="password", description="password, oauth, sso")
    is_suspicious: bool = Field(default=False)
    failed_attempts: int = Field(default=0)


class UserActivity(UUIDMixin, TimestampMixin):
    """User activity logging"""
    user_id: UUID = Field(..., description="Reference to user")
    activity_type: str = Field(..., description="Type of activity")
    description: str = Field(..., description="Activity description")
    
    # Activity context
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Activity data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = Field(default=True)
    error_message: Optional[str] = None
    
    # Performance
    duration_ms: Optional[int] = Field(None, ge=0)
    response_size: Optional[int] = Field(None, ge=0)


class APIKey(UUIDMixin, TimestampMixin):
    """API key for programmatic access"""
    user_id: UUID = Field(..., description="Reference to user")
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    key_hash: str = Field(..., description="Hashed API key")
    
    # Access control
    permissions: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = Field(None, description="Requests per minute")
    allowed_ips: List[str] = Field(default_factory=list)
    
    # Lifecycle
    is_active: bool = Field(default=True)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = Field(default=0, ge=0)
    
    # Security
    created_by: UUID = Field(..., description="User who created the key")
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[UUID] = None
    revocation_reason: Optional[str] = None


class UserNotification(UUIDMixin, TimestampMixin):
    """User notification model"""
    user_id: UUID = Field(..., description="Reference to user")
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1)
    
    # Notification details
    notification_type: str = Field(..., description="Type of notification")
    priority: str = Field(default="normal", description="low, normal, high, urgent")
    category: str = Field(..., description="Notification category")
    
    # Delivery
    delivery_channels: List[str] = Field(default_factory=list)
    is_read: bool = Field(default=False)
    read_at: Optional[datetime] = None
    
    # Action
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)