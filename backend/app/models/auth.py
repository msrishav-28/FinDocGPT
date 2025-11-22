"""
Authentication models for the Financial Intelligence System
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from enum import Enum


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    ANALYST = "analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    VIEWER = "viewer"


class UserPermission(str, Enum):
    """User permissions"""
    READ_DOCUMENTS = "read_documents"
    UPLOAD_DOCUMENTS = "upload_documents"
    DELETE_DOCUMENTS = "delete_documents"
    VIEW_ANALYTICS = "view_analytics"
    GENERATE_FORECASTS = "generate_forecasts"
    CREATE_RECOMMENDATIONS = "create_recommendations"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"
    COMPLIANCE_OFFICER = "compliance_officer"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        UserPermission.READ_DOCUMENTS,
        UserPermission.UPLOAD_DOCUMENTS,
        UserPermission.DELETE_DOCUMENTS,
        UserPermission.VIEW_ANALYTICS,
        UserPermission.GENERATE_FORECASTS,
        UserPermission.CREATE_RECOMMENDATIONS,
        UserPermission.MANAGE_USERS,
        UserPermission.SYSTEM_ADMIN,
        UserPermission.COMPLIANCE_OFFICER,
    ],
    UserRole.ANALYST: [
        UserPermission.READ_DOCUMENTS,
        UserPermission.UPLOAD_DOCUMENTS,
        UserPermission.VIEW_ANALYTICS,
        UserPermission.GENERATE_FORECASTS,
    ],
    UserRole.PORTFOLIO_MANAGER: [
        UserPermission.READ_DOCUMENTS,
        UserPermission.UPLOAD_DOCUMENTS,
        UserPermission.VIEW_ANALYTICS,
        UserPermission.GENERATE_FORECASTS,
        UserPermission.CREATE_RECOMMENDATIONS,
    ],
    UserRole.RISK_MANAGER: [
        UserPermission.READ_DOCUMENTS,
        UserPermission.VIEW_ANALYTICS,
        UserPermission.GENERATE_FORECASTS,
    ],
    UserRole.COMPLIANCE_OFFICER: [
        UserPermission.READ_DOCUMENTS,
        UserPermission.VIEW_ANALYTICS,
        UserPermission.COMPLIANCE_OFFICER,
        UserPermission.SYSTEM_ADMIN,  # Compliance officers need admin access for audit logs
    ],
    UserRole.VIEWER: [
        UserPermission.READ_DOCUMENTS,
        UserPermission.VIEW_ANALYTICS,
    ],
}


class User(BaseModel):
    """User model"""
    id: Optional[str] = None
    email: EmailStr
    username: str
    full_name: str
    role: UserRole
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class UserInDB(User):
    """User model with hashed password"""
    hashed_password: str


class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    username: str
    full_name: str
    password: str
    role: UserRole = UserRole.VIEWER


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    permissions: List[str] = []


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request model"""
    current_password: str
    new_password: str