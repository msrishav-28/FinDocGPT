"""
Authentication dependencies for FastAPI
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..models.auth import User, TokenData, UserPermission
from ..services.auth_service import auth_service
from ..services.rate_limit_service import rate_limit_service

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user from JWT token"""
    if not credentials:
        return None
    
    token_data = auth_service.verify_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = auth_service.get_user(token_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def require_authentication(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Require user to be authenticated"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_permission(permission: UserPermission):
    """Dependency factory for requiring specific permissions"""
    async def permission_checker(
        current_user: User = Depends(require_authentication)
    ) -> User:
        if not auth_service.has_permission(current_user, permission.value):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return current_user
    
    return permission_checker


def require_role(required_roles: list):
    """Dependency factory for requiring specific roles"""
    async def role_checker(
        current_user: User = Depends(require_authentication)
    ) -> User:
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {', '.join(required_roles)}"
            )
        return current_user
    
    return role_checker


async def check_rate_limit(
    request: Request,
    limit_type: str = "default",
    current_user: Optional[User] = Depends(get_current_user)
):
    """Check rate limits for the request"""
    user_id = current_user.id if current_user else None
    user_role = current_user.role.value if current_user else None
    
    is_allowed, rate_limit_info = rate_limit_service.check_rate_limit(
        request, limit_type, user_id, user_role
    )
    
    if not is_allowed:
        headers = rate_limit_service.get_rate_limit_headers(rate_limit_info)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers=headers
        )
    
    # Add rate limit headers to successful responses
    request.state.rate_limit_headers = rate_limit_service.get_rate_limit_headers(rate_limit_info)


def rate_limit(limit_type: str = "default"):
    """Dependency factory for rate limiting"""
    async def rate_limiter(
        request: Request,
        current_user: Optional[User] = Depends(get_current_user)
    ):
        await check_rate_limit(request, limit_type, current_user)
    
    return rate_limiter


# Common permission dependencies
require_read_documents = require_permission(UserPermission.READ_DOCUMENTS)
require_upload_documents = require_permission(UserPermission.UPLOAD_DOCUMENTS)
require_delete_documents = require_permission(UserPermission.DELETE_DOCUMENTS)
require_view_analytics = require_permission(UserPermission.VIEW_ANALYTICS)
require_generate_forecasts = require_permission(UserPermission.GENERATE_FORECASTS)
require_create_recommendations = require_permission(UserPermission.CREATE_RECOMMENDATIONS)
require_manage_users = require_permission(UserPermission.MANAGE_USERS)
require_system_admin = require_permission(UserPermission.SYSTEM_ADMIN)

# Common role dependencies
require_admin = require_role(["admin"])
require_analyst_or_above = require_role(["admin", "analyst", "portfolio_manager"])
require_manager_or_above = require_role(["admin", "portfolio_manager", "risk_manager"])