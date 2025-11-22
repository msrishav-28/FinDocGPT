"""
Security utilities for the Financial Intelligence System.

This module provides authentication, authorization, and security-related
functionality used throughout the application.
"""

from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import settings
from .exceptions import AuthenticationError, AuthorizationError

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.api.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.api.secret_key, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, settings.api.secret_key, algorithms=["HS256"])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current authenticated user from JWT token."""
    try:
        payload = verify_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token: missing user ID")
        
        # Here you would typically fetch user from database
        # For now, return the payload
        return {"user_id": user_id, **payload}
    
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permissions(*required_permissions: str):
    """Decorator to require specific permissions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs or dependencies
            current_user = kwargs.get('current_user')
            if not current_user:
                raise AuthorizationError("User authentication required")
            
            user_permissions = current_user.get('permissions', [])
            
            # Check if user has all required permissions
            missing_permissions = set(required_permissions) - set(user_permissions)
            if missing_permissions:
                raise AuthorizationError(
                    f"Missing required permissions: {', '.join(missing_permissions)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def is_admin(current_user: dict) -> bool:
    """Check if current user is an admin."""
    return current_user.get('role') == 'admin'


def can_access_resource(current_user: dict, resource_owner_id: str) -> bool:
    """Check if user can access a specific resource."""
    # Admin can access everything
    if is_admin(current_user):
        return True
    
    # User can access their own resources
    return current_user.get('user_id') == resource_owner_id