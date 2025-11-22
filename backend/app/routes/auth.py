"""
Authentication routes for user management and JWT tokens
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from ..models.auth import (
    User, UserCreate, UserUpdate, Token, LoginRequest, 
    RefreshTokenRequest, PasswordChangeRequest
)
from ..services.auth_service import auth_service
from ..dependencies.auth import (
    get_current_user, require_authentication, require_manage_users, rate_limit
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    response: Response,
    login_data: LoginRequest,
    _: None = Depends(rate_limit("default"))
):
    """Login user and return JWT tokens"""
    try:
        token = auth_service.login(login_data.username, login_data.password)
        
        # Add rate limit headers if available
        if hasattr(request.state, 'rate_limit_headers'):
            for key, value in request.state.rate_limit_headers.items():
                response.headers[key] = value
        
        return token
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/login/form", response_model=Token)
async def login_form(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    _: None = Depends(rate_limit("default"))
):
    """Login using OAuth2 password form (for compatibility)"""
    try:
        token = auth_service.login(form_data.username, form_data.password)
        
        # Add rate limit headers if available
        if hasattr(request.state, 'rate_limit_headers'):
            for key, value in request.state.rate_limit_headers.items():
                response.headers[key] = value
        
        return token
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: Request,
    response: Response,
    refresh_data: RefreshTokenRequest,
    _: None = Depends(rate_limit("default"))
):
    """Refresh access token using refresh token"""
    token = auth_service.refresh_access_token(refresh_data.refresh_token)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return token


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    refresh_data: RefreshTokenRequest,
    current_user: User = Depends(require_authentication),
    _: None = Depends(rate_limit("authenticated"))
):
    """Logout user by revoking refresh token"""
    success = auth_service.revoke_refresh_token(refresh_data.refresh_token)
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return {"message": "Logged out successfully", "success": success}


@router.get("/me", response_model=User)
async def get_current_user_info(
    request: Request,
    response: Response,
    current_user: User = Depends(require_authentication),
    _: None = Depends(rate_limit("authenticated"))
):
    """Get current user information"""
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return current_user


@router.post("/change-password")
async def change_password(
    request: Request,
    response: Response,
    password_data: PasswordChangeRequest,
    current_user: User = Depends(require_authentication),
    _: None = Depends(rate_limit("authenticated"))
):
    """Change user password"""
    # Verify current password
    user_in_db = auth_service.get_user(current_user.username)
    if not user_in_db or not auth_service.verify_password(
        password_data.current_password, user_in_db.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    user_in_db.hashed_password = auth_service.get_password_hash(password_data.new_password)
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return {"message": "Password changed successfully"}


@router.post("/users", response_model=User)
async def create_user(
    request: Request,
    response: Response,
    user_data: UserCreate,
    current_user: User = Depends(require_manage_users),
    _: None = Depends(rate_limit("authenticated"))
):
    """Create a new user (admin only)"""
    try:
        user = auth_service.create_user(user_data)
        
        # Add rate limit headers if available
        if hasattr(request.state, 'rate_limit_headers'):
            for key, value in request.state.rate_limit_headers.items():
                response.headers[key] = value
        
        return User(**user.dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/users")
async def list_users(
    request: Request,
    response: Response,
    current_user: User = Depends(require_manage_users),
    _: None = Depends(rate_limit("authenticated"))
):
    """List all users (admin only)"""
    users = []
    for user_in_db in auth_service.users_db.values():
        users.append(User(**user_in_db.dict()))
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return {"users": users}


@router.get("/users/{username}", response_model=User)
async def get_user(
    request: Request,
    response: Response,
    username: str,
    current_user: User = Depends(require_manage_users),
    _: None = Depends(rate_limit("authenticated"))
):
    """Get user by username (admin only)"""
    user = auth_service.get_user(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return User(**user.dict())


@router.put("/users/{username}", response_model=User)
async def update_user(
    request: Request,
    response: Response,
    username: str,
    user_update: UserUpdate,
    current_user: User = Depends(require_manage_users),
    _: None = Depends(rate_limit("authenticated"))
):
    """Update user (admin only)"""
    user = auth_service.get_user(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return User(**user.dict())


@router.delete("/users/{username}")
async def delete_user(
    request: Request,
    response: Response,
    username: str,
    current_user: User = Depends(require_manage_users),
    _: None = Depends(rate_limit("authenticated"))
):
    """Delete user (admin only)"""
    if username == current_user.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    if username not in auth_service.users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    del auth_service.users_db[username]
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return {"message": "User deleted successfully"}