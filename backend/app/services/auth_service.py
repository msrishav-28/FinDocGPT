"""
Authentication service for JWT token management and user authentication
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from ..models.auth import User, UserInDB, UserCreate, Token, TokenData, UserRole, ROLE_PERMISSIONS
from ..config import get_settings

settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.api.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.api.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = 7


class AuthService:
    """Authentication service for user management and JWT tokens"""
    
    def __init__(self):
        self.users_db: Dict[str, UserInDB] = {}
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        admin_user = UserInDB(
            id=str(uuid.uuid4()),
            email="admin@financial-intelligence.com",
            username="admin",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            hashed_password=self.get_password_hash("admin123"),
            created_at=datetime.utcnow()
        )
        self.users_db["admin"] = admin_user
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        return self.users_db.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username and password"""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, username: str) -> str:
        """Create JWT refresh token"""
        token_id = str(uuid.uuid4())
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": username,
            "exp": expire,
            "type": "refresh",
            "jti": token_id
        }
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Store refresh token info
        self.refresh_tokens[token_id] = {
            "username": username,
            "expires_at": expire,
            "created_at": datetime.utcnow()
        }
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        """Verify JWT token and return token data"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            token_type_claim: str = payload.get("type")
            
            if username is None or token_type_claim != token_type:
                return None
            
            # Get user permissions
            user = self.get_user(username)
            if not user:
                return None
            
            permissions = [perm.value for perm in ROLE_PERMISSIONS.get(user.role, [])]
            
            return TokenData(username=username, permissions=permissions)
        except JWTError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Token]:
        """Create new access token using refresh token"""
        token_data = self.verify_token(refresh_token, "refresh")
        if not token_data:
            return None
        
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            token_id = payload.get("jti")
            
            # Check if refresh token is still valid
            if token_id not in self.refresh_tokens:
                return None
            
            token_info = self.refresh_tokens[token_id]
            if token_info["expires_at"] < datetime.utcnow():
                # Remove expired token
                del self.refresh_tokens[token_id]
                return None
            
            # Create new access token
            user = self.get_user(token_data.username)
            if not user:
                return None
            
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = self.create_access_token(
                data={"sub": user.username}, expires_delta=access_token_expires
            )
            
            return Token(
                access_token=access_token,
                refresh_token=refresh_token,  # Keep the same refresh token
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        except JWTError:
            return None
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke a refresh token"""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            token_id = payload.get("jti")
            
            if token_id in self.refresh_tokens:
                del self.refresh_tokens[token_id]
                return True
            return False
        except JWTError:
            return False
    
    def create_user(self, user_create: UserCreate) -> UserInDB:
        """Create a new user"""
        if user_create.username in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        hashed_password = self.get_password_hash(user_create.password)
        user = UserInDB(
            id=str(uuid.uuid4()),
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            role=user_create.role,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        
        self.users_db[user.username] = user
        return user
    
    def login(self, username: str, password: str) -> Token:
        """Login user and return tokens"""
        user = self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        # Create tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        refresh_token = self.create_refresh_token(user.username)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        user_permissions = ROLE_PERMISSIONS.get(user.role, [])
        return any(perm.value == permission for perm in user_permissions)


# Global auth service instance
auth_service = AuthService()