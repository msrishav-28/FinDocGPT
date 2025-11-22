"""
Middleware for automatic audit logging of HTTP requests
"""

import time
import json
from typing import Callable, Optional
from uuid import uuid4
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ..models.audit import AuditEventType, AuditSeverity
from ..services.audit_service import audit_service


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log HTTP requests for audit purposes"""
    
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip audit logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID for correlation
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # Extract user information if available
        user_id = None
        username = None
        user_role = None
        session_id = None
        
        # Try to get user info from request state (set by auth middleware)
        if hasattr(request.state, 'user'):
            user = request.state.user
            user_id = getattr(user, 'id', None)
            username = getattr(user, 'username', None)
            user_role = getattr(user, 'role', None)
        
        # Try to get session ID from headers or cookies
        session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id')
        
        # Get client information
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent')
        
        # Capture request body for certain endpoints (be careful with sensitive data)
        request_body = None
        if request.method in ['POST', 'PUT', 'PATCH'] and self._should_log_body(request):
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON, fallback to string
                    try:
                        request_body = json.loads(body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        request_body = {"raw_body": body.decode('utf-8', errors='ignore')[:1000]}
                
                # Re-create request with body for downstream processing
                async def receive():
                    return {"type": "http.request", "body": body}
                
                request._receive = receive
            except Exception:
                # If we can't read the body, continue without it
                pass
        
        # Record start time
        start_time = time.time()
        
        # Determine event type based on HTTP method and path
        event_type = self._determine_event_type(request)
        
        # Process request
        response = None
        error_occurred = False
        error_message = None
        status_code = None
        response_size = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculate response size if possible
            if hasattr(response, 'body'):
                if isinstance(response.body, bytes):
                    response_size = len(response.body)
                elif isinstance(response, StreamingResponse):
                    # For streaming responses, we can't easily calculate size
                    response_size = None
            
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            status_code = 500
            raise
        
        finally:
            # Calculate duration
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            
            # Determine success based on status code
            success = not error_occurred and (status_code < 400 if status_code else False)
            
            # Determine severity based on status code and method
            severity = self._determine_severity(request.method, status_code, error_occurred)
            
            # Prepare event data
            event_data = {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "status_code": status_code,
                "content_type": request.headers.get('content-type'),
                "accept": request.headers.get('accept'),
                "referer": request.headers.get('referer'),
            }
            
            # Add request body if captured (and not sensitive)
            if request_body and not self._contains_sensitive_data(request_body):
                event_data["request_body"] = request_body
            
            # Log the audit event
            try:
                await audit_service.log_event(
                    event_type=event_type,
                    event_name=f"{request.method} {request.url.path}",
                    description=f"HTTP {request.method} request to {request.url.path}",
                    user_id=user_id,
                    username=username,
                    user_role=user_role,
                    session_id=session_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    request_id=request_id,
                    endpoint=request.url.path,
                    http_method=request.method,
                    event_data=event_data,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    response_size=response_size,
                    severity=severity
                )
            except Exception as audit_error:
                # Don't let audit logging failures break the request
                print(f"Audit logging failed: {audit_error}")
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _determine_event_type(self, request: Request) -> AuditEventType:
        """Determine audit event type based on request"""
        path = request.url.path.lower()
        method = request.method.upper()
        
        # Authentication endpoints
        if '/auth/' in path or '/login' in path:
            if method == 'POST':
                return AuditEventType.USER_LOGIN
            elif '/logout' in path:
                return AuditEventType.USER_LOGOUT
        
        # Document endpoints
        if '/documents' in path:
            if method == 'POST':
                return AuditEventType.DOCUMENT_UPLOADED
            elif method == 'GET':
                return AuditEventType.DOCUMENT_VIEWED
            elif method == 'DELETE':
                return AuditEventType.DOCUMENT_DELETED
        
        # Analysis endpoints
        if '/sentiment' in path:
            return AuditEventType.SENTIMENT_ANALYSIS
        elif '/anomaly' in path or '/anomalies' in path:
            return AuditEventType.ANOMALY_DETECTION
        elif '/forecast' in path:
            return AuditEventType.FORECAST_GENERATED
        elif '/recommend' in path:
            return AuditEventType.RECOMMENDATION_CREATED
        elif '/question' in path or '/ask' in path:
            return AuditEventType.QUESTION_ASKED
        
        # User management
        if '/users' in path:
            if method == 'POST':
                return AuditEventType.USER_CREATED
            elif method in ['PUT', 'PATCH']:
                return AuditEventType.USER_UPDATED
            elif method == 'DELETE':
                return AuditEventType.USER_DELETED
        
        # Default to generic system action
        return AuditEventType.SYSTEM_CONFIG_CHANGED
    
    def _determine_severity(self, method: str, status_code: Optional[int], error_occurred: bool) -> AuditSeverity:
        """Determine audit severity based on request characteristics"""
        if error_occurred or (status_code and status_code >= 500):
            return AuditSeverity.HIGH
        elif status_code and status_code >= 400:
            return AuditSeverity.MEDIUM
        elif method in ['DELETE', 'PUT', 'PATCH']:
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.LOW
    
    def _should_log_body(self, request: Request) -> bool:
        """Determine if request body should be logged"""
        path = request.url.path.lower()
        
        # Don't log bodies for sensitive endpoints
        sensitive_paths = [
            '/auth/',
            '/login',
            '/password',
            '/token'
        ]
        
        if any(sensitive_path in path for sensitive_path in sensitive_paths):
            return False
        
        # Don't log large file uploads
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
            return False
        
        return True
    
    def _contains_sensitive_data(self, data: dict) -> bool:
        """Check if data contains sensitive information that shouldn't be logged"""
        sensitive_keys = {
            'password', 'token', 'secret', 'key', 'auth', 'credential',
            'ssn', 'social_security', 'credit_card', 'bank_account',
            'private_key', 'api_key', 'access_token', 'refresh_token'
        }
        
        def check_dict(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                        return True
                    if isinstance(value, (dict, list)):
                        if check_dict(value):
                            return True
            elif isinstance(obj, list):
                for item in obj:
                    if check_dict(item):
                        return True
            return False
        
        return check_dict(data)