"""
Monitoring middleware for request tracking and performance metrics
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from ..services.monitoring_service import monitoring_service


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics and performance"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Extract endpoint information
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code
        
        # Record metrics
        monitoring_service.record_request(endpoint, method, response_time, status_code)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Request-ID"] = getattr(request.state, 'request_id', 'unknown')
        
        return response