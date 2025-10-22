"""
Rate limiting middleware for adding headers to responses
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to add rate limit headers to responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add rate limit headers if they were set during request processing
        if hasattr(request.state, 'rate_limit_headers'):
            for key, value in request.state.rate_limit_headers.items():
                response.headers[key] = value
        
        return response