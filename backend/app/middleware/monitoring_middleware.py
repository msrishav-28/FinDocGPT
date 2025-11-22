"""
Monitoring middleware for tracking API requests and performance metrics.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.monitoring.logger import get_logger, log_request
from app.monitoring.metrics import metrics, track_api_request

logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API requests and performance."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics."""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract user ID if available
        user_id = None
        if hasattr(request.state, 'user') and request.state.user:
            user_id = str(request.state.user.id)
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_id": user_id
            }
        )
        
        # Track request start
        metrics.increment("http.requests.started", tags={
            "method": request.method,
            "path": self._normalize_path(request.url.path)
        })
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Track successful request metrics
            track_api_request(
                endpoint=self._normalize_path(request.url.path),
                method=request.method,
                status_code=response.status_code,
                duration=duration
            )
            
            # Track response size if available
            if "content-length" in response.headers:
                response_size = int(response.headers["content-length"])
                metrics.histogram("http.response.size", response_size, 
                                tags={"method": request.method, "status": str(response.status_code)}, 
                                unit="bytes")
            
            # Log request completion
            log_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                user_id=user_id,
                request_id=request_id
            )
            
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration": duration,
                    "user_id": user_id
                }
            )
            
            # Add monitoring headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Track failed request metrics
            metrics.increment("http.requests.failed", tags={
                "method": request.method,
                "path": self._normalize_path(request.url.path),
                "error_type": type(e).__name__
            })
            
            metrics.timing("http.request.duration.failed", duration, tags={
                "method": request.method,
                "path": self._normalize_path(request.url.path)
            })
            
            # Log request error
            log_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration=duration,
                user_id=user_id,
                request_id=request_id
            )
            
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": duration,
                    "error": str(e),
                    "user_id": user_id
                },
                exc_info=True
            )
            
            # Re-raise the exception
            raise
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics to avoid high cardinality."""
        # Replace path parameters with placeholders
        import re
        
        # Replace UUIDs with placeholder
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
        
        # Replace numeric IDs with placeholder
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace other common patterns
        path = re.sub(r'/[^/]+\.(jpg|jpeg|png|gif|pdf|csv|json|xml)$', '/{file}', path)
        
        return path