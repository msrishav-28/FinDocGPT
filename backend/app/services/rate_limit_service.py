"""
Rate limiting service for API request throttling
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from fastapi import HTTPException, status, Request
from ..config import get_settings

settings = get_settings()


class RateLimitService:
    """Rate limiting service using sliding window algorithm"""
    
    def __init__(self):
        # Store request timestamps for each client
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Rate limit configurations (requests per minute)
        self.rate_limits = {
            "default": 60,  # Default rate limit
            "authenticated": 120,  # Higher limit for authenticated users
            "admin": 300,  # Even higher for admin users
            "upload": 10,  # Lower limit for file uploads
            "forecast": 30,  # Moderate limit for forecasting endpoints
            "analysis": 60,  # Standard limit for analysis endpoints
        }
        
        # Burst limits (requests per second)
        self.burst_limits = {
            "default": 10,
            "authenticated": 20,
            "admin": 50,
            "upload": 2,
            "forecast": 5,
            "analysis": 10,
        }
    
    def _get_client_id(self, request: Request, user_id: Optional[str] = None) -> str:
        """Get unique client identifier"""
        if user_id:
            return f"user:{user_id}"
        
        # Use IP address as fallback
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _clean_old_requests(self, request_times: deque, window_seconds: int):
        """Remove requests older than the time window"""
        current_time = time.time()
        while request_times and request_times[0] < current_time - window_seconds:
            request_times.popleft()
    
    def check_rate_limit(
        self, 
        request: Request, 
        limit_type: str = "default",
        user_id: Optional[str] = None,
        user_role: Optional[str] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limits
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        client_id = self._get_client_id(request, user_id)
        current_time = time.time()
        
        # Determine rate limit based on user role
        if user_role == "admin":
            effective_limit_type = "admin"
        elif user_id:
            effective_limit_type = "authenticated"
        else:
            effective_limit_type = limit_type
        
        # Get rate limits
        minute_limit = self.rate_limits.get(effective_limit_type, self.rate_limits["default"])
        burst_limit = self.burst_limits.get(effective_limit_type, self.burst_limits["default"])
        
        # Get request history for this client
        request_times = self.request_history[client_id]
        
        # Clean old requests (older than 1 minute)
        self._clean_old_requests(request_times, 60)
        
        # Check burst limit (requests in last second)
        recent_requests = sum(1 for req_time in request_times if req_time > current_time - 1)
        if recent_requests >= burst_limit:
            return False, {
                "limit": minute_limit,
                "remaining": max(0, minute_limit - len(request_times)),
                "reset": int(current_time + 60),
                "burst_limit": burst_limit,
                "burst_remaining": max(0, burst_limit - recent_requests)
            }
        
        # Check minute limit
        if len(request_times) >= minute_limit:
            return False, {
                "limit": minute_limit,
                "remaining": 0,
                "reset": int(current_time + 60),
                "burst_limit": burst_limit,
                "burst_remaining": max(0, burst_limit - recent_requests)
            }
        
        # Add current request
        request_times.append(current_time)
        
        return True, {
            "limit": minute_limit,
            "remaining": max(0, minute_limit - len(request_times)),
            "reset": int(current_time + 60),
            "burst_limit": burst_limit,
            "burst_remaining": max(0, burst_limit - recent_requests - 1)
        }
    
    def get_rate_limit_headers(self, rate_limit_info: Dict[str, int]) -> Dict[str, str]:
        """Get rate limit headers for response"""
        return {
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
            "X-RateLimit-Reset": str(rate_limit_info["reset"]),
            "X-RateLimit-Burst-Limit": str(rate_limit_info["burst_limit"]),
            "X-RateLimit-Burst-Remaining": str(rate_limit_info["burst_remaining"])
        }
    
    def clear_client_history(self, client_id: str):
        """Clear rate limit history for a client"""
        if client_id in self.request_history:
            del self.request_history[client_id]
    
    def get_client_stats(self, client_id: str) -> Dict[str, int]:
        """Get rate limit statistics for a client"""
        request_times = self.request_history.get(client_id, deque())
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(request_times, 60)
        
        recent_requests = sum(1 for req_time in request_times if req_time > current_time - 1)
        
        return {
            "requests_last_minute": len(request_times),
            "requests_last_second": recent_requests,
            "first_request": int(request_times[0]) if request_times else 0,
            "last_request": int(request_times[-1]) if request_times else 0
        }


# Global rate limit service instance
rate_limit_service = RateLimitService()