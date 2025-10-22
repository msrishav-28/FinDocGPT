"""
Monitoring and health check routes
"""

from fastapi import APIRouter, Depends, Request, Response
from typing import Optional
from ..services.monitoring_service import monitoring_service
from ..dependencies.auth import require_system_admin, rate_limit

router = APIRouter(prefix="/monitoring", tags=["System"])


@router.get("/health")
async def health_check(
    request: Request,
    response: Response,
    _: None = Depends(rate_limit("default"))
):
    """
    Comprehensive health check endpoint
    
    Returns the overall system health status including:
    - Database connectivity
    - Redis connectivity  
    - External API availability
    - System resource usage
    """
    health_status = await monitoring_service.get_health_status()
    
    # Set appropriate HTTP status code
    if health_status["status"] == "healthy":
        response.status_code = 200
    elif health_status["status"] == "degraded":
        response.status_code = 200  # Still operational
    else:
        response.status_code = 503  # Service unavailable
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return health_status


@router.get("/health/history")
async def health_history(
    request: Request,
    response: Response,
    hours: int = 24,
    current_user = Depends(require_system_admin),
    _: None = Depends(rate_limit("authenticated"))
):
    """
    Get health check history
    
    Returns historical health check data for monitoring trends.
    Requires admin privileges.
    """
    history = monitoring_service.get_health_history(hours)
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return {
        "history": history,
        "period_hours": hours,
        "total_checks": len(history)
    }


@router.get("/metrics")
async def get_metrics(
    request: Request,
    response: Response,
    current_user = Depends(require_system_admin),
    _: None = Depends(rate_limit("authenticated"))
):
    """
    Get comprehensive system metrics
    
    Returns detailed performance metrics including:
    - Request statistics
    - Endpoint performance
    - System resource usage
    - Service health status
    
    Requires admin privileges.
    """
    metrics = monitoring_service.get_performance_summary()
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return metrics


@router.get("/metrics/requests")
async def get_request_metrics(
    request: Request,
    response: Response,
    current_user = Depends(require_system_admin),
    _: None = Depends(rate_limit("authenticated"))
):
    """
    Get request performance metrics
    
    Returns detailed request statistics including response times,
    error rates, and throughput metrics.
    """
    metrics = monitoring_service.get_request_metrics()
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return metrics


@router.get("/metrics/endpoints")
async def get_endpoint_metrics(
    request: Request,
    response: Response,
    current_user = Depends(require_system_admin),
    _: None = Depends(rate_limit("authenticated"))
):
    """
    Get per-endpoint performance metrics
    
    Returns performance statistics broken down by API endpoint,
    including average response times and error rates.
    """
    metrics = monitoring_service.get_endpoint_metrics()
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return metrics


@router.get("/metrics/system")
async def get_system_metrics(
    request: Request,
    response: Response,
    current_user = Depends(require_system_admin),
    _: None = Depends(rate_limit("authenticated"))
):
    """
    Get system resource metrics
    
    Returns current system resource usage including:
    - CPU utilization
    - Memory usage
    - Disk usage
    - System uptime
    """
    metrics = monitoring_service.get_system_metrics()
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return metrics


@router.get("/ping")
async def ping(
    request: Request,
    response: Response,
    _: None = Depends(rate_limit("default"))
):
    """
    Simple ping endpoint for basic connectivity testing
    
    Returns a simple response to verify the API is responding.
    """
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return {
        "status": "ok",
        "message": "pong",
        "timestamp": monitoring_service.start_time.isoformat()
    }