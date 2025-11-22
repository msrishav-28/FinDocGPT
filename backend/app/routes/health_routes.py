"""
Health check endpoints for monitoring and load balancer health checks.
"""

from fastapi import APIRouter, Depends
from app.monitoring.health import health_monitor
from app.monitoring.metrics import metrics
from app.monitoring.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Basic health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "service": "financial-intelligence-backend",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system status."""
    try:
        health_status = await health_monitor.get_system_health()
        return health_status
    except Exception as e:
        logger.error(f"Error getting detailed health status: {e}")
        return {
            "status": "error",
            "message": "Unable to retrieve detailed health status",
            "error": str(e)
        }


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        # Check critical dependencies
        health_results = await health_monitor.run_all_checks()
        
        # Consider ready if database and redis are healthy
        critical_checks = ["database", "redis"]
        for check_name in critical_checks:
            if check_name in health_results:
                if health_results[check_name].status.value != "healthy":
                    return {
                        "status": "not_ready",
                        "reason": f"{check_name} is not healthy"
                    }, 503
        
        return {"status": "ready"}
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "reason": "Health check failed",
            "error": str(e)
        }, 503


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    try:
        # Simple check to ensure the application is responsive
        metrics.increment("health.liveness_check")
        return {"status": "alive"}
    
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return {
            "status": "not_alive",
            "error": str(e)
        }, 503