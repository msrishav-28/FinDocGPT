"""
API monitoring service for performance tracking and health monitoring
"""

import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from ..config import get_settings

settings = get_settings()


class MonitoringService:
    """Service for API performance monitoring and health tracking"""
    
    def __init__(self):
        # Request metrics
        self.request_count = 0
        self.request_times = deque(maxlen=1000)  # Last 1000 requests
        self.endpoint_metrics = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "errors": 0
        })
        
        # System metrics
        self.start_time = datetime.utcnow()
        self.health_checks = deque(maxlen=100)  # Last 100 health checks
        
        # Service health status
        self.service_status = {
            "database": {"status": "unknown", "last_check": None, "response_time": None},
            "redis": {"status": "unknown", "last_check": None, "response_time": None},
            "external_apis": {"status": "unknown", "last_check": None, "response_time": None}
        }
    
    def record_request(self, endpoint: str, method: str, response_time: float, status_code: int):
        """Record request metrics"""
        self.request_count += 1
        self.request_times.append({
            "timestamp": time.time(),
            "response_time": response_time,
            "status_code": status_code
        })
        
        # Update endpoint metrics
        key = f"{method} {endpoint}"
        metrics = self.endpoint_metrics[key]
        metrics["count"] += 1
        metrics["total_time"] += response_time
        metrics["min_time"] = min(metrics["min_time"], response_time)
        metrics["max_time"] = max(metrics["max_time"], response_time)
        
        if status_code >= 400:
            metrics["errors"] += 1 
   
    def get_request_metrics(self) -> Dict[str, Any]:
        """Get request performance metrics"""
        current_time = time.time()
        
        # Calculate metrics for last hour
        hour_ago = current_time - 3600
        recent_requests = [r for r in self.request_times if r["timestamp"] > hour_ago]
        
        if not recent_requests:
            return {
                "total_requests": self.request_count,
                "requests_last_hour": 0,
                "avg_response_time": 0,
                "error_rate": 0
            }
        
        response_times = [r["response_time"] for r in recent_requests]
        error_count = sum(1 for r in recent_requests if r["status_code"] >= 400)
        
        return {
            "total_requests": self.request_count,
            "requests_last_hour": len(recent_requests),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "error_rate": error_count / len(recent_requests) if recent_requests else 0,
            "requests_per_minute": len(recent_requests) / 60
        }
    
    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """Get per-endpoint performance metrics"""
        metrics = {}
        
        for endpoint, data in self.endpoint_metrics.items():
            if data["count"] > 0:
                metrics[endpoint] = {
                    "request_count": data["count"],
                    "avg_response_time": data["total_time"] / data["count"],
                    "min_response_time": data["min_time"],
                    "max_response_time": data["max_time"],
                    "error_count": data["errors"],
                    "error_rate": data["errors"] / data["count"]
                }
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk_usage": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "uptime": (datetime.utcnow() - self.start_time).total_seconds()
            }
        except Exception:
            return {"error": "Unable to collect system metrics"}
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            if service_name == "database":
                # Check database connection
                from ..database.connection import get_database
                db = await get_database()
                await db.fetch("SELECT 1")
                status = "healthy"
            
            elif service_name == "redis":
                # Check Redis connection
                import aioredis
                redis = aioredis.from_url(settings.database.redis_url)
                await redis.ping()
                await redis.close()
                status = "healthy"
            
            elif service_name == "external_apis":
                # Check external API availability
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL") as response:
                        if response.status == 200:
                            status = "healthy"
                        else:
                            status = "degraded"
            
            else:
                status = "unknown"
            
            response_time = time.time() - start_time
            
        except Exception as e:
            status = "unhealthy"
            response_time = time.time() - start_time
        
        # Update service status
        self.service_status[service_name] = {
            "status": status,
            "last_check": datetime.utcnow(),
            "response_time": response_time
        }
        
        return self.service_status[service_name]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        # Check all services
        for service in self.service_status.keys():
            await self.check_service_health(service)
        
        # Determine overall status
        service_statuses = [s["status"] for s in self.service_status.values()]
        
        if all(status == "healthy" for status in service_statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in service_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        health_data = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "uptime": (datetime.utcnow() - self.start_time).total_seconds(),
            "services": self.service_status.copy()
        }
        
        # Store health check result
        self.health_checks.append(health_data)
        
        return health_data
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            check for check in self.health_checks
            if datetime.fromisoformat(check["timestamp"]) > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "request_metrics": self.get_request_metrics(),
            "endpoint_metrics": self.get_endpoint_metrics(),
            "system_metrics": self.get_system_metrics(),
            "service_health": self.service_status.copy()
        }


# Global monitoring service instance
monitoring_service = MonitoringService()