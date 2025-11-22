"""
Health monitoring and alerting system.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import psutil
import aioredis
import asyncpg

from app.monitoring.logger import get_logger, log_security_event
from app.monitoring.metrics import metrics
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration: float
    details: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """Monitors system health and sends alerts."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.alert_handlers: List[Callable] = []
        self.check_interval = 30  # seconds
        self._monitoring_task = None
        
        # Register default health checks
        self._register_default_checks()
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def register_alert_handler(self, handler: Callable) -> None:
        """Register an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Registered alert handler")
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a single health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                timestamp=datetime.utcnow(),
                duration=0.0
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration = time.time() - start_time
            
            if isinstance(result, HealthCheck):
                result.duration = duration
                return result
            elif isinstance(result, tuple):
                status, message, details = result
                return HealthCheck(
                    name=name,
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    duration=duration,
                    details=details
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.CRITICAL,
                    message="Check passed" if result else "Check failed",
                    timestamp=datetime.utcnow(),
                    duration=duration
                )
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed with exception: {str(e)}",
                timestamp=datetime.utcnow(),
                duration=duration
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.checks:
            result = await self.run_check(name)
            results[name] = result
            self.last_results[name] = result
            
            # Track metrics
            metrics.gauge(f"health.{name}.status", 1 if result.status == HealthStatus.HEALTHY else 0)
            metrics.timing(f"health.{name}.duration", result.duration)
        
        return results
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = await self.run_all_checks()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        critical_count = 0
        warning_count = 0
        
        for result in results.values():
            if result.status == HealthStatus.CRITICAL:
                critical_count += 1
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.WARNING:
                warning_count += 1
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: {
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "details": result.details
            } for name, result in results.items()},
            "summary": {
                "total_checks": len(results),
                "healthy": len([r for r in results.values() if r.status == HealthStatus.HEALTHY]),
                "warning": warning_count,
                "critical": critical_count
            }
        }
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_task:
            return
        
        async def monitor_loop():
            while True:
                try:
                    results = await self.run_all_checks()
                    
                    # Check for status changes and send alerts
                    for name, result in results.items():
                        await self._check_for_alerts(name, result)
                    
                    await asyncio.sleep(self.check_interval)
                
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                    await asyncio.sleep(self.check_interval)
        
        self._monitoring_task = asyncio.create_task(monitor_loop())
        logger.info("Started health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Stopped health monitoring")
    
    async def _check_for_alerts(self, name: str, current: HealthCheck) -> None:
        """Check if an alert should be sent for a health check result."""
        previous = self.last_results.get(name)
        
        # Send alert if status changed to warning or critical
        if (not previous or 
            (previous.status != current.status and 
             current.status in [HealthStatus.WARNING, HealthStatus.CRITICAL])):
            
            await self._send_alert(name, current, previous)
    
    async def _send_alert(self, name: str, current: HealthCheck, previous: Optional[HealthCheck]) -> None:
        """Send alert for health check status change."""
        alert_data = {
            "check_name": name,
            "current_status": current.status.value,
            "current_message": current.message,
            "timestamp": current.timestamp.isoformat(),
            "duration": current.duration
        }
        
        if previous:
            alert_data["previous_status"] = previous.status.value
            alert_data["previous_message"] = previous.message
        
        # Log security event for critical issues
        if current.status == HealthStatus.CRITICAL:
            log_security_event(
                "health_check_critical",
                alert_data
            )
        
        # Send to all registered alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert_data)
                else:
                    handler(alert_data)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        
        async def check_database():
            """Check database connectivity."""
            try:
                conn = await asyncpg.connect(settings.DATABASE_URL)
                await conn.execute("SELECT 1")
                await conn.close()
                return HealthStatus.HEALTHY, "Database connection successful", None
            except Exception as e:
                return HealthStatus.CRITICAL, f"Database connection failed: {str(e)}", None
        
        async def check_redis():
            """Check Redis connectivity."""
            try:
                redis = aioredis.from_url(settings.CELERY_BROKER_URL)
                await redis.ping()
                await redis.close()
                return HealthStatus.HEALTHY, "Redis connection successful", None
            except Exception as e:
                return HealthStatus.CRITICAL, f"Redis connection failed: {str(e)}", None
        
        def check_memory():
            """Check system memory usage."""
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                return HealthStatus.CRITICAL, f"Memory usage critical: {usage_percent:.1f}%", {"usage": usage_percent}
            elif usage_percent > 80:
                return HealthStatus.WARNING, f"Memory usage high: {usage_percent:.1f}%", {"usage": usage_percent}
            else:
                return HealthStatus.HEALTHY, f"Memory usage normal: {usage_percent:.1f}%", {"usage": usage_percent}
        
        def check_cpu():
            """Check CPU usage."""
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                return HealthStatus.CRITICAL, f"CPU usage critical: {cpu_percent:.1f}%", {"usage": cpu_percent}
            elif cpu_percent > 80:
                return HealthStatus.WARNING, f"CPU usage high: {cpu_percent:.1f}%", {"usage": cpu_percent}
            else:
                return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent:.1f}%", {"usage": cpu_percent}
        
        def check_disk():
            """Check disk space."""
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 90:
                return HealthStatus.CRITICAL, f"Disk usage critical: {usage_percent:.1f}%", {"usage": usage_percent}
            elif usage_percent > 80:
                return HealthStatus.WARNING, f"Disk usage high: {usage_percent:.1f}%", {"usage": usage_percent}
            else:
                return HealthStatus.HEALTHY, f"Disk usage normal: {usage_percent:.1f}%", {"usage": usage_percent}
        
        # Register checks
        self.register_check("database", check_database)
        self.register_check("redis", check_redis)
        self.register_check("memory", check_memory)
        self.register_check("cpu", check_cpu)
        self.register_check("disk", check_disk)


# Global health monitor instance
health_monitor = HealthMonitor()


async def simple_alert_handler(alert_data: Dict[str, Any]) -> None:
    """Simple alert handler that logs alerts."""
    logger.warning(f"Health Alert: {alert_data}")


# Register default alert handler
health_monitor.register_alert_handler(simple_alert_handler)