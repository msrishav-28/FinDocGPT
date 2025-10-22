"""
Monitoring and maintenance tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List
from celery import current_task
from ..celery_app import celery_app
from ..services.monitoring_service import MonitoringService
from ..services.cache_service import CacheService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.monitoring_tasks.system_health_check")
def system_health_check(self) -> Dict[str, Any]:
    """
    Perform comprehensive system health check
    
    Returns:
        Dictionary with health check results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting system health check"})
        
        monitoring_service = MonitoringService()
        
        # Check database health
        self.update_state(state="PROCESSING", meta={"status": "Checking database health"})
        db_health = monitoring_service.check_database_health()
        
        # Check Redis health
        self.update_state(state="PROCESSING", meta={"status": "Checking Redis health"})
        redis_health = monitoring_service.check_redis_health()
        
        # Check external API health
        self.update_state(state="PROCESSING", meta={"status": "Checking external APIs"})
        api_health = monitoring_service.check_external_apis_health()
        
        # Check model performance
        self.update_state(state="PROCESSING", meta={"status": "Checking model performance"})
        model_health = monitoring_service.check_model_performance()
        
        # Check system resources
        self.update_state(state="PROCESSING", meta={"status": "Checking system resources"})
        resource_health = monitoring_service.check_system_resources()
        
        overall_status = "healthy"
        if any(check.get("status") != "healthy" for check in [db_health, redis_health, api_health, model_health, resource_health]):
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "database": db_health,
            "redis": redis_health,
            "external_apis": api_health,
            "models": model_health,
            "system_resources": resource_health,
            "timestamp": monitoring_service.get_current_timestamp(),
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Health check failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.monitoring_tasks.cleanup_old_results")
def cleanup_old_results(self, retention_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up old task results and logs
    
    Args:
        retention_hours: Number of hours to retain results
        
    Returns:
        Dictionary with cleanup results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting cleanup of old results"})
        
        monitoring_service = MonitoringService()
        
        # Clean up Celery results
        self.update_state(state="PROCESSING", meta={"status": "Cleaning up task results"})
        celery_cleanup = monitoring_service.cleanup_celery_results(retention_hours)
        
        # Clean up application logs
        self.update_state(state="PROCESSING", meta={"status": "Cleaning up application logs"})
        log_cleanup = monitoring_service.cleanup_application_logs(retention_hours)
        
        # Clean up temporary files
        self.update_state(state="PROCESSING", meta={"status": "Cleaning up temporary files"})
        temp_cleanup = monitoring_service.cleanup_temporary_files()
        
        return {
            "retention_hours": retention_hours,
            "celery_cleanup": celery_cleanup,
            "log_cleanup": log_cleanup,
            "temp_cleanup": temp_cleanup,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Cleanup of old results failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Cleanup failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.monitoring_tasks.performance_monitoring")
def performance_monitoring(self) -> Dict[str, Any]:
    """
    Monitor system and application performance
    
    Returns:
        Dictionary with performance metrics
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting performance monitoring"})
        
        monitoring_service = MonitoringService()
        
        # Collect performance metrics
        self.update_state(state="PROCESSING", meta={"status": "Collecting performance metrics"})
        performance_metrics = monitoring_service.collect_performance_metrics()
        
        # Analyze trends
        self.update_state(state="PROCESSING", meta={"status": "Analyzing performance trends"})
        trend_analysis = monitoring_service.analyze_performance_trends()
        
        # Check for performance alerts
        self.update_state(state="PROCESSING", meta={"status": "Checking for performance alerts"})
        alerts = monitoring_service.check_performance_alerts(performance_metrics)
        
        return {
            "performance_metrics": performance_metrics,
            "trend_analysis": trend_analysis,
            "alerts": alerts,
            "timestamp": monitoring_service.get_current_timestamp(),
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Performance monitoring failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.monitoring_tasks.cache_warming")
def cache_warming(self, cache_keys: List[str]) -> Dict[str, Any]:
    """
    Warm up cache with frequently accessed data
    
    Args:
        cache_keys: List of cache keys to warm up
        
    Returns:
        Dictionary with cache warming results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting cache warming"})
        
        cache_service = CacheService()
        
        warmed_keys = []
        failed_keys = []
        
        for i, key in enumerate(cache_keys):
            try:
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Warming cache key {i+1}/{len(cache_keys)}",
                        "progress": (i / len(cache_keys)) * 100
                    }
                )
                
                # Warm up cache key
                result = cache_service.warm_cache_key(key)
                warmed_keys.append({
                    "key": key,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Failed to warm cache key {key}: {str(e)}")
                failed_keys.append({
                    "key": key,
                    "error": str(e)
                })
        
        return {
            "total_keys": len(cache_keys),
            "warmed_successfully": len(warmed_keys),
            "failed_keys": len(failed_keys),
            "warmed_keys": warmed_keys,
            "failed_cache_keys": failed_keys,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Cache warming failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Cache warming failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.monitoring_tasks.generate_system_report")
def generate_system_report(self, report_type: str = "daily") -> Dict[str, Any]:
    """
    Generate comprehensive system report
    
    Args:
        report_type: Type of report (daily, weekly, monthly)
        
    Returns:
        Dictionary with system report
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting system report generation"})
        
        monitoring_service = MonitoringService()
        
        # Generate report based on type
        self.update_state(state="PROCESSING", meta={"status": f"Generating {report_type} report"})
        report = monitoring_service.generate_system_report(report_type)
        
        # Save report
        self.update_state(state="PROCESSING", meta={"status": "Saving report"})
        report_path = monitoring_service.save_system_report(report, report_type)
        
        return {
            "report_type": report_type,
            "report": report,
            "report_path": report_path,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"System report generation failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Report generation failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.monitoring_tasks.alert_processing")
def alert_processing(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and route system alerts
    
    Args:
        alert_data: Alert information
        
    Returns:
        Dictionary with alert processing results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Processing system alert"})
        
        monitoring_service = MonitoringService()
        
        # Process alert
        processed_alert = monitoring_service.process_alert(alert_data)
        
        # Route alert to appropriate channels
        routing_results = monitoring_service.route_alert(processed_alert)
        
        return {
            "alert_data": alert_data,
            "processed_alert": processed_alert,
            "routing_results": routing_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Alert processing failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Alert processing failed"}
        )
        raise