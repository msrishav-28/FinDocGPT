"""
Celery application configuration for asynchronous task processing
"""

import os
from celery import Celery
from .config import get_settings

settings = get_settings()

# Create Celery instance
celery_app = Celery(
    "financial_intelligence",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        "app.tasks.document_tasks",
        "app.tasks.sentiment_tasks", 
        "app.tasks.anomaly_tasks",
        "app.tasks.forecasting_tasks",
        "app.tasks.model_training_tasks",
        "app.tasks.data_processing_tasks",
        "app.tasks.monitoring_tasks"
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    worker_prefetch_multiplier=settings.celery.worker_prefetch_multiplier,
    task_acks_late=settings.celery.task_acks_late,
    worker_max_tasks_per_child=settings.celery.worker_max_tasks_per_child,
    
    # Task routing
    task_routes={
        "app.tasks.document_tasks.*": {"queue": "document_processing"},
        "app.tasks.sentiment_tasks.*": {"queue": "sentiment_analysis"},
        "app.tasks.anomaly_tasks.*": {"queue": "anomaly_detection"},
        "app.tasks.forecasting_tasks.*": {"queue": "forecasting"},
        "app.tasks.model_training_tasks.*": {"queue": "model_training"},
        "app.tasks.data_processing_tasks.*": {"queue": "data_processing"},
        "app.tasks.monitoring_tasks.*": {"queue": "monitoring"},
    },
    
    # Task priorities
    task_default_priority=5,
    worker_disable_rate_limits=False,
    
    # Result expiration
    result_expires=3600,  # 1 hour
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "update-market-data": {
            "task": "app.tasks.data_processing_tasks.update_market_data",
            "schedule": 300.0,  # Every 5 minutes during market hours
        },
        "retrain-models": {
            "task": "app.tasks.model_training_tasks.retrain_models",
            "schedule": 86400.0,  # Daily
        },
        "cleanup-old-results": {
            "task": "app.tasks.monitoring_tasks.cleanup_old_results",
            "schedule": 3600.0,  # Hourly
        },
        "health-check": {
            "task": "app.tasks.monitoring_tasks.system_health_check",
            "schedule": 600.0,  # Every 10 minutes
        },
    },
)

# Task annotations for monitoring
celery_app.conf.task_annotations = {
    "*": {"rate_limit": "100/m"},
    "app.tasks.model_training_tasks.*": {"rate_limit": "10/h"},
    "app.tasks.data_processing_tasks.update_market_data": {"rate_limit": "20/m"},
}