"""
Comprehensive logging configuration for the Financial Intelligence System.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
import os

from app.config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        if hasattr(record, 'method'):
            log_entry['method'] = record.method
        if hasattr(record, 'path'):
            log_entry['path'] = record.path
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record."""
        # Add environment information
        record.environment = settings.ENVIRONMENT
        record.service = "financial-intelligence-backend"
        
        return True


def setup_logging() -> None:
    """Setup comprehensive logging configuration."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with JSON formatting for production
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.ENVIRONMENT == "production":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)
    
    # File handlers for different log levels
    if settings.ENVIRONMENT != "test":
        # Application log file (all levels)
        app_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        app_handler.setFormatter(JSONFormatter())
        app_handler.addFilter(ContextFilter())
        root_logger.addHandler(app_handler)
        
        # Error log file (errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        error_handler.addFilter(ContextFilter())
        root_logger.addHandler(error_handler)
        
        # Access log file (for HTTP requests)
        access_handler = logging.handlers.RotatingFileHandler(
            log_dir / "access.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        access_handler.setFormatter(JSONFormatter())
        access_handler.addFilter(ContextFilter())
        
        # Create access logger
        access_logger = logging.getLogger("access")
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """Log HTTP request information."""
    access_logger = logging.getLogger("access")
    
    extra = {
        'method': method,
        'path': path,
        'status_code': status_code,
        'duration': duration,
    }
    
    if user_id:
        extra['user_id'] = user_id
    if request_id:
        extra['request_id'] = request_id
    
    access_logger.info(
        f"{method} {path} - {status_code} - {duration:.3f}s",
        extra=extra
    )


def log_business_event(
    event_type: str,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None
) -> None:
    """Log business events for analytics and auditing."""
    business_logger = logging.getLogger("business")
    
    extra = {
        'event_type': event_type,
        'event_data': event_data,
    }
    
    if user_id:
        extra['user_id'] = user_id
    
    business_logger.info(
        f"Business event: {event_type}",
        extra=extra
    )


def log_security_event(
    event_type: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """Log security-related events."""
    security_logger = logging.getLogger("security")
    
    extra = {
        'event_type': event_type,
        'details': details,
    }
    
    if user_id:
        extra['user_id'] = user_id
    if ip_address:
        extra['ip_address'] = ip_address
    
    security_logger.warning(
        f"Security event: {event_type}",
        extra=extra
    )


def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str = "ms",
    tags: Optional[Dict[str, str]] = None
) -> None:
    """Log performance metrics."""
    metrics_logger = logging.getLogger("metrics")
    
    extra = {
        'metric_name': metric_name,
        'value': value,
        'unit': unit,
        'tags': tags or {}
    }
    
    metrics_logger.info(
        f"Metric: {metric_name} = {value} {unit}",
        extra=extra
    )