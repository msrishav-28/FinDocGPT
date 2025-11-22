"""
Application metrics collection and monitoring.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import json

from app.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    unit: str = "count"


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # System metrics collection
        self._system_metrics_task = None
        self._start_system_metrics_collection()
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self.counters[key] += value
            
            metric = MetricPoint(
                name=name,
                value=self.counters[key],
                timestamp=datetime.utcnow(),
                tags=tags or {},
                unit="count"
            )
            self.metrics[name].append(metric)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = "count") -> None:
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
            
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                unit=unit
            )
            self.metrics[name].append(metric)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = "ms") -> None:
        """Record a histogram metric value."""
        with self._lock:
            key = self._make_key(name, tags)
            self.histograms[key].append(value)
            
            # Keep only last 1000 values for memory efficiency
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                unit=unit
            )
            self.metrics[name].append(metric)
    
    def timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.histogram(name, duration * 1000, tags, "ms")  # Convert to milliseconds
    
    def get_metrics(self, name: Optional[str] = None, since: Optional[datetime] = None) -> Dict[str, List[Dict]]:
        """Get collected metrics."""
        with self._lock:
            result = {}
            
            metrics_to_check = [name] if name else self.metrics.keys()
            
            for metric_name in metrics_to_check:
                if metric_name in self.metrics:
                    points = []
                    for point in self.metrics[metric_name]:
                        if since is None or point.timestamp >= since:
                            points.append(asdict(point))
                    result[metric_name] = points
            
            return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate histogram statistics
            for key, values in self.histograms.items():
                if values:
                    summary["histograms"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            
            return summary
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        
        return sorted_values[index]
    
    def _start_system_metrics_collection(self) -> None:
        """Start collecting system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.gauge("system.cpu.usage", cpu_percent, unit="percent")
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.gauge("system.memory.usage", memory.percent, unit="percent")
                    self.gauge("system.memory.available", memory.available, unit="bytes")
                    self.gauge("system.memory.used", memory.used, unit="bytes")
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.gauge("system.disk.usage", (disk.used / disk.total) * 100, unit="percent")
                    self.gauge("system.disk.free", disk.free, unit="bytes")
                    
                    # Network metrics
                    network = psutil.net_io_counters()
                    self.gauge("system.network.bytes_sent", network.bytes_sent, unit="bytes")
                    self.gauge("system.network.bytes_recv", network.bytes_recv, unit="bytes")
                    
                    # Process metrics
                    process = psutil.Process()
                    self.gauge("process.memory.rss", process.memory_info().rss, unit="bytes")
                    self.gauge("process.memory.vms", process.memory_info().vms, unit="bytes")
                    self.gauge("process.cpu.percent", process.cpu_percent(), unit="percent")
                    self.gauge("process.threads", process.num_threads(), unit="count")
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()


# Global metrics collector instance
metrics = MetricsCollector()


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            metrics.timing(self.metric_name, duration, self.tags)


def timer(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def async_timer(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Async context manager for timing operations."""
    class AsyncTimer:
        def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
            self.name = name
            self.tags = tags
            self.start_time = None
        
        async def __aenter__(self):
            self.start_time = time.time()
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                metrics.timing(self.name, duration, self.tags)
    
    return AsyncTimer(metric_name, tags)


def async_timer_decorator(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with async_timer(metric_name, tags):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Business metrics helpers
def track_user_action(action: str, user_id: str, success: bool = True) -> None:
    """Track user actions for analytics."""
    tags = {"action": action, "user_id": user_id, "success": str(success)}
    metrics.increment("user.action", tags=tags)


def track_api_request(endpoint: str, method: str, status_code: int, duration: float) -> None:
    """Track API request metrics."""
    tags = {"endpoint": endpoint, "method": method, "status": str(status_code)}
    metrics.increment("api.requests", tags=tags)
    metrics.timing("api.response_time", duration, tags)


def track_model_prediction(model_name: str, prediction_type: str, duration: float, success: bool = True) -> None:
    """Track ML model prediction metrics."""
    tags = {"model": model_name, "type": prediction_type, "success": str(success)}
    metrics.increment("ml.predictions", tags=tags)
    metrics.timing("ml.prediction_time", duration, tags)


def track_data_processing(operation: str, records_processed: int, duration: float, success: bool = True) -> None:
    """Track data processing metrics."""
    tags = {"operation": operation, "success": str(success)}
    metrics.increment("data.operations", tags=tags)
    metrics.gauge("data.records_processed", records_processed, tags)
    metrics.timing("data.processing_time", duration, tags)