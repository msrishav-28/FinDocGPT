"""
Background task service for asynchronous processing

This service provides:
- Asynchronous model training and data processing pipelines
- Task queue management and monitoring
- Integration with Celery for distributed task execution
"""

import logging
from typing import Dict, Any, List, Optional, Union
from celery.result import AsyncResult
from ..celery_app import celery_app
from ..tasks import (
    document_tasks,
    sentiment_tasks,
    anomaly_tasks,
    forecasting_tasks,
    model_training_tasks,
    data_processing_tasks,
    monitoring_tasks
)

logger = logging.getLogger(__name__)


class BackgroundTaskService:
    """Service for managing background tasks and asynchronous processing"""
    
    def __init__(self):
        self.celery_app = celery_app
    
    # Document Processing Tasks
    
    async def process_document_async(self, document_data: Dict[str, Any]) -> str:
        """
        Process a document asynchronously
        
        Args:
            document_data: Document content and metadata
            
        Returns:
            Task ID for tracking
        """
        try:
            task = document_tasks.process_document.delay(document_data)
            logger.info(f"Started document processing task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start document processing task: {str(e)}")
            raise
    
    async def batch_process_documents_async(self, documents_data: List[Dict[str, Any]]) -> str:
        """
        Process multiple documents in batch asynchronously
        
        Args:
            documents_data: List of document data
            
        Returns:
            Task ID for tracking
        """
        try:
            task = document_tasks.batch_process_documents.delay(documents_data)
            logger.info(f"Started batch document processing task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start batch document processing task: {str(e)}")
            raise
    
    # Sentiment Analysis Tasks
    
    async def analyze_sentiment_async(self, document_id: str) -> str:
        """
        Analyze document sentiment asynchronously
        
        Args:
            document_id: Document identifier
            
        Returns:
            Task ID for tracking
        """
        try:
            task = sentiment_tasks.analyze_document_sentiment.delay(document_id)
            logger.info(f"Started sentiment analysis task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start sentiment analysis task: {str(e)}")
            raise
    
    async def batch_sentiment_analysis_async(self, document_ids: List[str]) -> str:
        """
        Perform batch sentiment analysis asynchronously
        
        Args:
            document_ids: List of document identifiers
            
        Returns:
            Task ID for tracking
        """
        try:
            task = sentiment_tasks.batch_sentiment_analysis.delay(document_ids)
            logger.info(f"Started batch sentiment analysis task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start batch sentiment analysis task: {str(e)}")
            raise
    
    # Anomaly Detection Tasks
    
    async def detect_anomalies_async(self, company: str, metrics: List[str]) -> str:
        """
        Detect anomalies asynchronously
        
        Args:
            company: Company ticker symbol
            metrics: List of metrics to analyze
            
        Returns:
            Task ID for tracking
        """
        try:
            task = anomaly_tasks.detect_metric_anomalies.delay(company, metrics)
            logger.info(f"Started anomaly detection task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start anomaly detection task: {str(e)}")
            raise
    
    async def pattern_anomaly_detection_async(self, company: str, data_window: int = 252) -> str:
        """
        Detect pattern anomalies asynchronously
        
        Args:
            company: Company ticker symbol
            data_window: Number of days to analyze
            
        Returns:
            Task ID for tracking
        """
        try:
            task = anomaly_tasks.pattern_anomaly_detection.delay(company, data_window)
            logger.info(f"Started pattern anomaly detection task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start pattern anomaly detection task: {str(e)}")
            raise
    
    # Forecasting Tasks
    
    async def generate_forecast_async(self, ticker: str, horizons: List[int]) -> str:
        """
        Generate stock forecast asynchronously
        
        Args:
            ticker: Stock ticker symbol
            horizons: List of forecast horizons
            
        Returns:
            Task ID for tracking
        """
        try:
            task = forecasting_tasks.generate_stock_forecast.delay(ticker, horizons)
            logger.info(f"Started forecast generation task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start forecast generation task: {str(e)}")
            raise
    
    async def batch_forecast_generation_async(self, tickers: List[str], horizons: List[int]) -> str:
        """
        Generate forecasts for multiple stocks asynchronously
        
        Args:
            tickers: List of stock ticker symbols
            horizons: List of forecast horizons
            
        Returns:
            Task ID for tracking
        """
        try:
            task = forecasting_tasks.batch_forecast_generation.delay(tickers, horizons)
            logger.info(f"Started batch forecast generation task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start batch forecast generation task: {str(e)}")
            raise
    
    # Model Training Tasks
    
    async def retrain_sentiment_model_async(self, model_name: str, training_data_path: Optional[str] = None) -> str:
        """
        Retrain sentiment model asynchronously
        
        Args:
            model_name: Name of the model to retrain
            training_data_path: Optional path to training data
            
        Returns:
            Task ID for tracking
        """
        try:
            task = model_training_tasks.retrain_sentiment_model.delay(model_name, training_data_path)
            logger.info(f"Started sentiment model retraining task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start sentiment model retraining task: {str(e)}")
            raise
    
    async def retrain_forecasting_models_async(self, ticker: str, models: List[str]) -> str:
        """
        Retrain forecasting models asynchronously
        
        Args:
            ticker: Stock ticker symbol
            models: List of model names to retrain
            
        Returns:
            Task ID for tracking
        """
        try:
            task = model_training_tasks.retrain_forecasting_models.delay(ticker, models)
            logger.info(f"Started forecasting model retraining task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start forecasting model retraining task: {str(e)}")
            raise
    
    # Data Processing Tasks
    
    async def update_market_data_async(self, tickers: Optional[List[str]] = None) -> str:
        """
        Update market data asynchronously
        
        Args:
            tickers: Optional list of tickers to update
            
        Returns:
            Task ID for tracking
        """
        try:
            task = data_processing_tasks.update_market_data.delay(tickers)
            logger.info(f"Started market data update task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start market data update task: {str(e)}")
            raise
    
    async def sync_external_data_async(self, data_sources: List[str]) -> str:
        """
        Synchronize external data asynchronously
        
        Args:
            data_sources: List of data sources to sync
            
        Returns:
            Task ID for tracking
        """
        try:
            task = data_processing_tasks.sync_external_data.delay(data_sources)
            logger.info(f"Started external data sync task: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to start external data sync task: {str(e)}")
            raise
    
    # Task Management and Monitoring
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a background task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "info": result.info,
                "ready": result.ready(),
                "successful": result.successful() if result.ready() else None,
                "failed": result.failed() if result.ready() else None
            }
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {str(e)}")
            return {
                "task_id": task_id,
                "status": "UNKNOWN",
                "error": str(e)
            }
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a background task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Cancellation result
        """
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Cancelled task: {task_id}")
            return {
                "task_id": task_id,
                "cancelled": True
            }
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return {
                "task_id": task_id,
                "cancelled": False,
                "error": str(e)
            }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of active tasks
        
        Returns:
            List of active task information
        """
        try:
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return []
            
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    all_tasks.append({
                        "worker": worker,
                        "task_id": task.get("id"),
                        "name": task.get("name"),
                        "args": task.get("args"),
                        "kwargs": task.get("kwargs"),
                        "time_start": task.get("time_start")
                    })
            
            return all_tasks
        except Exception as e:
            logger.error(f"Failed to get active tasks: {str(e)}")
            return []
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get Celery worker statistics
        
        Returns:
            Worker statistics
        """
        try:
            inspect = self.celery_app.control.inspect()
            stats = inspect.stats()
            
            return {
                "workers": stats or {},
                "total_workers": len(stats) if stats else 0
            }
        except Exception as e:
            logger.error(f"Failed to get worker stats: {str(e)}")
            return {
                "workers": {},
                "total_workers": 0,
                "error": str(e)
            }
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """
        Get queue lengths for different task queues
        
        Returns:
            Dictionary of queue names and their lengths
        """
        try:
            inspect = self.celery_app.control.inspect()
            reserved = inspect.reserved()
            
            queue_lengths = {}
            if reserved:
                for worker, tasks in reserved.items():
                    for task in tasks:
                        queue = task.get("delivery_info", {}).get("routing_key", "default")
                        queue_lengths[queue] = queue_lengths.get(queue, 0) + 1
            
            return queue_lengths
        except Exception as e:
            logger.error(f"Failed to get queue lengths: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the background task system
        
        Returns:
            Health check results
        """
        try:
            # Check if Celery is responsive
            inspect = self.celery_app.control.inspect()
            stats = inspect.stats()
            
            # Check Redis connection (broker)
            from ..services.cache_service import cache_service
            redis_health = await cache_service.health_check()
            
            return {
                "celery_workers": len(stats) if stats else 0,
                "redis_broker": redis_health,
                "status": "healthy" if stats and redis_health.get("status") == "healthy" else "unhealthy"
            }
        except Exception as e:
            logger.error(f"Background task health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instance
background_task_service = BackgroundTaskService()