"""
API routes for background task management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from ..services.background_task_service import background_task_service
from ..services.auth_service import get_current_user
from ..models.user_models import User

router = APIRouter(prefix="/background-tasks", tags=["Background Tasks"])


# Request/Response Models

class DocumentProcessingRequest(BaseModel):
    document_data: Dict[str, Any]

class BatchDocumentProcessingRequest(BaseModel):
    documents_data: List[Dict[str, Any]]

class SentimentAnalysisRequest(BaseModel):
    document_id: str

class BatchSentimentAnalysisRequest(BaseModel):
    document_ids: List[str]

class AnomalyDetectionRequest(BaseModel):
    company: str
    metrics: List[str]

class PatternAnomalyDetectionRequest(BaseModel):
    company: str
    data_window: int = 252

class ForecastGenerationRequest(BaseModel):
    ticker: str
    horizons: List[int]

class BatchForecastGenerationRequest(BaseModel):
    tickers: List[str]
    horizons: List[int]

class ModelRetrainingRequest(BaseModel):
    model_name: str
    training_data_path: Optional[str] = None

class ForecastingModelRetrainingRequest(BaseModel):
    ticker: str
    models: List[str]

class MarketDataUpdateRequest(BaseModel):
    tickers: Optional[List[str]] = None

class ExternalDataSyncRequest(BaseModel):
    data_sources: List[str]

class TaskResponse(BaseModel):
    task_id: str
    message: str


# Document Processing Endpoints

@router.post("/document/process", response_model=TaskResponse)
async def process_document(
    request: DocumentProcessingRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous document processing"""
    try:
        task_id = await background_task_service.process_document_async(request.document_data)
        return TaskResponse(
            task_id=task_id,
            message="Document processing task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start document processing: {str(e)}")


@router.post("/document/batch-process", response_model=TaskResponse)
async def batch_process_documents(
    request: BatchDocumentProcessingRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous batch document processing"""
    try:
        task_id = await background_task_service.batch_process_documents_async(request.documents_data)
        return TaskResponse(
            task_id=task_id,
            message="Batch document processing task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch document processing: {str(e)}")


# Sentiment Analysis Endpoints

@router.post("/sentiment/analyze", response_model=TaskResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous sentiment analysis"""
    try:
        task_id = await background_task_service.analyze_sentiment_async(request.document_id)
        return TaskResponse(
            task_id=task_id,
            message="Sentiment analysis task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start sentiment analysis: {str(e)}")


@router.post("/sentiment/batch-analyze", response_model=TaskResponse)
async def batch_analyze_sentiment(
    request: BatchSentimentAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous batch sentiment analysis"""
    try:
        task_id = await background_task_service.batch_sentiment_analysis_async(request.document_ids)
        return TaskResponse(
            task_id=task_id,
            message="Batch sentiment analysis task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch sentiment analysis: {str(e)}")


# Anomaly Detection Endpoints

@router.post("/anomaly/detect", response_model=TaskResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous anomaly detection"""
    try:
        task_id = await background_task_service.detect_anomalies_async(request.company, request.metrics)
        return TaskResponse(
            task_id=task_id,
            message="Anomaly detection task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start anomaly detection: {str(e)}")


@router.post("/anomaly/pattern-detect", response_model=TaskResponse)
async def detect_pattern_anomalies(
    request: PatternAnomalyDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous pattern anomaly detection"""
    try:
        task_id = await background_task_service.pattern_anomaly_detection_async(
            request.company, request.data_window
        )
        return TaskResponse(
            task_id=task_id,
            message="Pattern anomaly detection task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pattern anomaly detection: {str(e)}")


# Forecasting Endpoints

@router.post("/forecast/generate", response_model=TaskResponse)
async def generate_forecast(
    request: ForecastGenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous forecast generation"""
    try:
        task_id = await background_task_service.generate_forecast_async(request.ticker, request.horizons)
        return TaskResponse(
            task_id=task_id,
            message="Forecast generation task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start forecast generation: {str(e)}")


@router.post("/forecast/batch-generate", response_model=TaskResponse)
async def batch_generate_forecasts(
    request: BatchForecastGenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous batch forecast generation"""
    try:
        task_id = await background_task_service.batch_forecast_generation_async(
            request.tickers, request.horizons
        )
        return TaskResponse(
            task_id=task_id,
            message="Batch forecast generation task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch forecast generation: {str(e)}")


# Model Training Endpoints

@router.post("/model/retrain-sentiment", response_model=TaskResponse)
async def retrain_sentiment_model(
    request: ModelRetrainingRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous sentiment model retraining"""
    try:
        task_id = await background_task_service.retrain_sentiment_model_async(
            request.model_name, request.training_data_path
        )
        return TaskResponse(
            task_id=task_id,
            message="Sentiment model retraining task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start sentiment model retraining: {str(e)}")


@router.post("/model/retrain-forecasting", response_model=TaskResponse)
async def retrain_forecasting_models(
    request: ForecastingModelRetrainingRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous forecasting model retraining"""
    try:
        task_id = await background_task_service.retrain_forecasting_models_async(
            request.ticker, request.models
        )
        return TaskResponse(
            task_id=task_id,
            message="Forecasting model retraining task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start forecasting model retraining: {str(e)}")


# Data Processing Endpoints

@router.post("/data/update-market", response_model=TaskResponse)
async def update_market_data(
    request: MarketDataUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous market data update"""
    try:
        task_id = await background_task_service.update_market_data_async(request.tickers)
        return TaskResponse(
            task_id=task_id,
            message="Market data update task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start market data update: {str(e)}")


@router.post("/data/sync-external", response_model=TaskResponse)
async def sync_external_data(
    request: ExternalDataSyncRequest,
    current_user: User = Depends(get_current_user)
):
    """Start asynchronous external data synchronization"""
    try:
        task_id = await background_task_service.sync_external_data_async(request.data_sources)
        return TaskResponse(
            task_id=task_id,
            message="External data sync task started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start external data sync: {str(e)}")


# Task Management Endpoints

@router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a background task"""
    try:
        status = background_task_service.get_task_status(task_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.delete("/task/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a background task"""
    try:
        result = background_task_service.cancel_task(task_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.get("/tasks/active")
async def get_active_tasks(
    current_user: User = Depends(get_current_user)
):
    """Get list of active background tasks"""
    try:
        tasks = background_task_service.get_active_tasks()
        return {"active_tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active tasks: {str(e)}")


@router.get("/workers/stats")
async def get_worker_stats(
    current_user: User = Depends(get_current_user)
):
    """Get Celery worker statistics"""
    try:
        stats = background_task_service.get_worker_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get worker stats: {str(e)}")


@router.get("/queues/lengths")
async def get_queue_lengths(
    current_user: User = Depends(get_current_user)
):
    """Get queue lengths for different task queues"""
    try:
        lengths = background_task_service.get_queue_lengths()
        return {"queue_lengths": lengths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue lengths: {str(e)}")


@router.get("/health")
async def background_tasks_health_check():
    """Health check for background task system"""
    try:
        health = await background_task_service.health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")