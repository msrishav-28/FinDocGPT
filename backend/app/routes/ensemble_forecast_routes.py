"""
API routes for ensemble forecasting functionality.
Provides endpoints for advanced forecasting, model performance monitoring,
and uncertainty quantification.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..services.forecast_service import get_ensemble_forecast, forecast_prices_advanced
from ..services.ensemble_forecasting_service import get_ensemble_forecasting_service, ModelType
from ..services.data_integration_service import data_integration_service
from ..services.model_performance_service import get_model_performance_service

router = APIRouter(prefix="/ensemble", tags=["ensemble_forecasting"])


class ForecastRequest(BaseModel):
    """Request model for ensemble forecasting"""
    ticker: str
    horizons: Optional[List[int]] = [30, 90, 180, 365]
    use_cache: bool = True


class ModelPerformanceRequest(BaseModel):
    """Request model for model performance tracking"""
    ticker: str
    model_type: Optional[str] = None
    actual_values: Dict[int, float]  # horizon -> actual_value


@router.get("/forecast/{ticker}")
async def get_stock_forecast(
    ticker: str,
    horizons: Optional[str] = Query(None, description="Comma-separated list of forecast horizons in days"),
    include_confidence: bool = Query(True, description="Include confidence intervals"),
    include_performance: bool = Query(True, description="Include model performance metrics")
):
    """
    Get ensemble forecast for a stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        horizons: Comma-separated forecast horizons in days (default: 30,90,180,365)
        include_confidence: Whether to include confidence intervals
        include_performance: Whether to include model performance metrics
    
    Returns:
        Comprehensive forecast with predictions, confidence intervals, and metadata
    """
    try:
        # Parse horizons
        if horizons:
            horizon_list = [int(h.strip()) for h in horizons.split(',')]
        else:
            horizon_list = [30, 90, 180, 365]
        
        # Validate horizons
        if not all(1 <= h <= 365 for h in horizon_list):
            raise HTTPException(status_code=400, detail="Horizons must be between 1 and 365 days")
        
        # Get ensemble forecast
        forecast_result = await get_ensemble_forecast(ticker, horizon_list)
        
        # Optionally remove confidence intervals or performance data
        if not include_confidence:
            forecast_result.pop('confidence_intervals', None)
        
        if not include_performance:
            forecast_result.pop('performance_summary', None)
        
        return JSONResponse(content=forecast_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@router.post("/forecast")
async def create_ensemble_forecast(request: ForecastRequest):
    """
    Create ensemble forecast with custom parameters.
    
    Args:
        request: Forecast request with ticker, horizons, and options
    
    Returns:
        Detailed ensemble forecast results
    """
    try:
        forecast_result = await get_ensemble_forecast(request.ticker, request.horizons)
        return JSONResponse(content=forecast_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast creation failed: {str(e)}")


@router.get("/forecast/{ticker}/simple")
async def get_simple_forecast(
    ticker: str,
    periods: int = Query(7, ge=1, le=365, description="Number of days to forecast")
):
    """
    Get simple forecast for backward compatibility.
    
    Args:
        ticker: Stock ticker symbol
        periods: Number of days to forecast (1-365)
    
    Returns:
        Simple forecast in legacy format
    """
    try:
        df_forecast = await forecast_prices_advanced(ticker, periods, use_ensemble=True)
        return JSONResponse(content=df_forecast.to_dict(orient='list'))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple forecast failed: {str(e)}")


@router.get("/models/performance")
async def get_model_performance(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    model_type: Optional[str] = Query(None, description="Filter by model type (prophet, arima, lstm)")
):
    """
    Get model performance summary.
    
    Args:
        ticker: Optional ticker filter
        model_type: Optional model type filter
    
    Returns:
        Model performance metrics and reliability scores
    """
    try:
        performance_service = get_model_performance_service(data_integration_service)
        
        # Parse model type
        mt = None
        if model_type:
            try:
                mt = ModelType(model_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
        
        performance_summary = performance_service.get_model_performance_summary(
            symbol=ticker, model_type=mt
        )
        
        return JSONResponse(content=performance_summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance retrieval failed: {str(e)}")


@router.post("/models/performance/update")
async def update_model_performance(request: ModelPerformanceRequest):
    """
    Update model performance with actual values.
    
    Args:
        request: Performance update request with ticker, model type, and actual values
    
    Returns:
        Updated performance metrics
    """
    try:
        performance_service = get_model_performance_service(data_integration_service)
        
        # Parse model type if provided
        mt = None
        if request.model_type:
            try:
                mt = ModelType(request.model_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
        
        # Create a dummy forecast for tracking (in real implementation, you'd retrieve the actual forecast)
        from ..services.ensemble_forecasting_service import EnsembleForecast
        dummy_forecast = EnsembleForecast(
            symbol=request.ticker,
            forecast_date=datetime.now(),
            horizons=request.actual_values,  # This is simplified
            confidence_intervals={},
            model_weights={},
            individual_forecasts={},
            ensemble_confidence=0.8,
            quality_score=0.8
        )
        
        # Track forecast accuracy
        await performance_service.track_forecast_accuracy(dummy_forecast, request.actual_values)
        
        # Get updated performance summary
        performance_summary = performance_service.get_model_performance_summary(
            symbol=request.ticker, model_type=mt
        )
        
        return JSONResponse(content=performance_summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance update failed: {str(e)}")


@router.get("/models/alerts")
async def get_performance_alerts(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high, critical)"),
    days: int = Query(7, ge=1, le=30, description="Number of days to look back for alerts")
):
    """
    Get recent performance alerts.
    
    Args:
        ticker: Optional ticker filter
        severity: Optional severity filter
        days: Number of days to look back
    
    Returns:
        List of performance alerts
    """
    try:
        performance_service = get_model_performance_service(data_integration_service)
        
        # Get alerts from the last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [
            {
                'model_type': alert.model_type.value,
                'symbol': alert.symbol,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'metrics': alert.metrics,
                'timestamp': alert.timestamp.isoformat(),
                'requires_retraining': alert.requires_retraining
            }
            for alert in performance_service.alert_history
            if alert.timestamp >= cutoff_date
        ]
        
        # Apply filters
        if ticker:
            recent_alerts = [a for a in recent_alerts if a['symbol'] == ticker]
        
        if severity:
            recent_alerts = [a for a in recent_alerts if a['severity'] == severity]
        
        return JSONResponse(content={'alerts': recent_alerts, 'count': len(recent_alerts)})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")


@router.get("/models/retraining")
async def get_retraining_recommendations():
    """
    Get model retraining recommendations.
    
    Returns:
        List of retraining recommendations sorted by priority
    """
    try:
        performance_service = get_model_performance_service(data_integration_service)
        recommendations = performance_service.generate_retraining_recommendations()
        
        recommendations_data = [
            {
                'model_type': rec.model_type.value,
                'symbol': rec.symbol,
                'priority': rec.priority,
                'reason': rec.reason,
                'performance_degradation': rec.performance_degradation,
                'data_freshness_score': rec.data_freshness_score,
                'recommended_action': rec.recommended_action,
                'estimated_improvement': rec.estimated_improvement
            }
            for rec in recommendations
        ]
        
        return JSONResponse(content={
            'recommendations': recommendations_data,
            'count': len(recommendations_data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining recommendations failed: {str(e)}")


@router.post("/models/train/{ticker}")
async def train_models(
    ticker: str,
    model_types: Optional[str] = Query(None, description="Comma-separated model types to train"),
    start_date: Optional[str] = Query(None, description="Training start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Training end date (YYYY-MM-DD)")
):
    """
    Train forecasting models for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        model_types: Comma-separated model types (prophet, arima, lstm)
        start_date: Training data start date
        end_date: Training data end date
    
    Returns:
        Training results for each model
    """
    try:
        ensemble_service = get_ensemble_forecasting_service(data_integration_service)
        
        # Parse dates
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = datetime.now() - timedelta(days=365 * 2)  # 2 years default
        
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime.now()
        
        # Train models
        training_results = await ensemble_service.train_models(ticker, start_dt, end_dt)
        
        # Convert results to serializable format
        results_data = {
            mt.value: success for mt, success in training_results.items()
        }
        
        return JSONResponse(content={
            'ticker': ticker,
            'training_period': {
                'start_date': start_dt.isoformat(),
                'end_date': end_dt.isoformat()
            },
            'results': results_data,
            'successful_models': sum(results_data.values()),
            'total_models': len(results_data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for ensemble forecasting service.
    
    Returns:
        Service health status and available models
    """
    try:
        ensemble_service = get_ensemble_forecasting_service(data_integration_service)
        performance_service = get_model_performance_service(data_integration_service)
        
        # Check available models
        available_models = []
        for model_type, forecaster in ensemble_service.forecasters.items():
            available_models.append({
                'type': model_type.value,
                'trained': forecaster.is_trained,
                'available': True
            })
        
        return JSONResponse(content={
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'available_models': available_models,
            'data_integration': 'available',
            'performance_tracking': 'available'
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )