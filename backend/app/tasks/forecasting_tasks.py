"""
Forecasting tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List
from celery import current_task
from ..celery_app import celery_app
from ..services.ensemble_forecasting_service import EnsembleForecastingService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.forecasting_tasks.generate_stock_forecast")
def generate_stock_forecast(self, ticker: str, horizons: List[int]) -> Dict[str, Any]:
    """
    Generate stock price forecasts asynchronously
    
    Args:
        ticker: Stock ticker symbol
        horizons: List of forecast horizons in days
        
    Returns:
        Dictionary with forecast results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting stock forecast generation"})
        
        forecasting_service = EnsembleForecastingService()
        
        # Generate forecasts for each horizon
        forecasts = {}
        confidence_intervals = {}
        
        for i, horizon in enumerate(horizons):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": f"Generating {horizon}-day forecast ({i+1}/{len(horizons)})",
                    "progress": (i / len(horizons)) * 100
                }
            )
            
            forecast_result = forecasting_service.forecast_stock_price(ticker, horizon)
            forecasts[horizon] = forecast_result.predicted_value
            confidence_intervals[horizon] = (
                forecast_result.confidence_lower,
                forecast_result.confidence_upper
            )
        
        # Get model performance metrics
        self.update_state(state="PROCESSING", meta={"status": "Calculating performance metrics"})
        performance_metrics = forecasting_service.get_model_performance(ticker)
        
        return {
            "ticker": ticker,
            "forecasts": forecasts,
            "confidence_intervals": confidence_intervals,
            "performance_metrics": performance_metrics,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Stock forecast generation failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Forecast generation failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.forecasting_tasks.batch_forecast_generation")
def batch_forecast_generation(self, tickers: List[str], horizons: List[int]) -> Dict[str, Any]:
    """
    Generate forecasts for multiple stocks
    
    Args:
        tickers: List of stock ticker symbols
        horizons: List of forecast horizons in days
        
    Returns:
        Dictionary with batch forecast results
    """
    try:
        total_tickers = len(tickers)
        results = []
        failed_forecasts = []
        
        for i, ticker in enumerate(tickers):
            try:
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Generating forecasts for {ticker} ({i+1}/{total_tickers})",
                        "progress": (i / total_tickers) * 100
                    }
                )
                
                # Generate forecast for individual ticker
                result = generate_stock_forecast.apply_async(args=[ticker, horizons]).get()
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to generate forecast for {ticker}: {str(e)}")
                failed_forecasts.append({"ticker": ticker, "error": str(e)})
        
        return {
            "total_tickers": total_tickers,
            "successful_forecasts": len(results),
            "failed_forecasts": len(failed_forecasts),
            "results": results,
            "failed_tickers": failed_forecasts,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Batch forecast generation failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Batch forecast generation failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.forecasting_tasks.update_forecast_models")
def update_forecast_models(self, ticker: str) -> Dict[str, Any]:
    """
    Update forecasting models with latest data
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with model update results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Updating forecast models"})
        
        forecasting_service = EnsembleForecastingService()
        
        # Update models with latest data
        update_results = forecasting_service.update_models(ticker)
        
        return {
            "ticker": ticker,
            "update_results": update_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Forecast model update failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Model update failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.forecasting_tasks.evaluate_forecast_accuracy")
def evaluate_forecast_accuracy(self, ticker: str, days_back: int = 30) -> Dict[str, Any]:
    """
    Evaluate forecast accuracy against actual values
    
    Args:
        ticker: Stock ticker symbol
        days_back: Number of days to evaluate
        
    Returns:
        Dictionary with accuracy evaluation results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Evaluating forecast accuracy"})
        
        forecasting_service = EnsembleForecastingService()
        
        # Evaluate accuracy
        accuracy_metrics = forecasting_service.evaluate_forecast_accuracy(ticker, days_back)
        
        return {
            "ticker": ticker,
            "evaluation_period": days_back,
            "accuracy_metrics": accuracy_metrics,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Forecast accuracy evaluation failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Accuracy evaluation failed"}
        )
        raise