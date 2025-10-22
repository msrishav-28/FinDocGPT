"""
Model training tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List, Optional
from celery import current_task
from ..celery_app import celery_app
from ..services.model_performance_service import ModelPerformanceService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.model_training_tasks.retrain_sentiment_model")
def retrain_sentiment_model(self, model_name: str, training_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrain sentiment analysis model asynchronously
    
    Args:
        model_name: Name of the sentiment model to retrain
        training_data_path: Optional path to new training data
        
    Returns:
        Dictionary with retraining results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting sentiment model retraining"})
        
        model_service = ModelPerformanceService()
        
        # Load training data
        self.update_state(state="PROCESSING", meta={"status": "Loading training data"})
        training_data = model_service.load_sentiment_training_data(training_data_path)
        
        # Retrain model
        self.update_state(state="PROCESSING", meta={"status": "Retraining model"})
        training_results = model_service.retrain_sentiment_model(model_name, training_data)
        
        # Evaluate model performance
        self.update_state(state="PROCESSING", meta={"status": "Evaluating model performance"})
        performance_metrics = model_service.evaluate_sentiment_model(model_name)
        
        # Deploy model if performance is satisfactory
        if performance_metrics.get("accuracy", 0) > 0.85:
            self.update_state(state="PROCESSING", meta={"status": "Deploying updated model"})
            deployment_result = model_service.deploy_sentiment_model(model_name)
        else:
            deployment_result = {"deployed": False, "reason": "Performance below threshold"}
        
        return {
            "model_name": model_name,
            "training_results": training_results,
            "performance_metrics": performance_metrics,
            "deployment_result": deployment_result,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Sentiment model retraining failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Model retraining failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.model_training_tasks.retrain_forecasting_models")
def retrain_forecasting_models(self, ticker: str, models: List[str]) -> Dict[str, Any]:
    """
    Retrain forecasting models for a specific ticker
    
    Args:
        ticker: Stock ticker symbol
        models: List of model names to retrain
        
    Returns:
        Dictionary with retraining results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting forecasting model retraining"})
        
        model_service = ModelPerformanceService()
        results = {}
        
        for i, model_name in enumerate(models):
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": f"Retraining {model_name} ({i+1}/{len(models)})",
                    "progress": (i / len(models)) * 100
                }
            )
            
            try:
                # Retrain individual model
                training_result = model_service.retrain_forecasting_model(ticker, model_name)
                
                # Evaluate performance
                performance = model_service.evaluate_forecasting_model(ticker, model_name)
                
                results[model_name] = {
                    "training_result": training_result,
                    "performance": performance,
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"Failed to retrain {model_name}: {str(e)}")
                results[model_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {
            "ticker": ticker,
            "models": models,
            "results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Forecasting model retraining failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Model retraining failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.model_training_tasks.retrain_anomaly_models")
def retrain_anomaly_models(self, company: str) -> Dict[str, Any]:
    """
    Retrain anomaly detection models for a company
    
    Args:
        company: Company ticker symbol
        
    Returns:
        Dictionary with retraining results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting anomaly model retraining"})
        
        model_service = ModelPerformanceService()
        
        # Retrain anomaly detection models
        self.update_state(state="PROCESSING", meta={"status": "Retraining anomaly models"})
        training_results = model_service.retrain_anomaly_models(company)
        
        # Update baselines
        self.update_state(state="PROCESSING", meta={"status": "Updating detection baselines"})
        baseline_results = model_service.update_anomaly_baselines(company)
        
        return {
            "company": company,
            "training_results": training_results,
            "baseline_results": baseline_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Anomaly model retraining failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Model retraining failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.model_training_tasks.retrain_models")
def retrain_models(self) -> Dict[str, Any]:
    """
    Periodic task to retrain all models
    
    Returns:
        Dictionary with overall retraining results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting periodic model retraining"})
        
        model_service = ModelPerformanceService()
        
        # Get list of models that need retraining
        models_to_retrain = model_service.get_models_needing_retraining()
        
        results = {
            "sentiment_models": [],
            "forecasting_models": [],
            "anomaly_models": [],
            "total_retrained": 0,
            "total_failed": 0
        }
        
        # Retrain sentiment models
        for model_info in models_to_retrain.get("sentiment", []):
            try:
                result = retrain_sentiment_model.apply_async(
                    args=[model_info["name"]]
                ).get()
                results["sentiment_models"].append(result)
                results["total_retrained"] += 1
            except Exception as e:
                logger.error(f"Failed to retrain sentiment model {model_info['name']}: {str(e)}")
                results["total_failed"] += 1
        
        # Retrain forecasting models
        for model_info in models_to_retrain.get("forecasting", []):
            try:
                result = retrain_forecasting_models.apply_async(
                    args=[model_info["ticker"], model_info["models"]]
                ).get()
                results["forecasting_models"].append(result)
                results["total_retrained"] += len(model_info["models"])
            except Exception as e:
                logger.error(f"Failed to retrain forecasting models for {model_info['ticker']}: {str(e)}")
                results["total_failed"] += len(model_info["models"])
        
        # Retrain anomaly models
        for model_info in models_to_retrain.get("anomaly", []):
            try:
                result = retrain_anomaly_models.apply_async(
                    args=[model_info["company"]]
                ).get()
                results["anomaly_models"].append(result)
                results["total_retrained"] += 1
            except Exception as e:
                logger.error(f"Failed to retrain anomaly models for {model_info['company']}: {str(e)}")
                results["total_failed"] += 1
        
        return {
            "results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Periodic model retraining failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Periodic retraining failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.model_training_tasks.optimize_model_hyperparameters")
def optimize_model_hyperparameters(self, model_type: str, model_name: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a specific model
    
    Args:
        model_type: Type of model (sentiment, forecasting, anomaly)
        model_name: Name of the model
        ticker: Optional ticker symbol for forecasting models
        
    Returns:
        Dictionary with optimization results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting hyperparameter optimization"})
        
        model_service = ModelPerformanceService()
        
        # Perform hyperparameter optimization
        optimization_results = model_service.optimize_hyperparameters(
            model_type, model_name, ticker
        )
        
        return {
            "model_type": model_type,
            "model_name": model_name,
            "ticker": ticker,
            "optimization_results": optimization_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Hyperparameter optimization failed"}
        )
        raise