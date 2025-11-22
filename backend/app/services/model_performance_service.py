"""
Model Performance Tracking and Uncertainty Quantification Service.
Implements confidence interval calculation, model performance monitoring,
automatic retraining, and forecast accuracy tracking.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
from collections import defaultdict, deque
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

from .ensemble_forecasting_service import ModelType, ForecastResult, EnsembleForecast, ModelPerformance
from .data_integration_service import DataIntegrationService, DataPoint

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    MAE = "mean_absolute_error"
    RMSE = "root_mean_squared_error"
    MAPE = "mean_absolute_percentage_error"
    SMAPE = "symmetric_mean_absolute_percentage_error"
    ACCURACY = "directional_accuracy"
    SHARPE = "sharpe_ratio"


class UncertaintyMethod(Enum):
    BOOTSTRAP = "bootstrap"
    QUANTILE_REGRESSION = "quantile_regression"
    BAYESIAN = "bayesian"
    ENSEMBLE_VARIANCE = "ensemble_variance"


@dataclass
class ConfidenceInterval:
    """Confidence interval with multiple confidence levels"""
    lower_50: float  # 25th percentile
    upper_50: float  # 75th percentile
    lower_80: float  # 10th percentile
    upper_80: float  # 90th percentile
    lower_95: float  # 2.5th percentile
    upper_95: float  # 97.5th percentile
    method: UncertaintyMethod
    uncertainty_score: float  # Overall uncertainty measure


@dataclass
class ModelReliabilityScore:
    """Model reliability assessment"""
    model_type: ModelType
    symbol: str
    overall_score: float  # 0-1, higher is better
    consistency_score: float  # How consistent predictions are
    accuracy_trend: float  # Trend in accuracy over time
    prediction_stability: float  # Stability of predictions
    data_quality_impact: float  # Impact of data quality on performance
    last_assessment: datetime


@dataclass
class PerformanceAlert:
    """Alert for performance degradation"""
    model_type: ModelType
    symbol: str
    alert_type: str  # 'accuracy_drop', 'high_uncertainty', 'data_quality'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: Dict[str, float]
    timestamp: datetime
    requires_retraining: bool


@dataclass
class RetrainingRecommendation:
    """Recommendation for model retraining"""
    model_type: ModelType
    symbol: str
    priority: str  # 'low', 'medium', 'high', 'urgent'
    reason: str
    performance_degradation: float
    data_freshness_score: float
    recommended_action: str
    estimated_improvement: float


class UncertaintyQuantifier:
    """Uncertainty quantification for forecasting models"""
    
    def __init__(self):
        self.bootstrap_samples = 1000
        self.confidence_levels = [0.5, 0.8, 0.95]
    
    def calculate_bootstrap_intervals(self, predictions: List[float], 
                                    residuals: List[float]) -> ConfidenceInterval:
        """Calculate confidence intervals using bootstrap method"""
        try:
            if len(residuals) < 10:
                # Fallback to simple standard deviation method
                return self._fallback_intervals(predictions, residuals)
            
            # Bootstrap resampling
            bootstrap_predictions = []
            for _ in range(self.bootstrap_samples):
                # Resample residuals with replacement
                resampled_residuals = np.random.choice(residuals, size=len(predictions), replace=True)
                bootstrap_pred = np.array(predictions) + resampled_residuals
                bootstrap_predictions.append(bootstrap_pred)
            
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate percentiles
            percentiles = {}
            for conf_level in self.confidence_levels:
                alpha = 1 - conf_level
                lower_p = (alpha / 2) * 100
                upper_p = (1 - alpha / 2) * 100
                
                lower = np.percentile(bootstrap_predictions, lower_p, axis=0)
                upper = np.percentile(bootstrap_predictions, upper_p, axis=0)
                percentiles[conf_level] = (lower, upper)
            
            # Calculate uncertainty score
            uncertainty_score = self._calculate_uncertainty_score(bootstrap_predictions)
            
            return ConfidenceInterval(
                lower_50=float(np.mean(percentiles[0.5][0])),
                upper_50=float(np.mean(percentiles[0.5][1])),
                lower_80=float(np.mean(percentiles[0.8][0])),
                upper_80=float(np.mean(percentiles[0.8][1])),
                lower_95=float(np.mean(percentiles[0.95][0])),
                upper_95=float(np.mean(percentiles[0.95][1])),
                method=UncertaintyMethod.BOOTSTRAP,
                uncertainty_score=uncertainty_score
            )
            
        except Exception as e:
            logger.error(f"Bootstrap interval calculation error: {e}")
            return self._fallback_intervals(predictions, residuals)
    
    def calculate_ensemble_variance_intervals(self, 
                                            individual_predictions: Dict[ModelType, List[float]]) -> ConfidenceInterval:
        """Calculate confidence intervals based on ensemble variance"""
        try:
            if not individual_predictions:
                return self._empty_intervals()
            
            # Stack predictions from all models
            pred_matrix = []
            for model_type, preds in individual_predictions.items():
                pred_matrix.append(preds)
            
            pred_matrix = np.array(pred_matrix)
            
            # Calculate mean and variance across models
            ensemble_mean = np.mean(pred_matrix, axis=0)
            ensemble_var = np.var(pred_matrix, axis=0)
            ensemble_std = np.sqrt(ensemble_var)
            
            # Calculate intervals using normal distribution assumption
            intervals = {}
            for conf_level in self.confidence_levels:
                z_score = stats.norm.ppf(1 - (1 - conf_level) / 2)
                lower = ensemble_mean - z_score * ensemble_std
                upper = ensemble_mean + z_score * ensemble_std
                intervals[conf_level] = (lower, upper)
            
            # Uncertainty score based on coefficient of variation
            uncertainty_score = np.mean(ensemble_std / np.maximum(np.abs(ensemble_mean), 1))
            
            return ConfidenceInterval(
                lower_50=float(np.mean(intervals[0.5][0])),
                upper_50=float(np.mean(intervals[0.5][1])),
                lower_80=float(np.mean(intervals[0.8][0])),
                upper_80=float(np.mean(intervals[0.8][1])),
                lower_95=float(np.mean(intervals[0.95][0])),
                upper_95=float(np.mean(intervals[0.95][1])),
                method=UncertaintyMethod.ENSEMBLE_VARIANCE,
                uncertainty_score=min(uncertainty_score, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Ensemble variance interval calculation error: {e}")
            return self._empty_intervals()
    
    def _calculate_uncertainty_score(self, bootstrap_predictions: np.ndarray) -> float:
        """Calculate overall uncertainty score from bootstrap samples"""
        # Calculate coefficient of variation across bootstrap samples
        mean_pred = np.mean(bootstrap_predictions, axis=0)
        std_pred = np.std(bootstrap_predictions, axis=0)
        cv = std_pred / np.maximum(np.abs(mean_pred), 1)
        return float(np.mean(cv))
    
    def _fallback_intervals(self, predictions: List[float], residuals: List[float]) -> ConfidenceInterval:
        """Fallback method using simple standard deviation"""
        if not residuals:
            return self._empty_intervals()
        
        std_residual = np.std(residuals)
        mean_pred = np.mean(predictions)
        
        return ConfidenceInterval(
            lower_50=mean_pred - 0.67 * std_residual,
            upper_50=mean_pred + 0.67 * std_residual,
            lower_80=mean_pred - 1.28 * std_residual,
            upper_80=mean_pred + 1.28 * std_residual,
            lower_95=mean_pred - 1.96 * std_residual,
            upper_95=mean_pred + 1.96 * std_residual,
            method=UncertaintyMethod.BOOTSTRAP,
            uncertainty_score=min(std_residual / max(abs(mean_pred), 1), 1.0)
        )
    
    def _empty_intervals(self) -> ConfidenceInterval:
        """Return empty confidence intervals"""
        return ConfidenceInterval(
            lower_50=0, upper_50=0, lower_80=0, upper_80=0,
            lower_95=0, upper_95=0,
            method=UncertaintyMethod.BOOTSTRAP,
            uncertainty_score=1.0
        )


class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.performance_history = defaultdict(lambda: deque(maxlen=max_history))
        self.prediction_history = defaultdict(lambda: deque(maxlen=max_history))
        self.accuracy_thresholds = {
            'critical': 0.3,
            'high': 0.5,
            'medium': 0.7,
            'low': 0.85
        }
    
    def record_prediction(self, model_type: ModelType, symbol: str, 
                         predicted: float, actual: Optional[float] = None,
                         timestamp: datetime = None):
        """Record a prediction and optionally its actual value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        key = f"{symbol}_{model_type.value}"
        
        prediction_record = {
            'timestamp': timestamp,
            'predicted': predicted,
            'actual': actual,
            'error': abs(predicted - actual) if actual is not None else None
        }
        
        self.prediction_history[key].append(prediction_record)
    
    def update_actual_value(self, model_type: ModelType, symbol: str, 
                          timestamp: datetime, actual: float):
        """Update actual value for a previous prediction"""
        key = f"{symbol}_{model_type.value}"
        
        # Find the prediction closest to the timestamp
        predictions = list(self.prediction_history[key])
        if not predictions:
            return
        
        closest_pred = min(predictions, 
                          key=lambda x: abs((x['timestamp'] - timestamp).total_seconds()))
        
        if closest_pred['actual'] is None:
            closest_pred['actual'] = actual
            closest_pred['error'] = abs(closest_pred['predicted'] - actual)
            
            # Calculate performance metrics
            self._update_performance_metrics(key, model_type, symbol)
    
    def _update_performance_metrics(self, key: str, model_type: ModelType, symbol: str):
        """Update performance metrics for a model"""
        predictions = [p for p in self.prediction_history[key] 
                      if p['actual'] is not None and p['error'] is not None]
        
        if len(predictions) < 5:  # Need minimum predictions for reliable metrics
            return
        
        # Extract values
        predicted_values = [p['predicted'] for p in predictions]
        actual_values = [p['actual'] for p in predictions]
        errors = [p['error'] for p in predictions]
        
        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        # MAPE (handle division by zero)
        mape_values = []
        for pred, actual in zip(predicted_values, actual_values):
            if abs(actual) > 1e-8:
                mape_values.append(abs((actual - pred) / actual))
        mape = np.mean(mape_values) * 100 if mape_values else float('inf')
        
        # Directional accuracy
        directions_correct = 0
        for i in range(1, len(predictions)):
            pred_direction = predicted_values[i] > predicted_values[i-1]
            actual_direction = actual_values[i] > actual_values[i-1]
            if pred_direction == actual_direction:
                directions_correct += 1
        
        directional_accuracy = directions_correct / max(len(predictions) - 1, 1)
        
        # Overall accuracy score
        accuracy_score = max(0, 1 - (rmse / max(np.std(actual_values), 1)))
        
        # Store performance
        performance = ModelPerformance(
            model_type=model_type,
            symbol=symbol,
            mae=mae,
            rmse=rmse,
            mape=mape,
            accuracy_score=accuracy_score,
            last_updated=datetime.now(),
            prediction_count=len(predictions)
        )
        
        self.performance_history[key].append({
            'timestamp': datetime.now(),
            'performance': performance,
            'directional_accuracy': directional_accuracy
        })
    
    def get_performance_trend(self, model_type: ModelType, symbol: str, 
                            days: int = 30) -> Dict[str, float]:
        """Get performance trend over specified period"""
        key = f"{symbol}_{model_type.value}"
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_performance = [
            p for p in self.performance_history[key]
            if p['timestamp'] >= cutoff_date
        ]
        
        if len(recent_performance) < 2:
            return {'trend': 0, 'current_accuracy': 0, 'volatility': 0}
        
        # Calculate trend in accuracy
        accuracies = [p['performance'].accuracy_score for p in recent_performance]
        timestamps = [p['timestamp'].timestamp() for p in recent_performance]
        
        if len(accuracies) > 1:
            # Linear regression for trend
            slope, _, _, _, _ = stats.linregress(timestamps, accuracies)
            trend = slope * 86400 * days  # Convert to change per period
        else:
            trend = 0
        
        current_accuracy = accuracies[-1] if accuracies else 0
        volatility = np.std(accuracies) if len(accuracies) > 1 else 0
        
        return {
            'trend': trend,
            'current_accuracy': current_accuracy,
            'volatility': volatility
        }
    
    def assess_model_reliability(self, model_type: ModelType, symbol: str) -> ModelReliabilityScore:
        """Assess overall model reliability"""
        key = f"{symbol}_{model_type.value}"
        
        # Get recent performance data
        performance_trend = self.get_performance_trend(model_type, symbol, days=30)
        
        # Calculate consistency score
        recent_predictions = list(self.prediction_history[key])[-50:]  # Last 50 predictions
        if len(recent_predictions) > 10:
            errors = [p['error'] for p in recent_predictions if p['error'] is not None]
            if errors:
                consistency_score = max(0, 1 - (np.std(errors) / max(np.mean(errors), 1)))
            else:
                consistency_score = 0
        else:
            consistency_score = 0.5  # Neutral score for insufficient data
        
        # Calculate prediction stability
        if len(recent_predictions) > 5:
            pred_values = [p['predicted'] for p in recent_predictions]
            pred_changes = [abs(pred_values[i] - pred_values[i-1]) 
                           for i in range(1, len(pred_values))]
            avg_change = np.mean(pred_changes) if pred_changes else 0
            avg_value = np.mean(pred_values)
            stability_score = max(0, 1 - (avg_change / max(abs(avg_value), 1)))
        else:
            stability_score = 0.5
        
        # Overall reliability score
        overall_score = (
            performance_trend['current_accuracy'] * 0.4 +
            consistency_score * 0.3 +
            stability_score * 0.2 +
            max(0, performance_trend['trend']) * 0.1
        )
        
        return ModelReliabilityScore(
            model_type=model_type,
            symbol=symbol,
            overall_score=overall_score,
            consistency_score=consistency_score,
            accuracy_trend=performance_trend['trend'],
            prediction_stability=stability_score,
            data_quality_impact=0.8,  # Placeholder - would be calculated from data quality metrics
            last_assessment=datetime.now()
        )


class ModelPerformanceService:
    """Main service for model performance tracking and uncertainty quantification"""
    
    def __init__(self, data_integration_service: DataIntegrationService):
        self.data_service = data_integration_service
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.performance_tracker = PerformanceTracker()
        self.retraining_queue = deque()
        self.alert_history = deque(maxlen=1000)
    
    async def calculate_forecast_confidence(self, forecast: EnsembleForecast, 
                                          historical_residuals: Dict[ModelType, List[float]] = None) -> Dict[int, ConfidenceInterval]:
        """Calculate confidence intervals for ensemble forecast"""
        confidence_intervals = {}
        
        for horizon, prediction in forecast.horizons.items():
            # Method 1: Use ensemble variance if multiple models
            if len(forecast.individual_forecasts) > 1:
                individual_preds = {
                    model_type: [pred] for model_type, pred in 
                    [(mt, fc.predicted_values[list(forecast.horizons.keys()).index(horizon)]) 
                     for mt, fc in forecast.individual_forecasts.items()
                     if list(forecast.horizons.keys()).index(horizon) < len(fc.predicted_values)]
                }
                
                if individual_preds:
                    confidence_intervals[horizon] = self.uncertainty_quantifier.calculate_ensemble_variance_intervals(individual_preds)
                    continue
            
            # Method 2: Use bootstrap with historical residuals
            if historical_residuals:
                all_residuals = []
                for model_type, residuals in historical_residuals.items():
                    all_residuals.extend(residuals)
                
                if all_residuals:
                    confidence_intervals[horizon] = self.uncertainty_quantifier.calculate_bootstrap_intervals(
                        [prediction], all_residuals
                    )
                    continue
            
            # Method 3: Fallback using forecast confidence intervals
            if horizon in forecast.confidence_intervals:
                lower, upper = forecast.confidence_intervals[horizon]
                margin = (upper - lower) / 2
                
                confidence_intervals[horizon] = ConfidenceInterval(
                    lower_50=prediction - margin * 0.5,
                    upper_50=prediction + margin * 0.5,
                    lower_80=prediction - margin * 0.8,
                    upper_80=prediction + margin * 0.8,
                    lower_95=lower,
                    upper_95=upper,
                    method=UncertaintyMethod.ENSEMBLE_VARIANCE,
                    uncertainty_score=margin / max(abs(prediction), 1)
                )
        
        return confidence_intervals
    
    async def monitor_model_performance(self, model_type: ModelType, symbol: str) -> List[PerformanceAlert]:
        """Monitor model performance and generate alerts"""
        alerts = []
        
        # Assess model reliability
        reliability = self.performance_tracker.assess_model_reliability(model_type, symbol)
        
        # Check for accuracy degradation
        if reliability.overall_score < self.performance_tracker.accuracy_thresholds['critical']:
            alerts.append(PerformanceAlert(
                model_type=model_type,
                symbol=symbol,
                alert_type='accuracy_drop',
                severity='critical',
                message=f"Critical accuracy drop detected for {model_type.value} on {symbol}",
                metrics={'accuracy': reliability.overall_score},
                timestamp=datetime.now(),
                requires_retraining=True
            ))
        elif reliability.overall_score < self.performance_tracker.accuracy_thresholds['high']:
            alerts.append(PerformanceAlert(
                model_type=model_type,
                symbol=symbol,
                alert_type='accuracy_drop',
                severity='high',
                message=f"Significant accuracy drop detected for {model_type.value} on {symbol}",
                metrics={'accuracy': reliability.overall_score},
                timestamp=datetime.now(),
                requires_retraining=True
            ))
        
        # Check for negative accuracy trend
        if reliability.accuracy_trend < -0.1:  # Declining by more than 10% per month
            alerts.append(PerformanceAlert(
                model_type=model_type,
                symbol=symbol,
                alert_type='accuracy_trend',
                severity='medium',
                message=f"Declining accuracy trend detected for {model_type.value} on {symbol}",
                metrics={'trend': reliability.accuracy_trend},
                timestamp=datetime.now(),
                requires_retraining=False
            ))
        
        # Check for high prediction instability
        if reliability.prediction_stability < 0.5:
            alerts.append(PerformanceAlert(
                model_type=model_type,
                symbol=symbol,
                alert_type='instability',
                severity='medium',
                message=f"High prediction instability detected for {model_type.value} on {symbol}",
                metrics={'stability': reliability.prediction_stability},
                timestamp=datetime.now(),
                requires_retraining=False
            ))
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def generate_retraining_recommendations(self) -> List[RetrainingRecommendation]:
        """Generate recommendations for model retraining"""
        recommendations = []
        
        # Check all models that have generated alerts requiring retraining
        retraining_needed = set()
        for alert in list(self.alert_history)[-100:]:  # Check recent alerts
            if alert.requires_retraining:
                retraining_needed.add((alert.model_type, alert.symbol))
        
        for model_type, symbol in retraining_needed:
            reliability = self.performance_tracker.assess_model_reliability(model_type, symbol)
            
            # Determine priority based on accuracy score
            if reliability.overall_score < 0.3:
                priority = 'urgent'
            elif reliability.overall_score < 0.5:
                priority = 'high'
            elif reliability.overall_score < 0.7:
                priority = 'medium'
            else:
                priority = 'low'
            
            # Estimate potential improvement
            historical_best = 0.8  # Placeholder - would be calculated from historical data
            estimated_improvement = max(0, historical_best - reliability.overall_score)
            
            recommendations.append(RetrainingRecommendation(
                model_type=model_type,
                symbol=symbol,
                priority=priority,
                reason=f"Accuracy dropped to {reliability.overall_score:.2f}",
                performance_degradation=1 - reliability.overall_score,
                data_freshness_score=0.8,  # Placeholder
                recommended_action="Retrain with recent data and hyperparameter optimization",
                estimated_improvement=estimated_improvement
            ))
        
        # Sort by priority and estimated improvement
        priority_order = {'urgent': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: (priority_order[x.priority], -x.estimated_improvement))
        
        return recommendations
    
    async def track_forecast_accuracy(self, forecast: EnsembleForecast, 
                                    actual_values: Dict[int, float]):
        """Track forecast accuracy against actual values"""
        for horizon, actual_value in actual_values.items():
            if horizon in forecast.horizons:
                predicted_value = forecast.horizons[horizon]
                
                # Record prediction for each individual model
                for model_type, individual_forecast in forecast.individual_forecasts.items():
                    horizon_index = list(forecast.horizons.keys()).index(horizon)
                    if horizon_index < len(individual_forecast.predicted_values):
                        individual_pred = individual_forecast.predicted_values[horizon_index]
                        self.performance_tracker.record_prediction(
                            model_type, forecast.symbol, individual_pred, actual_value,
                            forecast.forecast_date + timedelta(days=horizon)
                        )
                
                # Monitor performance and generate alerts if needed
                for model_type in forecast.individual_forecasts.keys():
                    alerts = await self.monitor_model_performance(model_type, forecast.symbol)
                    if alerts:
                        logger.info(f"Generated {len(alerts)} performance alerts for {model_type.value} on {forecast.symbol}")
    
    def get_model_performance_summary(self, symbol: str = None, 
                                    model_type: ModelType = None) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        summary = {
            'timestamp': datetime.now(),
            'models': {},
            'alerts': len([a for a in self.alert_history if a.timestamp > datetime.now() - timedelta(days=7)]),
            'retraining_recommendations': len(self.generate_retraining_recommendations())
        }
        
        # Get performance for specific models or all models
        models_to_check = []
        if model_type and symbol:
            models_to_check = [(model_type, symbol)]
        else:
            # Get all unique model-symbol combinations from performance history
            for key in self.performance_tracker.performance_history.keys():
                parts = key.split('_', 1)
                if len(parts) == 2:
                    symbol_part, model_part = parts
                    try:
                        mt = ModelType(model_part)
                        models_to_check.append((mt, symbol_part))
                    except ValueError:
                        continue
        
        for mt, sym in models_to_check:
            if symbol and sym != symbol:
                continue
            if model_type and mt != model_type:
                continue
            
            reliability = self.performance_tracker.assess_model_reliability(mt, sym)
            performance_trend = self.performance_tracker.get_performance_trend(mt, sym)
            
            model_key = f"{sym}_{mt.value}"
            summary['models'][model_key] = {
                'reliability_score': reliability.overall_score,
                'accuracy_trend': performance_trend['trend'],
                'current_accuracy': performance_trend['current_accuracy'],
                'consistency': reliability.consistency_score,
                'stability': reliability.prediction_stability,
                'last_assessment': reliability.last_assessment.isoformat()
            }
        
        return summary


# Global instance
model_performance_service = None

def get_model_performance_service(data_integration_service: DataIntegrationService = None) -> ModelPerformanceService:
    """Get or create model performance service instance"""
    global model_performance_service
    if model_performance_service is None:
        if data_integration_service is None:
            from .data_integration_service import data_integration_service as default_service
            data_integration_service = default_service
        model_performance_service = ModelPerformanceService(data_integration_service)
    return model_performance_service