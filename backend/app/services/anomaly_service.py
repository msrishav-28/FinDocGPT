"""
Statistical Anomaly Detection Service

This service implements comprehensive anomaly detection for financial metrics using:
- Statistical outlier detection (Z-score, IQR, Isolation Forest)
- Dynamic baseline establishment with rolling windows and seasonal adjustments
- Multi-metric correlation analysis for pattern detection
- Machine learning models for complex pattern anomalies
- Risk assessment and anomaly management

Features:
- Multi-algorithm ensemble for robust detection
- Dynamic threshold adjustment based on market conditions
- Contextual explanation generation
- Historical anomaly tracking and resolution management
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

# Statistical and ML libraries
from scipy import stats
from scipy.stats import zscore, iqr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

from ..models.anomaly import (
    Anomaly, AnomalySeverity, AnomalyType, AnomalyStatus,
    PatternAnomalies, RiskAssessment, AnomalyHistory,
    AnomalyCorrelation, AnomalyBaseline, AnomalyAlert
)
from ..config import get_settings
from ..database.connection import get_database

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class DetectionResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    score: float
    method: str
    threshold: float
    explanation: str
    confidence: float


class BaselineType(str, Enum):
    """Types of baseline models"""
    ROLLING_MEAN = "rolling_mean"
    SEASONAL_DECOMPOSE = "seasonal_decompose"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    STATISTICAL_MODEL = "statistical_model"


class AnomalyDetectionEngine:
    """Multi-algorithm anomaly detection engine"""
    
    def __init__(self):
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.detection_history: Dict[str, List[Dict]] = {}
        self.correlation_cache: Dict[str, Dict] = {}
        
        # Detection parameters
        self.z_score_threshold = 2.5
        self.iqr_multiplier = 1.5
        self.isolation_forest_contamination = 0.1
        self.min_baseline_periods = 12  # Minimum quarters for baseline
        
        # Seasonal parameters
        self.seasonal_periods = {
            'quarterly': 4,
            'monthly': 12,
            'weekly': 52
        }
    
    async def detect_metric_anomalies(
        self, 
        company: str, 
        metrics: List[str],
        lookback_periods: int = 20,
        detection_methods: List[str] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies in financial metrics using multiple algorithms
        
        Args:
            company: Company identifier
            metrics: List of metric names to analyze
            lookback_periods: Number of historical periods to consider
            detection_methods: List of methods to use ['zscore', 'iqr', 'isolation_forest']
        
        Returns:
            List of detected anomalies
        """
        if detection_methods is None:
            detection_methods = ['zscore', 'iqr', 'isolation_forest']
        
        detected_anomalies = []
        
        for metric in metrics:
            try:
                # Get historical data for the metric
                historical_data = await self._get_metric_data(company, metric, lookback_periods)
                
                if len(historical_data) < self.min_baseline_periods:
                    logger.warning(f"Insufficient data for {company} {metric}: {len(historical_data)} periods")
                    continue
                
                # Establish or update baseline
                baseline = await self._establish_baseline(company, metric, historical_data)
                
                # Get current value
                current_value = historical_data[-1] if historical_data else None
                if current_value is None:
                    continue
                
                # Run detection algorithms
                detection_results = []
                
                if 'zscore' in detection_methods:
                    result = await self._detect_zscore_anomaly(
                        current_value, historical_data, baseline
                    )
                    detection_results.append(result)
                
                if 'iqr' in detection_methods:
                    result = await self._detect_iqr_anomaly(
                        current_value, historical_data, baseline
                    )
                    detection_results.append(result)
                
                if 'isolation_forest' in detection_methods:
                    result = await self._detect_isolation_forest_anomaly(
                        current_value, historical_data, baseline
                    )
                    detection_results.append(result)
                
                # Combine results using ensemble approach
                anomaly = await self._combine_detection_results(
                    company, metric, current_value, detection_results, baseline
                )
                
                if anomaly:
                    detected_anomalies.append(anomaly)
                    
            except Exception as e:
                logger.error(f"Failed to detect anomalies for {company} {metric}: {e}")
                continue
        
        return detected_anomalies
    
    async def _get_metric_data(
        self, 
        company: str, 
        metric: str, 
        periods: int
    ) -> List[float]:
        """Get historical metric data from database"""
        db = await get_database()
        
        try:
            # Query financial metrics from database
            # This assumes we have a financial_metrics table
            query = """
                SELECT value, period_date
                FROM financial_metrics 
                WHERE company = $1 AND metric_name = $2
                ORDER BY period_date DESC
                LIMIT $3
            """
            
            rows = await db.fetch(query, company, metric, periods)
            
            if not rows:
                # Fallback: generate sample data for testing
                logger.warning(f"No data found for {company} {metric}, generating sample data")
                return self._generate_sample_data(periods)
            
            # Extract values and reverse to get chronological order
            values = [float(row['value']) for row in reversed(rows)]
            return values
            
        except Exception as e:
            logger.error(f"Failed to get metric data: {e}")
            # Return sample data for testing
            return self._generate_sample_data(periods)
    
    def _generate_sample_data(self, periods: int) -> List[float]:
        """Generate sample financial data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate base trend with some seasonality and noise
        base_value = 100.0
        trend = np.linspace(0, 20, periods)  # Upward trend
        seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, periods))  # Seasonal pattern
        noise = np.random.normal(0, 5, periods)  # Random noise
        
        # Add some anomalies in the last few periods
        data = base_value + trend + seasonal + noise
        
        # Inject anomalies in last 3 periods
        if periods > 3:
            data[-3] *= 1.5  # 50% increase
            data[-1] *= 0.7  # 30% decrease
        
        return data.tolist()
    
    async def _establish_baseline(
        self, 
        company: str, 
        metric: str, 
        historical_data: List[float]
    ) -> AnomalyBaseline:
        """Establish dynamic baseline for anomaly detection"""
        
        baseline_key = f"{company}_{metric}"
        
        # Convert to pandas series for easier analysis
        data_series = pd.Series(historical_data)
        
        # Calculate rolling statistics
        rolling_window = min(8, len(historical_data) // 2)  # Adaptive window size
        rolling_mean = data_series.rolling(window=rolling_window).mean().iloc[-1]
        rolling_std = data_series.rolling(window=rolling_window).std().iloc[-1]
        
        # Seasonal decomposition if enough data
        seasonal_adjustments = None
        if len(historical_data) >= 12:
            try:
                decomposition = seasonal_decompose(
                    data_series, 
                    model='additive', 
                    period=4,  # Quarterly seasonality
                    extrapolate_trend='freq'
                )
                seasonal_adjustments = {
                    'seasonal_component': decomposition.seasonal.iloc[-1],
                    'trend_component': decomposition.trend.iloc[-1],
                    'residual_std': decomposition.resid.std()
                }
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
        
        # Calculate baseline parameters
        baseline_parameters = {
            'mean': float(np.mean(historical_data)),
            'std': float(np.std(historical_data)),
            'rolling_mean': float(rolling_mean) if not np.isnan(rolling_mean) else float(np.mean(historical_data)),
            'rolling_std': float(rolling_std) if not np.isnan(rolling_std) else float(np.std(historical_data)),
            'median': float(np.median(historical_data)),
            'q25': float(np.percentile(historical_data, 25)),
            'q75': float(np.percentile(historical_data, 75)),
            'iqr': float(iqr(historical_data)),
            'min': float(np.min(historical_data)),
            'max': float(np.max(historical_data))
        }
        
        # Performance metrics (placeholder - would be calculated from historical accuracy)
        performance_metrics = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77
        }
        
        baseline = AnomalyBaseline(
            company=company,
            metric_name=metric,
            baseline_type=BaselineType.ROLLING_MEAN.value,
            baseline_parameters=baseline_parameters,
            baseline_period=f"{len(historical_data)}_periods",
            last_updated=datetime.now(),
            performance_metrics=performance_metrics,
            seasonal_adjustments=seasonal_adjustments
        )
        
        # Cache the baseline
        self.baselines[baseline_key] = baseline
        
        return baseline
    
    async def _detect_zscore_anomaly(
        self, 
        current_value: float, 
        historical_data: List[float], 
        baseline: AnomalyBaseline
    ) -> DetectionResult:
        """Detect anomalies using Z-score method"""
        
        # Calculate Z-score using rolling statistics for better adaptation
        rolling_mean = baseline.baseline_parameters['rolling_mean']
        rolling_std = baseline.baseline_parameters['rolling_std']
        
        if rolling_std == 0:
            rolling_std = baseline.baseline_parameters['std']
        
        if rolling_std == 0:
            return DetectionResult(
                is_anomaly=False,
                score=0.0,
                method="zscore",
                threshold=self.z_score_threshold,
                explanation="Cannot calculate Z-score: zero standard deviation",
                confidence=0.0
            )
        
        z_score = abs(current_value - rolling_mean) / rolling_std
        is_anomaly = z_score > self.z_score_threshold
        
        # Calculate confidence based on how far beyond threshold
        confidence = min(1.0, z_score / (self.z_score_threshold * 2))
        
        explanation = (
            f"Z-score: {z_score:.2f} (threshold: {self.z_score_threshold}). "
            f"Current value {current_value:.2f} vs rolling mean {rolling_mean:.2f} "
            f"(±{rolling_std:.2f})"
        )
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            score=z_score,
            method="zscore",
            threshold=self.z_score_threshold,
            explanation=explanation,
            confidence=confidence
        )
    
    async def _detect_iqr_anomaly(
        self, 
        current_value: float, 
        historical_data: List[float], 
        baseline: AnomalyBaseline
    ) -> DetectionResult:
        """Detect anomalies using Interquartile Range (IQR) method"""
        
        q25 = baseline.baseline_parameters['q25']
        q75 = baseline.baseline_parameters['q75']
        iqr_value = baseline.baseline_parameters['iqr']
        
        # Calculate IQR bounds
        lower_bound = q25 - self.iqr_multiplier * iqr_value
        upper_bound = q75 + self.iqr_multiplier * iqr_value
        
        is_anomaly = current_value < lower_bound or current_value > upper_bound
        
        # Calculate score as distance from nearest bound
        if current_value < lower_bound:
            score = (lower_bound - current_value) / iqr_value
            direction = "below"
        elif current_value > upper_bound:
            score = (current_value - upper_bound) / iqr_value
            direction = "above"
        else:
            score = 0.0
            direction = "within"
        
        # Calculate confidence
        confidence = min(1.0, score / 2.0) if is_anomaly else 0.0
        
        explanation = (
            f"IQR analysis: value {current_value:.2f} is {direction} normal range "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]. "
            f"IQR bounds: Q25={q25:.2f}, Q75={q75:.2f}, IQR={iqr_value:.2f}"
        )
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            score=score,
            method="iqr",
            threshold=self.iqr_multiplier,
            explanation=explanation,
            confidence=confidence
        )
    
    async def _detect_isolation_forest_anomaly(
        self, 
        current_value: float, 
        historical_data: List[float], 
        baseline: AnomalyBaseline
    ) -> DetectionResult:
        """Detect anomalies using Isolation Forest"""
        
        try:
            # Prepare data for Isolation Forest
            data_array = np.array(historical_data + [current_value]).reshape(-1, 1)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.isolation_forest_contamination,
                random_state=42,
                n_estimators=100
            )
            
            predictions = iso_forest.fit_predict(data_array)
            anomaly_scores = iso_forest.decision_function(data_array)
            
            # Check if current value is anomaly
            current_prediction = predictions[-1]
            current_score = anomaly_scores[-1]
            
            is_anomaly = current_prediction == -1
            
            # Normalize score to positive value (lower scores indicate more anomalous)
            normalized_score = abs(current_score) if is_anomaly else 0.0
            
            # Calculate confidence based on how negative the score is
            confidence = min(1.0, abs(current_score) * 2) if is_anomaly else 0.0
            
            explanation = (
                f"Isolation Forest: anomaly score {current_score:.3f}. "
                f"{'Anomalous' if is_anomaly else 'Normal'} pattern detected. "
                f"Contamination threshold: {self.isolation_forest_contamination}"
            )
            
            return DetectionResult(
                is_anomaly=is_anomaly,
                score=normalized_score,
                method="isolation_forest",
                threshold=self.isolation_forest_contamination,
                explanation=explanation,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            return DetectionResult(
                is_anomaly=False,
                score=0.0,
                method="isolation_forest",
                threshold=self.isolation_forest_contamination,
                explanation=f"Isolation Forest failed: {str(e)}",
                confidence=0.0
            )
    
    async def _combine_detection_results(
        self,
        company: str,
        metric: str,
        current_value: float,
        detection_results: List[DetectionResult],
        baseline: AnomalyBaseline
    ) -> Optional[Anomaly]:
        """Combine results from multiple detection methods"""
        
        if not detection_results:
            return None
        
        # Count anomaly detections
        anomaly_count = sum(1 for result in detection_results if result.is_anomaly)
        total_methods = len(detection_results)
        
        # Require majority consensus for anomaly detection
        consensus_threshold = 0.5
        is_ensemble_anomaly = (anomaly_count / total_methods) >= consensus_threshold
        
        if not is_ensemble_anomaly:
            return None
        
        # Calculate ensemble scores and confidence
        anomaly_results = [r for r in detection_results if r.is_anomaly]
        ensemble_score = np.mean([r.score for r in anomaly_results])
        ensemble_confidence = np.mean([r.confidence for r in anomaly_results])
        
        # Determine severity based on ensemble score and confidence
        severity = self._determine_severity(ensemble_score, ensemble_confidence, anomaly_count, total_methods)
        
        # Determine anomaly type based on detection methods
        anomaly_type = self._determine_anomaly_type(detection_results)
        
        # Calculate expected value from baseline
        expected_value = baseline.baseline_parameters['rolling_mean']
        deviation_score = abs(current_value - expected_value) / baseline.baseline_parameters['rolling_std']
        
        # Generate explanation
        method_explanations = [r.explanation for r in anomaly_results]
        explanation = self._generate_ensemble_explanation(
            current_value, expected_value, method_explanations, anomaly_count, total_methods
        )
        
        # Generate historical context
        historical_context = self._generate_historical_context(
            current_value, baseline, metric
        )
        
        # Create anomaly object
        anomaly = Anomaly(
            company=company,
            metric_name=metric,
            current_value=current_value,
            expected_value=expected_value,
            deviation_score=deviation_score,
            severity=severity,
            anomaly_type=anomaly_type,
            status=AnomalyStatus.DETECTED,
            explanation=explanation,
            historical_context=historical_context,
            potential_causes=self._suggest_potential_causes(metric, current_value, expected_value),
            detection_method=f"ensemble({','.join([r.method for r in anomaly_results])})",
            confidence=ensemble_confidence,
            baseline_period=baseline.baseline_period
        )
        
        return anomaly
    
    def _determine_severity(
        self, 
        ensemble_score: float, 
        ensemble_confidence: float, 
        anomaly_count: int, 
        total_methods: int
    ) -> AnomalySeverity:
        """Determine anomaly severity based on ensemble results"""
        
        # Calculate severity score combining multiple factors
        consensus_factor = anomaly_count / total_methods
        severity_score = ensemble_score * ensemble_confidence * consensus_factor
        
        if severity_score >= 3.0 and consensus_factor >= 0.8:
            return AnomalySeverity.CRITICAL
        elif severity_score >= 2.0 and consensus_factor >= 0.6:
            return AnomalySeverity.HIGH
        elif severity_score >= 1.0 and consensus_factor >= 0.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _determine_anomaly_type(self, detection_results: List[DetectionResult]) -> AnomalyType:
        """Determine anomaly type based on detection methods"""
        
        anomaly_methods = [r.method for r in detection_results if r.is_anomaly]
        
        if 'isolation_forest' in anomaly_methods:
            return AnomalyType.PATTERN_DEVIATION
        elif len(set(anomaly_methods)) > 1:
            return AnomalyType.STATISTICAL_OUTLIER
        elif 'zscore' in anomaly_methods:
            return AnomalyType.STATISTICAL_OUTLIER
        elif 'iqr' in anomaly_methods:
            return AnomalyType.STATISTICAL_OUTLIER
        else:
            return AnomalyType.STATISTICAL_OUTLIER
    
    def _generate_ensemble_explanation(
        self,
        current_value: float,
        expected_value: float,
        method_explanations: List[str],
        anomaly_count: int,
        total_methods: int
    ) -> str:
        """Generate comprehensive explanation for ensemble detection"""
        
        deviation_pct = ((current_value - expected_value) / expected_value) * 100
        direction = "above" if current_value > expected_value else "below"
        
        explanation = (
            f"Anomaly detected: current value {current_value:.2f} is {abs(deviation_pct):.1f}% "
            f"{direction} expected value {expected_value:.2f}. "
            f"Consensus: {anomaly_count}/{total_methods} detection methods agree. "
        )
        
        # Add method-specific details
        if len(method_explanations) <= 2:
            explanation += " Details: " + " | ".join(method_explanations)
        else:
            explanation += f" Multiple detection methods confirm anomalous behavior."
        
        return explanation
    
    def _generate_historical_context(
        self, 
        current_value: float, 
        baseline: AnomalyBaseline, 
        metric: str
    ) -> str:
        """Generate historical context for the anomaly"""
        
        params = baseline.baseline_parameters
        
        # Compare to historical statistics
        percentile_rank = self._calculate_percentile_rank(current_value, params)
        
        context_parts = []
        
        # Historical comparison
        if current_value > params['max']:
            context_parts.append(f"This is the highest {metric} value on record")
        elif current_value < params['min']:
            context_parts.append(f"This is the lowest {metric} value on record")
        elif percentile_rank > 95:
            context_parts.append(f"This value is in the top 5% historically")
        elif percentile_rank < 5:
            context_parts.append(f"This value is in the bottom 5% historically")
        
        # Volatility context
        if params['std'] > 0:
            volatility_ratio = abs(current_value - params['mean']) / params['std']
            if volatility_ratio > 3:
                context_parts.append("This represents extremely high volatility")
            elif volatility_ratio > 2:
                context_parts.append("This represents high volatility")
        
        # Trend context
        if abs(current_value - params['rolling_mean']) > abs(current_value - params['mean']):
            context_parts.append("This deviates more from recent trend than historical average")
        
        return ". ".join(context_parts) if context_parts else "No significant historical context available"
    
    def _calculate_percentile_rank(self, value: float, params: Dict[str, float]) -> float:
        """Calculate approximate percentile rank of value"""
        # Simple approximation using normal distribution
        mean = params['mean']
        std = params['std']
        
        if std == 0:
            return 50.0
        
        z_score = (value - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        
        return max(0, min(100, percentile))
    
    def _suggest_potential_causes(
        self, 
        metric: str, 
        current_value: float, 
        expected_value: float
    ) -> List[str]:
        """Suggest potential causes for the anomaly based on metric type"""
        
        causes = []
        is_increase = current_value > expected_value
        
        # Metric-specific potential causes
        metric_lower = metric.lower()
        
        if 'revenue' in metric_lower:
            if is_increase:
                causes.extend([
                    "Strong product demand or new product launch",
                    "Market expansion or new customer acquisition",
                    "Seasonal factors or one-time events",
                    "Price increases or improved pricing strategy"
                ])
            else:
                causes.extend([
                    "Market downturn or competitive pressure",
                    "Product issues or customer churn",
                    "Seasonal decline or economic factors",
                    "Supply chain disruptions"
                ])
        
        elif 'profit' in metric_lower or 'margin' in metric_lower:
            if is_increase:
                causes.extend([
                    "Cost reduction initiatives",
                    "Operational efficiency improvements",
                    "Favorable market conditions",
                    "One-time gains or accounting adjustments"
                ])
            else:
                causes.extend([
                    "Increased costs or inflation",
                    "Investment in growth initiatives",
                    "Market competition or pricing pressure",
                    "One-time charges or write-offs"
                ])
        
        elif 'cash' in metric_lower:
            if is_increase:
                causes.extend([
                    "Strong operational cash generation",
                    "Asset sales or financing activities",
                    "Improved working capital management",
                    "Delayed capital expenditures"
                ])
            else:
                causes.extend([
                    "Large capital investments",
                    "Debt repayments or dividends",
                    "Working capital increases",
                    "Operational cash flow challenges"
                ])
        
        else:
            # Generic causes
            if is_increase:
                causes.extend([
                    "Positive business developments",
                    "Market or industry tailwinds",
                    "Management initiatives showing results",
                    "External factors or one-time events"
                ])
            else:
                causes.extend([
                    "Business challenges or headwinds",
                    "Market or industry pressures",
                    "Operational difficulties",
                    "External factors or one-time charges"
                ])
        
        return causes[:4]  # Return top 4 most relevant causes


# Global anomaly detection engine instance
_anomaly_engine = None


async def get_anomaly_engine() -> AnomalyDetectionEngine:
    """Get or create anomaly detection engine instance"""
    global _anomaly_engine
    if _anomaly_engine is None:
        _anomaly_engine = AnomalyDetectionEngine()
    return _anomaly_engine


# Main service functions
async def detect_metric_anomalies(
    company: str, 
    metrics: List[str],
    lookback_periods: int = 20
) -> List[Anomaly]:
    """Detect anomalies in financial metrics"""
    engine = await get_anomaly_engine()
    return await engine.detect_metric_anomalies(company, metrics, lookback_periods)


async def analyze_correlation_anomalies(
    company: str,
    metrics: List[str],
    correlation_threshold: float = 0.7
) -> List[AnomalyCorrelation]:
    """Analyze correlations between metrics to detect systemic anomalies"""
    engine = await get_anomaly_engine()
    
    correlations = []
    
    # Get data for all metrics
    metric_data = {}
    for metric in metrics:
        data = await engine._get_metric_data(company, metric, 20)
        if len(data) >= 10:  # Minimum data requirement
            metric_data[metric] = data
    
    if len(metric_data) < 2:
        return correlations
    
    # Calculate correlations between all metric pairs
    metric_names = list(metric_data.keys())
    
    for i, metric1 in enumerate(metric_names):
        for j, metric2 in enumerate(metric_names[i+1:], i+1):
            try:
                data1 = metric_data[metric1]
                data2 = metric_data[metric2]
                
                # Ensure same length
                min_len = min(len(data1), len(data2))
                data1 = data1[-min_len:]
                data2 = data2[-min_len:]
                
                # Calculate correlation
                correlation_coef = np.corrcoef(data1, data2)[0, 1]
                
                if not np.isnan(correlation_coef) and abs(correlation_coef) >= correlation_threshold:
                    # This is a placeholder - in real implementation, you'd have anomaly IDs
                    correlation = AnomalyCorrelation(
                        primary_anomaly_id="placeholder_id_1",
                        correlated_anomaly_ids=["placeholder_id_2"],
                        correlation_strength=float(correlation_coef),
                        correlation_type="metric_correlation",
                        statistical_significance=0.95  # Placeholder
                    )
                    correlations.append(correlation)
                    
            except Exception as e:
                logger.error(f"Failed to calculate correlation between {metric1} and {metric2}: {e}")
                continue
    
    return correlations


async def get_anomaly_history(
    company: str,
    days_back: int = 90
) -> AnomalyHistory:
    """Get historical anomaly data for a company"""
    db = await get_database()
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Query anomalies from database
        query = """
            SELECT 
                severity,
                anomaly_type,
                status,
                created_at,
                resolved_at
            FROM anomalies
            WHERE company = $1 
            AND created_at >= $2 
            AND created_at <= $3
            ORDER BY created_at DESC
        """
        
        rows = await db.fetch(query, company, start_date, end_date)
        
        # Aggregate statistics
        total_anomalies = len(rows)
        anomalies_by_severity = {}
        anomalies_by_type = {}
        resolved_count = 0
        resolution_times = []
        
        for row in rows:
            # Count by severity
            severity = row['severity']
            anomalies_by_severity[severity] = anomalies_by_severity.get(severity, 0) + 1
            
            # Count by type
            anomaly_type = row['anomaly_type']
            anomalies_by_type[anomaly_type] = anomalies_by_type.get(anomaly_type, 0) + 1
            
            # Calculate resolution metrics
            if row['status'] in ['resolved', 'explained']:
                resolved_count += 1
                if row['resolved_at']:
                    resolution_time = (row['resolved_at'] - row['created_at']).total_seconds() / 3600
                    resolution_times.append(resolution_time)
        
        # Calculate resolution rate and average time
        resolution_rate = resolved_count / total_anomalies if total_anomalies > 0 else 0.0
        avg_resolution_time = np.mean(resolution_times) if resolution_times else None
        
        return AnomalyHistory(
            company=company,
            time_period=f"{days_back}_days",
            total_anomalies=total_anomalies,
            anomalies_by_severity=anomalies_by_severity,
            anomalies_by_type=anomalies_by_type,
            resolution_rate=resolution_rate,
            average_resolution_time=avg_resolution_time,
            recurring_patterns=[]  # Placeholder - would analyze patterns
        )
        
    except Exception as e:
        logger.error(f"Failed to get anomaly history for {company}: {e}")
        return AnomalyHistory(
            company=company,
            time_period=f"{days_back}_days",
            total_anomalies=0,
            anomalies_by_severity={},
            anomalies_by_type={},
            resolution_rate=0.0,
            average_resolution_time=None,
            recurring_patterns=[]
        )


async def store_anomaly(anomaly: Anomaly) -> str:
    """Store anomaly in database"""
    db = await get_database()
    
    try:
        query = """
            INSERT INTO anomalies 
            (id, company, metric_name, current_value, expected_value, deviation_score,
             severity, anomaly_type, status, explanation, historical_context, 
             potential_causes, detection_method, confidence, baseline_period, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING id
        """
        
        result = await db.fetchrow(
            query,
            anomaly.id,
            anomaly.company,
            anomaly.metric_name,
            anomaly.current_value,
            anomaly.expected_value,
            anomaly.deviation_score,
            anomaly.severity.value,
            anomaly.anomaly_type.value,
            anomaly.status.value,
            anomaly.explanation,
            anomaly.historical_context,
            json.dumps(anomaly.potential_causes),
            anomaly.detection_method,
            anomaly.confidence,
            anomaly.baseline_period,
            anomaly.created_at
        )
        
        return str(result['id'])
        
    except Exception as e:
        logger.error(f"Failed to store anomaly: {e}")
        raise


class PatternAnomalyDetector:
    """Machine learning-based pattern anomaly detection"""
    
    def __init__(self):
        self.pattern_models = {}
        self.severity_classifier = None
        self.explanation_templates = {
            'trend_break': "Significant trend change detected: {description}",
            'seasonal_anomaly': "Seasonal pattern deviation: {description}",
            'volatility_spike': "Unusual volatility pattern: {description}",
            'correlation_break': "Correlation pattern disruption: {description}",
            'clustering_anomaly': "Outlier in behavioral clustering: {description}"
        }
    
    async def analyze_pattern_anomalies(
        self,
        company: str,
        metrics_data: Dict[str, List[float]],
        time_periods: List[datetime] = None
    ) -> PatternAnomalies:
        """
        Analyze complex pattern anomalies using machine learning models
        
        Args:
            company: Company identifier
            metrics_data: Dictionary of metric names to historical values
            time_periods: Optional list of time periods corresponding to data points
        
        Returns:
            PatternAnomalies object with detected patterns
        """
        
        if not metrics_data or not any(len(data) >= 10 for data in metrics_data.values()):
            return PatternAnomalies(
                company=company,
                analysis_period="insufficient_data",
                detected_patterns=[],
                pattern_confidence=0.0,
                anomalous_patterns=[],
                pattern_explanation="Insufficient data for pattern analysis"
            )
        
        detected_patterns = []
        anomalous_patterns = []
        
        try:
            # 1. Trend break detection
            trend_anomalies = await self._detect_trend_breaks(company, metrics_data)
            anomalous_patterns.extend(trend_anomalies)
            
            # 2. Seasonal anomaly detection
            seasonal_anomalies = await self._detect_seasonal_anomalies(company, metrics_data)
            anomalous_patterns.extend(seasonal_anomalies)
            
            # 3. Volatility pattern detection
            volatility_anomalies = await self._detect_volatility_patterns(company, metrics_data)
            anomalous_patterns.extend(volatility_anomalies)
            
            # 4. Multi-metric correlation pattern detection
            correlation_anomalies = await self._detect_correlation_patterns(company, metrics_data)
            anomalous_patterns.extend(correlation_anomalies)
            
            # 5. Clustering-based anomaly detection
            clustering_anomalies = await self._detect_clustering_anomalies(company, metrics_data)
            anomalous_patterns.extend(clustering_anomalies)
            
            # Calculate overall pattern confidence
            if anomalous_patterns:
                pattern_confidence = np.mean([a.confidence for a in anomalous_patterns])
            else:
                pattern_confidence = 0.0
            
            # Generate pattern summary
            pattern_types = [a.anomaly_type.value for a in anomalous_patterns]
            pattern_summary = self._generate_pattern_summary(pattern_types, len(anomalous_patterns))
            
            return PatternAnomalies(
                company=company,
                analysis_period=f"{max(len(data) for data in metrics_data.values())}_periods",
                detected_patterns=detected_patterns,
                pattern_confidence=pattern_confidence,
                anomalous_patterns=anomalous_patterns,
                pattern_explanation=pattern_summary
            )
            
        except Exception as e:
            logger.error(f"Pattern anomaly analysis failed for {company}: {e}")
            return PatternAnomalies(
                company=company,
                analysis_period="error",
                detected_patterns=[],
                pattern_confidence=0.0,
                anomalous_patterns=[],
                pattern_explanation=f"Pattern analysis failed: {str(e)}"
            )
    
    async def _detect_trend_breaks(
        self,
        company: str,
        metrics_data: Dict[str, List[float]]
    ) -> List[Anomaly]:
        """Detect significant trend breaks using change point detection"""
        
        trend_anomalies = []
        
        for metric_name, data in metrics_data.items():
            if len(data) < 15:  # Need sufficient data for trend analysis
                continue
            
            try:
                # Convert to pandas series
                series = pd.Series(data)
                
                # Calculate rolling trends
                short_window = 5
                long_window = 10
                
                short_trend = series.rolling(window=short_window).mean().diff()
                long_trend = series.rolling(window=long_window).mean().diff()
                
                # Detect trend breaks
                recent_short_trend = short_trend.iloc[-3:].mean()
                recent_long_trend = long_trend.iloc[-5:].mean()
                
                # Check for significant trend change
                trend_change = abs(recent_short_trend - recent_long_trend)
                trend_threshold = np.std(series.diff()) * 2
                
                if trend_change > trend_threshold:
                    # Determine trend direction
                    if recent_short_trend > recent_long_trend:
                        trend_direction = "accelerating upward"
                    else:
                        trend_direction = "accelerating downward"
                    
                    # Calculate severity
                    severity = self._classify_pattern_severity(
                        trend_change / trend_threshold,
                        confidence=0.8
                    )
                    
                    # Create anomaly
                    anomaly = Anomaly(
                        company=company,
                        metric_name=metric_name,
                        current_value=data[-1],
                        expected_value=series.rolling(window=long_window).mean().iloc[-1],
                        deviation_score=trend_change / trend_threshold,
                        severity=severity,
                        anomaly_type=AnomalyType.TREND_BREAK,
                        explanation=self.explanation_templates['trend_break'].format(
                            description=f"{metric_name} trend is {trend_direction} with change magnitude {trend_change:.3f}"
                        ),
                        detection_method="trend_break_analysis",
                        confidence=min(0.9, trend_change / trend_threshold / 3),
                        potential_causes=self._get_trend_break_causes(metric_name, trend_direction)
                    )
                    
                    trend_anomalies.append(anomaly)
                    
            except Exception as e:
                logger.warning(f"Trend break detection failed for {metric_name}: {e}")
                continue
        
        return trend_anomalies
    
    async def _detect_seasonal_anomalies(
        self,
        company: str,
        metrics_data: Dict[str, List[float]]
    ) -> List[Anomaly]:
        """Detect seasonal pattern deviations"""
        
        seasonal_anomalies = []
        
        for metric_name, data in metrics_data.items():
            if len(data) < 12:  # Need at least 3 seasons
                continue
            
            try:
                series = pd.Series(data)
                
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    series,
                    model='additive',
                    period=4,  # Quarterly seasonality
                    extrapolate_trend='freq'
                )
                
                # Analyze residuals for anomalies
                residuals = decomposition.resid.dropna()
                residual_std = residuals.std()
                recent_residual = residuals.iloc[-1]
                
                # Check if recent residual is anomalous
                if abs(recent_residual) > 2 * residual_std:
                    # Calculate expected seasonal value
                    expected_seasonal = (
                        decomposition.trend.iloc[-1] + 
                        decomposition.seasonal.iloc[-1]
                    )
                    
                    severity = self._classify_pattern_severity(
                        abs(recent_residual) / residual_std,
                        confidence=0.75
                    )
                    
                    anomaly = Anomaly(
                        company=company,
                        metric_name=metric_name,
                        current_value=data[-1],
                        expected_value=expected_seasonal,
                        deviation_score=abs(recent_residual) / residual_std,
                        severity=severity,
                        anomaly_type=AnomalyType.SEASONAL_ANOMALY,
                        explanation=self.explanation_templates['seasonal_anomaly'].format(
                            description=f"{metric_name} deviates from seasonal pattern by {recent_residual:.3f}"
                        ),
                        detection_method="seasonal_decomposition",
                        confidence=min(0.85, abs(recent_residual) / residual_std / 3),
                        potential_causes=self._get_seasonal_anomaly_causes(metric_name)
                    )
                    
                    seasonal_anomalies.append(anomaly)
                    
            except Exception as e:
                logger.warning(f"Seasonal anomaly detection failed for {metric_name}: {e}")
                continue
        
        return seasonal_anomalies
    
    async def _detect_volatility_patterns(
        self,
        company: str,
        metrics_data: Dict[str, List[float]]
    ) -> List[Anomaly]:
        """Detect unusual volatility patterns"""
        
        volatility_anomalies = []
        
        for metric_name, data in metrics_data.items():
            if len(data) < 10:
                continue
            
            try:
                series = pd.Series(data)
                
                # Calculate rolling volatility
                returns = series.pct_change().dropna()
                rolling_vol = returns.rolling(window=5).std()
                
                # Calculate volatility metrics
                recent_vol = rolling_vol.iloc[-3:].mean()
                historical_vol = rolling_vol.iloc[:-3].mean()
                vol_threshold = historical_vol * 2
                
                # Detect volatility spikes
                if recent_vol > vol_threshold:
                    volatility_ratio = recent_vol / historical_vol
                    
                    severity = self._classify_pattern_severity(
                        volatility_ratio,
                        confidence=0.7
                    )
                    
                    anomaly = Anomaly(
                        company=company,
                        metric_name=metric_name,
                        current_value=data[-1],
                        expected_value=series.mean(),
                        deviation_score=volatility_ratio,
                        severity=severity,
                        anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                        explanation=self.explanation_templates['volatility_spike'].format(
                            description=f"{metric_name} volatility increased {volatility_ratio:.1f}x above normal"
                        ),
                        detection_method="volatility_analysis",
                        confidence=min(0.8, volatility_ratio / 4),
                        potential_causes=self._get_volatility_causes(metric_name)
                    )
                    
                    volatility_anomalies.append(anomaly)
                    
            except Exception as e:
                logger.warning(f"Volatility pattern detection failed for {metric_name}: {e}")
                continue
        
        return volatility_anomalies
    
    async def _detect_correlation_patterns(
        self,
        company: str,
        metrics_data: Dict[str, List[float]]
    ) -> List[Anomaly]:
        """Detect correlation pattern breaks between metrics"""
        
        correlation_anomalies = []
        
        if len(metrics_data) < 2:
            return correlation_anomalies
        
        try:
            # Create DataFrame from metrics data
            min_length = min(len(data) for data in metrics_data.values())
            if min_length < 10:
                return correlation_anomalies
            
            # Align data lengths
            aligned_data = {}
            for metric, data in metrics_data.items():
                aligned_data[metric] = data[-min_length:]
            
            df = pd.DataFrame(aligned_data)
            
            # Calculate rolling correlations
            window = min(8, min_length // 2)
            
            metric_pairs = [(col1, col2) for i, col1 in enumerate(df.columns) 
                           for col2 in df.columns[i+1:]]
            
            for metric1, metric2 in metric_pairs:
                try:
                    # Calculate historical and recent correlations
                    historical_corr = df[[metric1, metric2]].iloc[:-3].corr().iloc[0, 1]
                    recent_corr = df[[metric1, metric2]].iloc[-window:].corr().iloc[0, 1]
                    
                    # Skip if correlations are NaN
                    if np.isnan(historical_corr) or np.isnan(recent_corr):
                        continue
                    
                    # Detect significant correlation breaks
                    corr_change = abs(recent_corr - historical_corr)
                    
                    if corr_change > 0.4:  # Significant correlation change
                        severity = self._classify_pattern_severity(
                            corr_change * 2,
                            confidence=0.65
                        )
                        
                        anomaly = Anomaly(
                            company=company,
                            metric_name=f"{metric1}_vs_{metric2}",
                            current_value=recent_corr,
                            expected_value=historical_corr,
                            deviation_score=corr_change,
                            severity=severity,
                            anomaly_type=AnomalyType.CORRELATION_BREAK,
                            explanation=self.explanation_templates['correlation_break'].format(
                                description=f"Correlation between {metric1} and {metric2} changed from {historical_corr:.3f} to {recent_corr:.3f}"
                            ),
                            detection_method="correlation_analysis",
                            confidence=min(0.8, corr_change * 2),
                            potential_causes=self._get_correlation_break_causes(metric1, metric2)
                        )
                        
                        correlation_anomalies.append(anomaly)
                        
                except Exception as e:
                    logger.warning(f"Correlation analysis failed for {metric1}-{metric2}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Correlation pattern detection failed: {e}")
        
        return correlation_anomalies
    
    async def _detect_clustering_anomalies(
        self,
        company: str,
        metrics_data: Dict[str, List[float]]
    ) -> List[Anomaly]:
        """Detect anomalies using clustering-based approach"""
        
        clustering_anomalies = []
        
        try:
            # Prepare data for clustering
            min_length = min(len(data) for data in metrics_data.values())
            if min_length < 8 or len(metrics_data) < 2:
                return clustering_anomalies
            
            # Create feature matrix
            feature_matrix = []
            for i in range(min_length):
                features = [data[i] for data in metrics_data.values()]
                feature_matrix.append(features)
            
            feature_matrix = np.array(feature_matrix)
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = dbscan.fit_predict(scaled_features)
            
            # Identify outliers (label = -1)
            outlier_indices = np.where(cluster_labels == -1)[0]
            
            # Focus on recent outliers
            recent_indices = list(range(max(0, min_length - 3), min_length))
            recent_outliers = [idx for idx in outlier_indices if idx in recent_indices]
            
            for outlier_idx in recent_outliers:
                # Calculate distance to nearest cluster center
                outlier_point = scaled_features[outlier_idx]
                
                # Find distances to all non-outlier points
                non_outlier_points = scaled_features[cluster_labels != -1]
                if len(non_outlier_points) > 0:
                    distances = np.linalg.norm(non_outlier_points - outlier_point, axis=1)
                    min_distance = np.min(distances)
                    
                    severity = self._classify_pattern_severity(
                        min_distance,
                        confidence=0.6
                    )
                    
                    # Create anomaly for the most significant metric
                    metric_contributions = np.abs(outlier_point)
                    dominant_metric_idx = np.argmax(metric_contributions)
                    dominant_metric = list(metrics_data.keys())[dominant_metric_idx]
                    
                    anomaly = Anomaly(
                        company=company,
                        metric_name=dominant_metric,
                        current_value=list(metrics_data.values())[dominant_metric_idx][outlier_idx],
                        expected_value=np.mean(list(metrics_data.values())[dominant_metric_idx]),
                        deviation_score=min_distance,
                        severity=severity,
                        anomaly_type=AnomalyType.PATTERN_DEVIATION,
                        explanation=self.explanation_templates['clustering_anomaly'].format(
                            description=f"Multi-metric pattern outlier detected with distance {min_distance:.3f}"
                        ),
                        detection_method="clustering_analysis",
                        confidence=min(0.75, min_distance / 2),
                        potential_causes=self._get_clustering_anomaly_causes(dominant_metric)
                    )
                    
                    clustering_anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Clustering anomaly detection failed: {e}")
        
        return clustering_anomalies
    
    def _classify_pattern_severity(
        self,
        magnitude: float,
        confidence: float
    ) -> AnomalySeverity:
        """Classify anomaly severity based on magnitude and confidence"""
        
        severity_score = magnitude * confidence
        
        if severity_score >= 4.0:
            return AnomalySeverity.CRITICAL
        elif severity_score >= 2.5:
            return AnomalySeverity.HIGH
        elif severity_score >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _generate_pattern_summary(
        self,
        pattern_types: List[str],
        total_anomalies: int
    ) -> str:
        """Generate summary of detected patterns"""
        
        if total_anomalies == 0:
            return "No significant pattern anomalies detected"
        
        pattern_counts = {}
        for pattern_type in pattern_types:
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        summary_parts = [f"Detected {total_anomalies} pattern anomalies"]
        
        if pattern_counts:
            pattern_descriptions = []
            for pattern_type, count in pattern_counts.items():
                pattern_name = pattern_type.replace('_', ' ').title()
                pattern_descriptions.append(f"{count} {pattern_name}")
            
            summary_parts.append(f"Types: {', '.join(pattern_descriptions)}")
        
        return ". ".join(summary_parts)
    
    def _get_trend_break_causes(self, metric_name: str, trend_direction: str) -> List[str]:
        """Get potential causes for trend breaks"""
        causes = [
            f"Significant business change affecting {metric_name}",
            "Market condition shift or external factors",
            "Management strategy or operational changes",
            "Seasonal or cyclical pattern disruption"
        ]
        
        if "upward" in trend_direction:
            causes.extend([
                "Positive business developments or improvements",
                "Market expansion or competitive advantages"
            ])
        else:
            causes.extend([
                "Business challenges or market headwinds",
                "Operational difficulties or competitive pressure"
            ])
        
        return causes[:4]
    
    def _get_seasonal_anomaly_causes(self, metric_name: str) -> List[str]:
        """Get potential causes for seasonal anomalies"""
        return [
            "Unusual seasonal factors or weather patterns",
            "Market timing changes or customer behavior shifts",
            "Supply chain disruptions affecting seasonality",
            "Competitive actions or industry changes",
            "Economic conditions affecting seasonal patterns"
        ]
    
    def _get_volatility_causes(self, metric_name: str) -> List[str]:
        """Get potential causes for volatility spikes"""
        return [
            "Market uncertainty or external shocks",
            "Business model changes or operational volatility",
            "Competitive dynamics or industry disruption",
            "Regulatory changes or policy uncertainty",
            "Management decisions or strategic shifts"
        ]
    
    def _get_correlation_break_causes(self, metric1: str, metric2: str) -> List[str]:
        """Get potential causes for correlation breaks"""
        return [
            f"Structural change in relationship between {metric1} and {metric2}",
            "Business model evolution or operational changes",
            "Market dynamics affecting metric relationships",
            "External factors disrupting normal correlations",
            "Management actions changing business relationships"
        ]
    
    def _get_clustering_anomaly_causes(self, metric_name: str) -> List[str]:
        """Get potential causes for clustering anomalies"""
        return [
            "Multi-dimensional business pattern change",
            "Complex interaction of multiple factors",
            "Systemic business or market shift",
            "Operational model changes affecting multiple metrics",
            "External factors with broad business impact"
        ]


# Global pattern detector instance
_pattern_detector = None


async def get_pattern_detector() -> PatternAnomalyDetector:
    """Get or create pattern anomaly detector instance"""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PatternAnomalyDetector()
    return _pattern_detector


# Enhanced service functions
async def analyze_pattern_anomalies(
    company: str,
    metrics: List[str] = None,
    lookback_periods: int = 20
) -> PatternAnomalies:
    """Analyze pattern-based anomalies for a company"""
    
    if metrics is None:
        metrics = ['revenue', 'profit_margin', 'cash_flow', 'debt_ratio']
    
    # Get anomaly engine to fetch data
    engine = await get_anomaly_engine()
    
    # Collect metrics data
    metrics_data = {}
    for metric in metrics:
        data = await engine._get_metric_data(company, metric, lookback_periods)
        if len(data) >= 8:  # Minimum data requirement
            metrics_data[metric] = data
    
    if not metrics_data:
        return PatternAnomalies(
            company=company,
            analysis_period="no_data",
            detected_patterns=[],
            pattern_confidence=0.0,
            anomalous_patterns=[],
            pattern_explanation="No sufficient data available for pattern analysis"
        )
    
    # Analyze patterns
    detector = await get_pattern_detector()
    return await detector.analyze_pattern_anomalies(company, metrics_data)


async def classify_anomaly_severity(
    anomaly_score: float,
    confidence: float,
    historical_context: Dict[str, Any] = None
) -> AnomalySeverity:
    """Classify anomaly severity using enhanced criteria"""
    
    # Base severity from score and confidence
    base_severity_score = anomaly_score * confidence
    
    # Adjust based on historical context
    if historical_context:
        # Check if this is a record high/low
        if historical_context.get('is_record_extreme', False):
            base_severity_score *= 1.5
        
        # Check volatility context
        volatility_factor = historical_context.get('volatility_factor', 1.0)
        if volatility_factor > 2.0:
            base_severity_score *= 1.2
        
        # Check trend context
        if historical_context.get('breaks_long_term_trend', False):
            base_severity_score *= 1.3
    
    # Classify severity
    if base_severity_score >= 4.0:
        return AnomalySeverity.CRITICAL
    elif base_severity_score >= 2.5:
        return AnomalySeverity.HIGH
    elif base_severity_score >= 1.5:
        return AnomalySeverity.MEDIUM
    else:
        return AnomalySeverity.LOW


async def generate_anomaly_explanation(
    anomaly: Anomaly,
    include_technical_details: bool = False
) -> str:
    """Generate detailed explanation for an anomaly"""
    
    explanation_parts = []
    
    # Basic anomaly description
    deviation_pct = ((anomaly.current_value - anomaly.expected_value) / anomaly.expected_value) * 100
    direction = "above" if anomaly.current_value > anomaly.expected_value else "below"
    
    explanation_parts.append(
        f"Anomaly detected in {anomaly.metric_name} for {anomaly.company}. "
        f"Current value ({anomaly.current_value:.2f}) is {abs(deviation_pct):.1f}% "
        f"{direction} expected value ({anomaly.expected_value:.2f})."
    )
    
    # Severity and confidence
    explanation_parts.append(
        f"Severity: {anomaly.severity.value.upper()} "
        f"(confidence: {anomaly.confidence:.1%})"
    )
    
    # Detection method and type
    if include_technical_details:
        explanation_parts.append(
            f"Detected using {anomaly.detection_method} as {anomaly.anomaly_type.value.replace('_', ' ')}"
        )
    
    # Historical context
    if anomaly.historical_context:
        explanation_parts.append(f"Context: {anomaly.historical_context}")
    
    # Potential causes
    if anomaly.potential_causes:
        causes_text = ", ".join(anomaly.potential_causes[:3])
        explanation_parts.append(f"Potential causes: {causes_text}")
    
    return " ".join(explanation_parts)


class RiskAssessmentEngine:
    """Risk assessment and anomaly management system"""
    
    def __init__(self):
        self.risk_models = {}
        self.escalation_rules = {}
        self.resolution_tracking = {}
        
        # Risk scoring weights
        self.risk_weights = {
            'severity': 0.3,
            'confidence': 0.2,
            'historical_frequency': 0.15,
            'business_impact': 0.2,
            'correlation_risk': 0.15
        }
        
        # Risk categories and their impact multipliers
        self.risk_categories = {
            'market_risk': 1.2,
            'operational_risk': 1.0,
            'regulatory_risk': 1.5,
            'reputational_risk': 1.1
        }
    
    async def assess_anomaly_risk(
        self,
        anomaly: Anomaly,
        company_context: Dict[str, Any] = None,
        market_context: Dict[str, Any] = None
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment for detected anomalies
        
        Args:
            anomaly: The detected anomaly
            company_context: Company-specific context (size, industry, etc.)
            market_context: Market conditions and external factors
        
        Returns:
            RiskAssessment with detailed risk analysis
        """
        
        try:
            # Calculate base risk components
            severity_risk = self._calculate_severity_risk(anomaly)
            confidence_risk = self._calculate_confidence_risk(anomaly)
            historical_risk = await self._calculate_historical_risk(anomaly)
            business_impact_risk = await self._calculate_business_impact_risk(
                anomaly, company_context
            )
            correlation_risk = await self._calculate_correlation_risk(anomaly)
            
            # Calculate overall risk score
            risk_score = (
                severity_risk * self.risk_weights['severity'] +
                confidence_risk * self.risk_weights['confidence'] +
                historical_risk * self.risk_weights['historical_frequency'] +
                business_impact_risk * self.risk_weights['business_impact'] +
                correlation_risk * self.risk_weights['correlation_risk']
            )
            
            # Calculate specific risk categories
            market_risk = await self._assess_market_risk(anomaly, market_context)
            operational_risk = await self._assess_operational_risk(anomaly, company_context)
            regulatory_risk = await self._assess_regulatory_risk(anomaly)
            reputational_risk = await self._assess_reputational_risk(anomaly)
            
            # Determine financial impact and probability
            financial_impact = self._estimate_financial_impact(anomaly, company_context)
            probability_of_impact = self._calculate_impact_probability(
                risk_score, anomaly.confidence
            )
            
            # Determine time horizon
            time_horizon = self._determine_time_horizon(anomaly)
            
            # Generate recommendations
            recommended_actions = await self._generate_risk_recommendations(
                anomaly, risk_score, {
                    'market': market_risk,
                    'operational': operational_risk,
                    'regulatory': regulatory_risk,
                    'reputational': reputational_risk
                }
            )
            
            monitoring_recommendations = self._generate_monitoring_recommendations(
                anomaly, risk_score
            )
            
            # Set escalation threshold
            escalation_threshold = self._calculate_escalation_threshold(
                risk_score, anomaly.severity
            )
            
            return RiskAssessment(
                anomaly_id=anomaly.id,
                risk_score=min(1.0, risk_score),
                financial_impact=financial_impact,
                probability_of_impact=probability_of_impact,
                time_horizon=time_horizon,
                market_risk=market_risk,
                operational_risk=operational_risk,
                regulatory_risk=regulatory_risk,
                reputational_risk=reputational_risk,
                recommended_actions=recommended_actions,
                monitoring_recommendations=monitoring_recommendations,
                escalation_threshold=escalation_threshold
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed for anomaly {anomaly.id}: {e}")
            # Return minimal risk assessment
            return RiskAssessment(
                anomaly_id=anomaly.id,
                risk_score=0.5,
                financial_impact="Unable to assess",
                probability_of_impact=0.5,
                time_horizon="Unknown",
                market_risk=0.5,
                operational_risk=0.5,
                regulatory_risk=0.5,
                reputational_risk=0.5,
                recommended_actions=["Investigate anomaly manually"],
                monitoring_recommendations=["Monitor metric closely"],
                escalation_threshold=0.7
            )
    
    def _calculate_severity_risk(self, anomaly: Anomaly) -> float:
        """Calculate risk component based on anomaly severity"""
        severity_scores = {
            AnomalySeverity.LOW: 0.2,
            AnomalySeverity.MEDIUM: 0.5,
            AnomalySeverity.HIGH: 0.8,
            AnomalySeverity.CRITICAL: 1.0
        }
        return severity_scores.get(anomaly.severity, 0.5)
    
    def _calculate_confidence_risk(self, anomaly: Anomaly) -> float:
        """Calculate risk component based on detection confidence"""
        # Higher confidence means higher risk (more certain about the anomaly)
        return min(1.0, anomaly.confidence * 1.2)
    
    async def _calculate_historical_risk(self, anomaly: Anomaly) -> float:
        """Calculate risk based on historical frequency of similar anomalies"""
        
        try:
            db = await get_database()
            
            # Query historical anomalies for same company and metric
            query = """
                SELECT COUNT(*) as anomaly_count,
                       AVG(CASE WHEN status IN ('resolved', 'explained') THEN 1 ELSE 0 END) as resolution_rate
                FROM anomalies
                WHERE company = $1 
                AND metric_name = $2
                AND severity = $3
                AND created_at >= NOW() - INTERVAL '1 year'
            """
            
            result = await db.fetchrow(
                query, 
                anomaly.company, 
                anomaly.metric_name, 
                anomaly.severity.value
            )
            
            if result:
                anomaly_count = result['anomaly_count'] or 0
                resolution_rate = result['resolution_rate'] or 0.0
                
                # Higher frequency = higher risk, but good resolution rate reduces risk
                frequency_risk = min(1.0, anomaly_count / 10)  # Normalize to 0-1
                resolution_factor = 1.0 - resolution_rate  # Lower resolution rate = higher risk
                
                return frequency_risk * resolution_factor
            
            return 0.3  # Default moderate risk for unknown history
            
        except Exception as e:
            logger.error(f"Historical risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_business_impact_risk(
        self, 
        anomaly: Anomaly, 
        company_context: Dict[str, Any] = None
    ) -> float:
        """Calculate risk based on potential business impact"""
        
        # Base impact by metric type
        metric_impacts = {
            'revenue': 0.9,
            'profit': 0.8,
            'cash_flow': 0.8,
            'margin': 0.7,
            'debt': 0.6,
            'expenses': 0.5,
            'volume': 0.4
        }
        
        # Find matching metric type
        base_impact = 0.5  # Default
        metric_lower = anomaly.metric_name.lower()
        
        for metric_type, impact in metric_impacts.items():
            if metric_type in metric_lower:
                base_impact = impact
                break
        
        # Adjust based on company context
        if company_context:
            # Company size factor
            company_size = company_context.get('size', 'medium')
            if company_size == 'large':
                base_impact *= 1.2  # Larger companies have higher systemic risk
            elif company_size == 'small':
                base_impact *= 0.8
            
            # Industry factor
            industry = company_context.get('industry', '')
            high_risk_industries = ['financial', 'healthcare', 'energy', 'utilities']
            if any(sector in industry.lower() for sector in high_risk_industries):
                base_impact *= 1.1
        
        # Adjust based on deviation magnitude
        deviation_factor = min(2.0, anomaly.deviation_score / 2)
        
        return min(1.0, base_impact * deviation_factor)
    
    async def _calculate_correlation_risk(self, anomaly: Anomaly) -> float:
        """Calculate risk based on correlation with other anomalies"""
        
        try:
            db = await get_database()
            
            # Look for concurrent anomalies in the same company
            query = """
                SELECT COUNT(*) as concurrent_anomalies
                FROM anomalies
                WHERE company = $1 
                AND id != $2
                AND created_at >= $3 - INTERVAL '7 days'
                AND created_at <= $3 + INTERVAL '7 days'
                AND status = 'detected'
            """
            
            result = await db.fetchrow(
                query, 
                anomaly.company, 
                anomaly.id, 
                anomaly.created_at
            )
            
            if result:
                concurrent_count = result['concurrent_anomalies'] or 0
                # More concurrent anomalies = higher systemic risk
                correlation_risk = min(1.0, concurrent_count / 5)
                return correlation_risk
            
            return 0.1  # Low correlation risk if no concurrent anomalies
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.1
    
    async def _assess_market_risk(
        self, 
        anomaly: Anomaly, 
        market_context: Dict[str, Any] = None
    ) -> float:
        """Assess market-related risk factors"""
        
        base_market_risk = 0.3
        
        # Adjust based on market context
        if market_context:
            volatility = market_context.get('market_volatility', 0.5)
            economic_indicators = market_context.get('economic_indicators', {})
            
            # High market volatility increases risk
            base_market_risk += volatility * 0.3
            
            # Economic indicators
            if economic_indicators.get('recession_probability', 0) > 0.3:
                base_market_risk += 0.2
            
            if economic_indicators.get('inflation_rate', 0) > 0.05:  # >5% inflation
                base_market_risk += 0.1
        
        # Metric-specific market risk
        metric_lower = anomaly.metric_name.lower()
        if 'revenue' in metric_lower or 'sales' in metric_lower:
            base_market_risk *= 1.2  # Revenue more sensitive to market
        elif 'cost' in metric_lower or 'expense' in metric_lower:
            base_market_risk *= 0.8  # Costs less directly market-sensitive
        
        return min(1.0, base_market_risk)
    
    async def _assess_operational_risk(
        self, 
        anomaly: Anomaly, 
        company_context: Dict[str, Any] = None
    ) -> float:
        """Assess operational risk factors"""
        
        base_operational_risk = 0.4
        
        # Metric-specific operational risk
        metric_lower = anomaly.metric_name.lower()
        
        operational_metrics = ['efficiency', 'productivity', 'cost', 'expense', 'margin']
        if any(term in metric_lower for term in operational_metrics):
            base_operational_risk = 0.7  # Higher operational risk for operational metrics
        
        # Adjust based on anomaly type
        if anomaly.anomaly_type in [AnomalyType.PATTERN_DEVIATION, AnomalyType.TREND_BREAK]:
            base_operational_risk *= 1.2  # Pattern changes suggest operational issues
        
        # Company context adjustments
        if company_context:
            operational_maturity = company_context.get('operational_maturity', 'medium')
            if operational_maturity == 'low':
                base_operational_risk *= 1.3
            elif operational_maturity == 'high':
                base_operational_risk *= 0.8
        
        return min(1.0, base_operational_risk)
    
    async def _assess_regulatory_risk(self, anomaly: Anomaly) -> float:
        """Assess regulatory and compliance risk factors"""
        
        base_regulatory_risk = 0.2
        
        # Metrics that might have regulatory implications
        regulatory_metrics = ['debt', 'leverage', 'capital', 'liquidity', 'compliance']
        metric_lower = anomaly.metric_name.lower()
        
        if any(term in metric_lower for term in regulatory_metrics):
            base_regulatory_risk = 0.6
        
        # High severity anomalies have higher regulatory risk
        if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
            base_regulatory_risk *= 1.5
        
        return min(1.0, base_regulatory_risk)
    
    async def _assess_reputational_risk(self, anomaly: Anomaly) -> float:
        """Assess reputational risk factors"""
        
        base_reputational_risk = 0.3
        
        # Public-facing metrics have higher reputational risk
        public_metrics = ['revenue', 'profit', 'growth', 'performance', 'customer']
        metric_lower = anomaly.metric_name.lower()
        
        if any(term in metric_lower for term in public_metrics):
            base_reputational_risk = 0.5
        
        # Negative anomalies have higher reputational risk
        if anomaly.current_value < anomaly.expected_value:
            base_reputational_risk *= 1.2
        
        # Critical anomalies have high reputational risk
        if anomaly.severity == AnomalySeverity.CRITICAL:
            base_reputational_risk *= 1.4
        
        return min(1.0, base_reputational_risk)
    
    def _estimate_financial_impact(
        self, 
        anomaly: Anomaly, 
        company_context: Dict[str, Any] = None
    ) -> str:
        """Estimate potential financial impact"""
        
        # Calculate relative impact based on deviation
        deviation_pct = abs((anomaly.current_value - anomaly.expected_value) / anomaly.expected_value)
        
        # Estimate impact categories
        if deviation_pct < 0.05:  # <5%
            impact_category = "Minimal"
        elif deviation_pct < 0.15:  # 5-15%
            impact_category = "Low"
        elif deviation_pct < 0.30:  # 15-30%
            impact_category = "Moderate"
        elif deviation_pct < 0.50:  # 30-50%
            impact_category = "High"
        else:  # >50%
            impact_category = "Severe"
        
        # Adjust based on metric importance
        metric_lower = anomaly.metric_name.lower()
        if 'revenue' in metric_lower:
            if deviation_pct > 0.1:  # Revenue changes >10% are significant
                impact_category = "High" if impact_category in ["Minimal", "Low"] else impact_category
        
        # Add context if available
        if company_context and 'annual_revenue' in company_context:
            annual_revenue = company_context['annual_revenue']
            estimated_impact = annual_revenue * deviation_pct
            
            if estimated_impact > 1000000:  # >$1M
                return f"{impact_category} (Est. ${estimated_impact/1000000:.1f}M impact)"
            elif estimated_impact > 1000:  # >$1K
                return f"{impact_category} (Est. ${estimated_impact/1000:.0f}K impact)"
        
        return f"{impact_category} financial impact"
    
    def _calculate_impact_probability(self, risk_score: float, confidence: float) -> float:
        """Calculate probability that the risk will materialize"""
        
        # Base probability from risk score
        base_probability = risk_score * 0.7
        
        # Adjust based on detection confidence
        confidence_factor = confidence * 0.3
        
        # Combined probability
        probability = base_probability + confidence_factor
        
        return min(1.0, probability)
    
    def _determine_time_horizon(self, anomaly: Anomaly) -> str:
        """Determine time horizon for potential impact"""
        
        # Based on anomaly type and severity
        if anomaly.anomaly_type == AnomalyType.VOLATILITY_SPIKE:
            return "Immediate (1-7 days)"
        elif anomaly.anomaly_type in [AnomalyType.TREND_BREAK, AnomalyType.PATTERN_DEVIATION]:
            return "Short-term (1-4 weeks)"
        elif anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
            return "Short-term (1-4 weeks)"
        else:
            return "Medium-term (1-3 months)"
    
    async def _generate_risk_recommendations(
        self,
        anomaly: Anomaly,
        risk_score: float,
        risk_categories: Dict[str, float]
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        
        # General recommendations based on risk score
        if risk_score >= 0.8:
            recommendations.append("Immediate investigation and response required")
            recommendations.append("Escalate to senior management")
        elif risk_score >= 0.6:
            recommendations.append("Prioritize investigation within 24 hours")
            recommendations.append("Prepare contingency plans")
        elif risk_score >= 0.4:
            recommendations.append("Schedule detailed analysis within 48 hours")
        else:
            recommendations.append("Monitor closely and investigate when resources allow")
        
        # Specific recommendations based on dominant risk category
        max_risk_category = max(risk_categories.items(), key=lambda x: x[1])
        dominant_risk, risk_level = max_risk_category
        
        if risk_level >= 0.6:
            if dominant_risk == 'market':
                recommendations.append("Assess market conditions and competitive landscape")
                recommendations.append("Review pricing and market positioning strategies")
            elif dominant_risk == 'operational':
                recommendations.append("Review operational processes and efficiency metrics")
                recommendations.append("Investigate potential operational disruptions")
            elif dominant_risk == 'regulatory':
                recommendations.append("Review compliance requirements and regulatory changes")
                recommendations.append("Consult with legal and compliance teams")
            elif dominant_risk == 'reputational':
                recommendations.append("Prepare communication strategy for stakeholders")
                recommendations.append("Monitor public perception and media coverage")
        
        # Metric-specific recommendations
        metric_lower = anomaly.metric_name.lower()
        if 'revenue' in metric_lower:
            recommendations.append("Analyze customer behavior and market demand")
        elif 'cost' in metric_lower or 'expense' in metric_lower:
            recommendations.append("Review cost structure and vendor relationships")
        elif 'cash' in metric_lower:
            recommendations.append("Assess liquidity position and cash management")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def _generate_monitoring_recommendations(
        self, 
        anomaly: Anomaly, 
        risk_score: float
    ) -> List[str]:
        """Generate monitoring recommendations"""
        
        recommendations = []
        
        # Frequency based on risk score
        if risk_score >= 0.8:
            recommendations.append("Monitor daily with real-time alerts")
        elif risk_score >= 0.6:
            recommendations.append("Monitor every 2-3 days")
        elif risk_score >= 0.4:
            recommendations.append("Monitor weekly")
        else:
            recommendations.append("Monitor bi-weekly")
        
        # Specific monitoring recommendations
        recommendations.append(f"Track {anomaly.metric_name} trend and volatility")
        recommendations.append("Monitor related metrics for correlation patterns")
        
        if anomaly.anomaly_type == AnomalyType.SEASONAL_ANOMALY:
            recommendations.append("Compare against seasonal baselines")
        elif anomaly.anomaly_type == AnomalyType.CORRELATION_BREAK:
            recommendations.append("Monitor metric relationships and dependencies")
        
        recommendations.append("Set up automated alerts for threshold breaches")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_escalation_threshold(
        self, 
        risk_score: float, 
        severity: AnomalySeverity
    ) -> float:
        """Calculate threshold for automatic escalation"""
        
        # Base threshold from severity
        severity_thresholds = {
            AnomalySeverity.LOW: 0.8,
            AnomalySeverity.MEDIUM: 0.7,
            AnomalySeverity.HIGH: 0.6,
            AnomalySeverity.CRITICAL: 0.5
        }
        
        base_threshold = severity_thresholds.get(severity, 0.7)
        
        # Adjust based on current risk score
        if risk_score >= 0.8:
            base_threshold *= 0.8  # Lower threshold for high-risk anomalies
        elif risk_score <= 0.3:
            base_threshold *= 1.2  # Higher threshold for low-risk anomalies
        
        return min(1.0, max(0.3, base_threshold))


# Anomaly Management System
class AnomalyManager:
    """Comprehensive anomaly management and tracking system"""
    
    def __init__(self):
        self.active_anomalies = {}
        self.resolution_workflows = {}
        self.escalation_rules = {}
    
    async def track_anomaly_lifecycle(
        self, 
        anomaly: Anomaly,
        risk_assessment: RiskAssessment
    ) -> Dict[str, Any]:
        """Track anomaly through its complete lifecycle"""
        
        lifecycle_data = {
            'anomaly_id': anomaly.id,
            'detection_timestamp': anomaly.created_at,
            'current_status': anomaly.status.value,
            'risk_score': risk_assessment.risk_score,
            'escalation_threshold': risk_assessment.escalation_threshold,
            'recommended_actions': risk_assessment.recommended_actions,
            'monitoring_schedule': risk_assessment.monitoring_recommendations,
            'resolution_target': self._calculate_resolution_target(risk_assessment),
            'stakeholders_notified': [],
            'investigation_notes': [],
            'resolution_actions': []
        }
        
        # Store in tracking system
        self.active_anomalies[str(anomaly.id)] = lifecycle_data
        
        # Check if immediate escalation is needed
        if risk_assessment.risk_score >= risk_assessment.escalation_threshold:
            await self._trigger_escalation(anomaly, risk_assessment)
        
        return lifecycle_data
    
    async def update_anomaly_status(
        self,
        anomaly_id: str,
        new_status: AnomalyStatus,
        notes: str = None,
        resolved_by: str = None
    ) -> bool:
        """Update anomaly status and tracking information"""
        
        try:
            db = await get_database()
            
            # Update database
            query = """
                UPDATE anomalies 
                SET status = $1, 
                    investigated_by = COALESCE($2, investigated_by),
                    resolution_notes = COALESCE($3, resolution_notes),
                    resolved_at = CASE WHEN $1 IN ('resolved', 'explained') THEN NOW() ELSE resolved_at END
                WHERE id = $4
            """
            
            await db.execute(query, new_status.value, resolved_by, notes, anomaly_id)
            
            # Update tracking system
            if anomaly_id in self.active_anomalies:
                self.active_anomalies[anomaly_id]['current_status'] = new_status.value
                if notes:
                    self.active_anomalies[anomaly_id]['investigation_notes'].append({
                        'timestamp': datetime.now(),
                        'note': notes,
                        'author': resolved_by
                    })
                
                # Remove from active tracking if resolved
                if new_status in [AnomalyStatus.RESOLVED, AnomalyStatus.FALSE_POSITIVE]:
                    del self.active_anomalies[anomaly_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update anomaly status: {e}")
            return False
    
    async def get_active_anomalies(
        self, 
        company: str = None,
        severity_filter: List[AnomalySeverity] = None
    ) -> List[Dict[str, Any]]:
        """Get list of active anomalies with tracking information"""
        
        try:
            db = await get_database()
            
            # Build query
            query = """
                SELECT a.*, 
                       EXTRACT(EPOCH FROM (NOW() - a.created_at))/3600 as hours_since_detection
                FROM anomalies a
                WHERE a.status NOT IN ('resolved', 'false_positive')
            """
            params = []
            
            if company:
                query += " AND a.company = $1"
                params.append(company)
            
            if severity_filter:
                severity_values = [s.value for s in severity_filter]
                query += f" AND a.severity = ANY(${'2' if company else '1'})"
                params.append(severity_values)
            
            query += " ORDER BY a.created_at DESC"
            
            rows = await db.fetch(query, *params)
            
            active_anomalies = []
            for row in rows:
                anomaly_data = dict(row)
                
                # Add tracking information if available
                anomaly_id = str(row['id'])
                if anomaly_id in self.active_anomalies:
                    anomaly_data.update(self.active_anomalies[anomaly_id])
                
                active_anomalies.append(anomaly_data)
            
            return active_anomalies
            
        except Exception as e:
            logger.error(f"Failed to get active anomalies: {e}")
            return []
    
    def _calculate_resolution_target(self, risk_assessment: RiskAssessment) -> datetime:
        """Calculate target resolution time based on risk assessment"""
        
        base_hours = 72  # Default 3 days
        
        # Adjust based on risk score
        if risk_assessment.risk_score >= 0.8:
            base_hours = 24  # 1 day for high risk
        elif risk_assessment.risk_score >= 0.6:
            base_hours = 48  # 2 days for medium-high risk
        elif risk_assessment.risk_score <= 0.3:
            base_hours = 168  # 1 week for low risk
        
        return datetime.now() + timedelta(hours=base_hours)
    
    async def _trigger_escalation(
        self, 
        anomaly: Anomaly, 
        risk_assessment: RiskAssessment
    ):
        """Trigger escalation process for high-risk anomalies"""
        
        try:
            # Create escalation alert
            alert = AnomalyAlert(
                anomaly_id=anomaly.id,
                alert_level=anomaly.severity,
                recipient_groups=['management', 'risk_team'],
                message=f"High-risk anomaly detected: {anomaly.explanation}",
                delivery_channels=['email', 'dashboard'],
                acknowledgment_required=True
            )
            
            # Store alert (would integrate with notification system)
            logger.warning(f"ESCALATION: {alert.message}")
            
            # Update tracking
            if str(anomaly.id) in self.active_anomalies:
                self.active_anomalies[str(anomaly.id)]['escalated'] = True
                self.active_anomalies[str(anomaly.id)]['escalation_timestamp'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Escalation failed for anomaly {anomaly.id}: {e}")


# Global instances
_risk_engine = None
_anomaly_manager = None


async def get_risk_engine() -> RiskAssessmentEngine:
    """Get or create risk assessment engine instance"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskAssessmentEngine()
    return _risk_engine


async def get_anomaly_manager() -> AnomalyManager:
    """Get or create anomaly manager instance"""
    global _anomaly_manager
    if _anomaly_manager is None:
        _anomaly_manager = AnomalyManager()
    return _anomaly_manager


# Enhanced service functions for risk assessment and management
async def assess_anomaly_risk(
    anomaly: Anomaly,
    company_context: Dict[str, Any] = None,
    market_context: Dict[str, Any] = None
) -> RiskAssessment:
    """Assess risk for a detected anomaly"""
    engine = await get_risk_engine()
    return await engine.assess_anomaly_risk(anomaly, company_context, market_context)


async def manage_anomaly_lifecycle(
    anomaly: Anomaly,
    risk_assessment: RiskAssessment
) -> Dict[str, Any]:
    """Manage anomaly through its complete lifecycle"""
    manager = await get_anomaly_manager()
    return await manager.track_anomaly_lifecycle(anomaly, risk_assessment)


async def update_anomaly_resolution(
    anomaly_id: str,
    status: AnomalyStatus,
    notes: str = None,
    resolved_by: str = None
) -> bool:
    """Update anomaly resolution status"""
    manager = await get_anomaly_manager()
    return await manager.update_anomaly_status(anomaly_id, status, notes, resolved_by)


async def get_anomaly_dashboard_data(
    company: str = None
) -> Dict[str, Any]:
    """Get comprehensive anomaly dashboard data"""
    
    manager = await get_anomaly_manager()
    
    # Get active anomalies
    active_anomalies = await manager.get_active_anomalies(company)
    
    # Calculate summary statistics
    total_active = len(active_anomalies)
    severity_counts = {}
    high_risk_count = 0
    
    for anomaly in active_anomalies:
        severity = anomaly.get('severity', 'unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if anomaly.get('risk_score', 0) >= 0.7:
            high_risk_count += 1
    
    # Get historical data
    history = await get_anomaly_history(company or 'all', days_back=30)
    
    dashboard_data = {
        'summary': {
            'total_active_anomalies': total_active,
            'high_risk_anomalies': high_risk_count,
            'severity_distribution': severity_counts,
            'resolution_rate': history.resolution_rate,
            'avg_resolution_time_hours': history.average_resolution_time
        },
        'active_anomalies': active_anomalies[:10],  # Top 10 most recent
        'trends': {
            'total_anomalies_30d': history.total_anomalies,
            'anomalies_by_type': history.anomalies_by_type,
            'recurring_patterns': history.recurring_patterns
        }
    }
    
    return dashboard_data