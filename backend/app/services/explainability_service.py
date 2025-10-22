"""
Model explainability and decision tracking service
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum

from ..models.audit import ModelDecisionLog
from ..services.audit_service import audit_service
from ..database.connection import get_database_connection


class ModelType(str, Enum):
    """Types of models in the system"""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECASTING = "forecasting"
    RECOMMENDATION = "recommendation"
    DOCUMENT_QA = "document_qa"


class ExplanationType(str, Enum):
    """Types of explanations"""
    FEATURE_IMPORTANCE = "feature_importance"
    LIME = "lime"
    SHAP = "shap"
    ATTENTION_WEIGHTS = "attention_weights"
    RULE_BASED = "rule_based"
    NATURAL_LANGUAGE = "natural_language"


@dataclass
class ModelDecision:
    """Represents a model decision with explanation"""
    model_name: str
    model_version: str
    model_type: ModelType
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    decision_factors: Optional[List[str]] = None
    processing_time_ms: Optional[int] = None
    user_id: Optional[UUID] = None
    request_id: Optional[str] = None


@dataclass
class ExplanationResult:
    """Result of model explanation generation"""
    explanation_type: ExplanationType
    explanation_text: str
    feature_scores: Dict[str, float]
    confidence_score: float
    decision_boundary: Optional[Dict[str, Any]] = None
    counterfactual_examples: Optional[List[Dict[str, Any]]] = None


class ExplainabilityService:
    """Service for model explainability and decision tracking"""
    
    def __init__(self):
        self._model_registry = {}
        self._explanation_cache = {}
    
    async def initialize(self):
        """Initialize the explainability service"""
        await self._ensure_explainability_tables()
    
    async def _ensure_explainability_tables(self):
        """Ensure explainability tables exist"""
        async with get_database_connection() as conn:
            # Create model_versions table for version control
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_name VARCHAR(100) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    model_config JSONB NOT NULL,
                    training_data_hash VARCHAR(64),
                    performance_metrics JSONB,
                    deployment_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT true,
                    created_by UUID,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(model_name, version)
                );
            """)
            
            # Create model_performance table for tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    dataset_name VARCHAR(100),
                    metrics JSONB NOT NULL,
                    confusion_matrix JSONB,
                    feature_importance JSONB,
                    bias_metrics JSONB,
                    drift_score FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create explanation_cache table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_cache (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    input_hash VARCHAR(64) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    explanation_type VARCHAR(50) NOT NULL,
                    explanation_data JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    UNIQUE(input_hash, model_name, model_version, explanation_type)
                );
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_name_version ON model_versions(model_name, version);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_name_date ON model_performance(model_name, evaluation_date);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_explanation_cache_hash ON explanation_cache(input_hash);")
    
    async def register_model(
        self,
        model_name: str,
        version: str,
        model_type: ModelType,
        model_config: Dict[str, Any],
        training_data_hash: Optional[str] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        created_by: Optional[UUID] = None
    ) -> UUID:
        """Register a new model version"""
        
        async with get_database_connection() as conn:
            # Deactivate previous versions
            await conn.execute("""
                UPDATE model_versions 
                SET is_active = false 
                WHERE model_name = $1 AND is_active = true
            """, model_name)
            
            # Insert new version
            result = await conn.fetchrow("""
                INSERT INTO model_versions (
                    model_name, version, model_type, model_config,
                    training_data_hash, performance_metrics, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """,
                model_name, version, model_type.value, json.dumps(model_config),
                training_data_hash, json.dumps(performance_metrics) if performance_metrics else None,
                created_by
            )
            
            # Log model registration
            await audit_service.log_event(
                event_type="model_deployed",
                event_name=f"Model {model_name} v{version} Registered",
                description=f"Registered new version of {model_type.value} model",
                user_id=created_by,
                resource_type="model",
                resource_id=model_name,
                resource_name=f"{model_name} v{version}",
                event_data={
                    "model_name": model_name,
                    "version": version,
                    "model_type": model_type.value,
                    "config": model_config
                }
            )
            
            return result['id']
    
    async def log_model_decision(
        self,
        decision: ModelDecision,
        explanation_result: Optional[ExplanationResult] = None
    ) -> UUID:
        """Log a model decision with optional explanation"""
        
        # Prepare explanation data
        explanation_text = None
        feature_importance = None
        decision_factors = None
        
        if explanation_result:
            explanation_text = explanation_result.explanation_text
            feature_importance = explanation_result.feature_scores
            decision_factors = list(explanation_result.feature_scores.keys())[:10]  # Top 10 factors
        
        # Log the decision
        decision_id = await audit_service.log_model_decision(
            model_name=decision.model_name,
            model_version=decision.model_version,
            model_type=decision.model_type.value,
            input_data=decision.input_data,
            prediction=decision.prediction,
            user_id=decision.user_id,
            request_id=decision.request_id,
            confidence_score=decision.confidence_score,
            feature_importance=feature_importance,
            explanation=explanation_text,
            decision_factors=decision_factors,
            processing_time_ms=decision.processing_time_ms
        )
        
        return decision_id
    
    async def explain_sentiment_decision(
        self,
        model_name: str,
        model_version: str,
        input_text: str,
        prediction: Dict[str, Any],
        model_output: Optional[Dict[str, Any]] = None
    ) -> ExplanationResult:
        """Generate explanation for sentiment analysis decision"""
        
        # For sentiment analysis, we can use attention weights or feature importance
        # This is a simplified implementation - in practice, you'd use LIME, SHAP, or attention visualization
        
        # Extract key features from text
        words = input_text.lower().split()
        
        # Simulate feature importance (in practice, this would come from the actual model)
        sentiment_keywords = {
            'positive': ['good', 'great', 'excellent', 'strong', 'growth', 'profit', 'success', 'beat', 'exceed'],
            'negative': ['bad', 'poor', 'weak', 'decline', 'loss', 'miss', 'concern', 'risk', 'challenge'],
            'neutral': ['stable', 'maintain', 'continue', 'expect', 'forecast', 'guidance']
        }
        
        feature_scores = {}
        predicted_sentiment = prediction.get('sentiment', 'neutral')
        
        for word in words:
            if word in sentiment_keywords['positive']:
                score = 0.8 if predicted_sentiment == 'positive' else -0.3
            elif word in sentiment_keywords['negative']:
                score = 0.8 if predicted_sentiment == 'negative' else -0.3
            elif word in sentiment_keywords['neutral']:
                score = 0.5 if predicted_sentiment == 'neutral' else 0.1
            else:
                score = 0.1  # Neutral words
            
            feature_scores[word] = score
        
        # Sort by importance
        feature_scores = dict(sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:20])
        
        # Generate natural language explanation
        top_positive = [word for word, score in feature_scores.items() if score > 0.5][:3]
        top_negative = [word for word, score in feature_scores.items() if score < -0.2][:3]
        
        explanation_parts = []
        if top_positive:
            explanation_parts.append(f"Positive indicators: {', '.join(top_positive)}")
        if top_negative:
            explanation_parts.append(f"Negative indicators: {', '.join(top_negative)}")
        
        confidence = prediction.get('confidence', 0.5)
        explanation_parts.append(f"Overall confidence: {confidence:.2f}")
        
        explanation_text = ". ".join(explanation_parts)
        
        return ExplanationResult(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            explanation_text=explanation_text,
            feature_scores=feature_scores,
            confidence_score=confidence
        )
    
    async def explain_anomaly_decision(
        self,
        model_name: str,
        model_version: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> ExplanationResult:
        """Generate explanation for anomaly detection decision"""
        
        # Extract metrics and their values
        metrics = input_data.get('metrics', {})
        anomaly_score = prediction.get('anomaly_score', 0.0)
        is_anomaly = prediction.get('is_anomaly', False)
        
        # Calculate feature importance based on deviation from normal ranges
        feature_scores = {}
        
        # Simulate normal ranges (in practice, these would come from training data)
        normal_ranges = {
            'revenue_growth': (-0.1, 0.3),
            'profit_margin': (0.05, 0.25),
            'debt_ratio': (0.0, 0.6),
            'current_ratio': (1.0, 3.0),
            'roa': (0.02, 0.15),
            'roe': (0.05, 0.25)
        }
        
        explanation_parts = []
        
        for metric, value in metrics.items():
            if metric in normal_ranges:
                min_val, max_val = normal_ranges[metric]
                
                if value < min_val:
                    deviation = (min_val - value) / (max_val - min_val)
                    feature_scores[metric] = min(deviation, 1.0)
                    explanation_parts.append(f"{metric} is below normal range ({value:.3f} < {min_val})")
                elif value > max_val:
                    deviation = (value - max_val) / (max_val - min_val)
                    feature_scores[metric] = min(deviation, 1.0)
                    explanation_parts.append(f"{metric} is above normal range ({value:.3f} > {max_val})")
                else:
                    feature_scores[metric] = 0.0
        
        # Sort by importance
        feature_scores = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
        
        if is_anomaly:
            explanation_text = f"Anomaly detected (score: {anomaly_score:.3f}). " + ". ".join(explanation_parts[:3])
        else:
            explanation_text = f"No anomaly detected (score: {anomaly_score:.3f}). All metrics within normal ranges."
        
        return ExplanationResult(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            explanation_text=explanation_text,
            feature_scores=feature_scores,
            confidence_score=anomaly_score
        )
    
    async def explain_forecast_decision(
        self,
        model_name: str,
        model_version: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> ExplanationResult:
        """Generate explanation for forecasting decision"""
        
        historical_data = input_data.get('historical_data', [])
        forecast_value = prediction.get('forecast', 0.0)
        confidence_interval = prediction.get('confidence_interval', {})
        
        # Analyze trends and patterns
        if len(historical_data) >= 2:
            recent_trend = historical_data[-1] - historical_data[-2] if len(historical_data) >= 2 else 0
            overall_trend = (historical_data[-1] - historical_data[0]) / len(historical_data) if len(historical_data) > 1 else 0
        else:
            recent_trend = 0
            overall_trend = 0
        
        # Feature importance based on time series components
        feature_scores = {
            'recent_trend': abs(recent_trend) * 0.4,
            'overall_trend': abs(overall_trend) * 0.3,
            'volatility': np.std(historical_data) * 0.2 if len(historical_data) > 1 else 0,
            'seasonality': 0.1  # Simplified seasonality component
        }
        
        # Generate explanation
        explanation_parts = []
        
        if recent_trend > 0:
            explanation_parts.append(f"Recent upward trend (+{recent_trend:.2f})")
        elif recent_trend < 0:
            explanation_parts.append(f"Recent downward trend ({recent_trend:.2f})")
        
        if overall_trend > 0:
            explanation_parts.append(f"Long-term growth pattern (+{overall_trend:.2f})")
        elif overall_trend < 0:
            explanation_parts.append(f"Long-term decline pattern ({overall_trend:.2f})")
        
        lower_bound = confidence_interval.get('lower', forecast_value * 0.9)
        upper_bound = confidence_interval.get('upper', forecast_value * 1.1)
        
        explanation_text = f"Forecast: {forecast_value:.2f} (range: {lower_bound:.2f} - {upper_bound:.2f}). " + ". ".join(explanation_parts)
        
        return ExplanationResult(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            explanation_text=explanation_text,
            feature_scores=feature_scores,
            confidence_score=prediction.get('confidence', 0.5)
        )
    
    async def explain_recommendation_decision(
        self,
        model_name: str,
        model_version: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> ExplanationResult:
        """Generate explanation for investment recommendation decision"""
        
        # Extract input factors
        sentiment_score = input_data.get('sentiment_score', 0.0)
        anomaly_score = input_data.get('anomaly_score', 0.0)
        forecast_return = input_data.get('forecast_return', 0.0)
        risk_score = input_data.get('risk_score', 0.5)
        
        recommendation = prediction.get('recommendation', 'HOLD')
        confidence = prediction.get('confidence', 0.5)
        
        # Calculate feature importance
        feature_scores = {
            'sentiment_analysis': abs(sentiment_score) * 0.25,
            'forecast_return': abs(forecast_return) * 0.35,
            'risk_assessment': risk_score * 0.20,
            'anomaly_detection': anomaly_score * 0.15,
            'market_conditions': 0.05  # Simplified market factor
        }
        
        # Generate decision factors
        decision_factors = []
        
        if sentiment_score > 0.2:
            decision_factors.append("Positive market sentiment")
        elif sentiment_score < -0.2:
            decision_factors.append("Negative market sentiment")
        
        if forecast_return > 0.1:
            decision_factors.append("Strong expected returns")
        elif forecast_return < -0.05:
            decision_factors.append("Weak expected returns")
        
        if risk_score > 0.7:
            decision_factors.append("High risk profile")
        elif risk_score < 0.3:
            decision_factors.append("Low risk profile")
        
        if anomaly_score > 0.5:
            decision_factors.append("Unusual market patterns detected")
        
        # Generate explanation
        explanation_text = f"Recommendation: {recommendation} (confidence: {confidence:.2f}). "
        explanation_text += f"Key factors: {', '.join(decision_factors[:3])}"
        
        return ExplanationResult(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            explanation_text=explanation_text,
            feature_scores=feature_scores,
            confidence_score=confidence,
            decision_boundary={
                'buy_threshold': 0.6,
                'sell_threshold': 0.4,
                'current_score': confidence
            }
        )
    
    async def get_model_performance_history(
        self,
        model_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get model performance history"""
        
        conditions = ["model_name = $1"]
        params = [model_name]
        
        if start_date:
            conditions.append(f"evaluation_date >= ${len(params) + 1}")
            params.append(start_date)
        
        if end_date:
            conditions.append(f"evaluation_date <= ${len(params) + 1}")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions)
        
        async with get_database_connection() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM model_performance
                WHERE {where_clause}
                ORDER BY evaluation_date DESC
            """, *params)
            
            return [dict(row) for row in rows]
    
    async def track_model_performance(
        self,
        model_name: str,
        model_version: str,
        dataset_name: str,
        metrics: Dict[str, Any],
        confusion_matrix: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        bias_metrics: Optional[Dict[str, float]] = None,
        drift_score: Optional[float] = None
    ) -> UUID:
        """Track model performance metrics"""
        
        async with get_database_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO model_performance (
                    model_name, model_version, dataset_name, metrics,
                    confusion_matrix, feature_importance, bias_metrics, drift_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """,
                model_name, model_version, dataset_name, json.dumps(metrics),
                json.dumps(confusion_matrix) if confusion_matrix else None,
                json.dumps(feature_importance) if feature_importance else None,
                json.dumps(bias_metrics) if bias_metrics else None,
                drift_score
            )
            
            return result['id']


# Global explainability service instance
explainability_service = ExplainabilityService()