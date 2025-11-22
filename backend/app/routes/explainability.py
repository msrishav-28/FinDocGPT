"""
Model explainability and decision tracking API routes
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.explainability_service import (
    explainability_service, ModelType, ExplanationType, ModelDecision, ExplanationResult
)
from ..dependencies.auth import get_current_user, require_permission
from ..models.auth import UserPermission

router = APIRouter(prefix="/explainability", tags=["explainability"])


class ModelRegistrationRequest(BaseModel):
    """Request to register a new model version"""
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Version string")
    model_type: ModelType = Field(..., description="Type of model")
    model_config: Dict[str, Any] = Field(..., description="Model configuration")
    training_data_hash: Optional[str] = Field(None, description="Hash of training data")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Initial performance metrics")


class ModelVersionResponse(BaseModel):
    """Response for model version information"""
    id: UUID
    model_name: str
    version: str
    model_type: str
    model_config: Dict[str, Any]
    training_data_hash: Optional[str]
    performance_metrics: Optional[Dict[str, Any]]
    deployment_date: datetime
    is_active: bool
    created_by: Optional[UUID]


class ExplanationRequest(BaseModel):
    """Request for model explanation"""
    model_name: str
    model_version: str
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    explanation_type: Optional[ExplanationType] = ExplanationType.FEATURE_IMPORTANCE


class ExplanationResponse(BaseModel):
    """Response containing model explanation"""
    explanation_type: str
    explanation_text: str
    feature_scores: Dict[str, float]
    confidence_score: float
    decision_boundary: Optional[Dict[str, Any]] = None
    counterfactual_examples: Optional[List[Dict[str, Any]]] = None


class ModelDecisionRequest(BaseModel):
    """Request to log a model decision"""
    model_name: str
    model_version: str
    model_type: ModelType
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence_score: Optional[float] = None
    processing_time_ms: Optional[int] = None
    request_id: Optional[str] = None


class ModelPerformanceRequest(BaseModel):
    """Request to track model performance"""
    model_name: str
    model_version: str
    dataset_name: str
    metrics: Dict[str, Any]
    confusion_matrix: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    bias_metrics: Optional[Dict[str, float]] = None
    drift_score: Optional[float] = None


class ModelPerformanceResponse(BaseModel):
    """Response for model performance data"""
    id: UUID
    model_name: str
    model_version: str
    evaluation_date: datetime
    dataset_name: str
    metrics: Dict[str, Any]
    confusion_matrix: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    bias_metrics: Optional[Dict[str, float]]
    drift_score: Optional[float]


@router.post("/models/register", response_model=Dict[str, str])
async def register_model(
    request: ModelRegistrationRequest,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Register a new model version"""
    
    try:
        model_id = await explainability_service.register_model(
            model_name=request.model_name,
            version=request.version,
            model_type=request.model_type,
            model_config=request.model_config,
            training_data_hash=request.training_data_hash,
            performance_metrics=request.performance_metrics,
            created_by=current_user.id
        )
        
        return {
            "message": f"Model {request.model_name} v{request.version} registered successfully",
            "model_id": str(model_id)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")


@router.post("/decisions/log", response_model=Dict[str, str])
async def log_model_decision(
    request: ModelDecisionRequest,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.GENERATE_FORECASTS))
):
    """Log a model decision for audit and explainability"""
    
    try:
        decision = ModelDecision(
            model_name=request.model_name,
            model_version=request.model_version,
            model_type=request.model_type,
            input_data=request.input_data,
            prediction=request.prediction,
            confidence_score=request.confidence_score,
            processing_time_ms=request.processing_time_ms,
            user_id=current_user.id,
            request_id=request.request_id
        )
        
        decision_id = await explainability_service.log_model_decision(decision)
        
        return {
            "message": "Model decision logged successfully",
            "decision_id": str(decision_id)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log model decision: {str(e)}")


@router.post("/explain/sentiment", response_model=ExplanationResponse)
async def explain_sentiment_decision(
    model_name: str,
    model_version: str,
    input_text: str,
    prediction: Dict[str, Any],
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Generate explanation for sentiment analysis decision"""
    
    try:
        explanation = await explainability_service.explain_sentiment_decision(
            model_name=model_name,
            model_version=model_version,
            input_text=input_text,
            prediction=prediction
        )
        
        return ExplanationResponse(
            explanation_type=explanation.explanation_type.value,
            explanation_text=explanation.explanation_text,
            feature_scores=explanation.feature_scores,
            confidence_score=explanation.confidence_score,
            decision_boundary=explanation.decision_boundary,
            counterfactual_examples=explanation.counterfactual_examples
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sentiment explanation: {str(e)}")


@router.post("/explain/anomaly", response_model=ExplanationResponse)
async def explain_anomaly_decision(
    model_name: str,
    model_version: str,
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Generate explanation for anomaly detection decision"""
    
    try:
        explanation = await explainability_service.explain_anomaly_decision(
            model_name=model_name,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction
        )
        
        return ExplanationResponse(
            explanation_type=explanation.explanation_type.value,
            explanation_text=explanation.explanation_text,
            feature_scores=explanation.feature_scores,
            confidence_score=explanation.confidence_score,
            decision_boundary=explanation.decision_boundary,
            counterfactual_examples=explanation.counterfactual_examples
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate anomaly explanation: {str(e)}")


@router.post("/explain/forecast", response_model=ExplanationResponse)
async def explain_forecast_decision(
    model_name: str,
    model_version: str,
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Generate explanation for forecasting decision"""
    
    try:
        explanation = await explainability_service.explain_forecast_decision(
            model_name=model_name,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction
        )
        
        return ExplanationResponse(
            explanation_type=explanation.explanation_type.value,
            explanation_text=explanation.explanation_text,
            feature_scores=explanation.feature_scores,
            confidence_score=explanation.confidence_score,
            decision_boundary=explanation.decision_boundary,
            counterfactual_examples=explanation.counterfactual_examples
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast explanation: {str(e)}")


@router.post("/explain/recommendation", response_model=ExplanationResponse)
async def explain_recommendation_decision(
    model_name: str,
    model_version: str,
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Generate explanation for investment recommendation decision"""
    
    try:
        explanation = await explainability_service.explain_recommendation_decision(
            model_name=model_name,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction
        )
        
        return ExplanationResponse(
            explanation_type=explanation.explanation_type.value,
            explanation_text=explanation.explanation_text,
            feature_scores=explanation.feature_scores,
            confidence_score=explanation.confidence_score,
            decision_boundary=explanation.decision_boundary,
            counterfactual_examples=explanation.counterfactual_examples
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation explanation: {str(e)}")


@router.post("/performance/track", response_model=Dict[str, str])
async def track_model_performance(
    request: ModelPerformanceRequest,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Track model performance metrics"""
    
    try:
        performance_id = await explainability_service.track_model_performance(
            model_name=request.model_name,
            model_version=request.model_version,
            dataset_name=request.dataset_name,
            metrics=request.metrics,
            confusion_matrix=request.confusion_matrix,
            feature_importance=request.feature_importance,
            bias_metrics=request.bias_metrics,
            drift_score=request.drift_score
        )
        
        return {
            "message": "Model performance tracked successfully",
            "performance_id": str(performance_id)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track model performance: {str(e)}")


@router.get("/performance/{model_name}", response_model=List[ModelPerformanceResponse])
async def get_model_performance_history(
    model_name: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Get model performance history"""
    
    try:
        performance_data = await explainability_service.get_model_performance_history(
            model_name=model_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return [ModelPerformanceResponse(**data) for data in performance_data]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model performance history: {str(e)}")


@router.get("/models/{model_name}/decisions")
async def get_model_decisions(
    model_name: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Get model decision history for analysis"""
    
    try:
        from ..services.audit_service import audit_service
        
        # Get model decision logs from audit service
        decisions = await audit_service.get_audit_logs(
            event_type=None,  # We'll filter by model decisions in the service
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        # Filter for model decision logs
        model_decisions = [
            decision for decision in decisions 
            if decision.get('resource_type') == 'model' and decision.get('resource_id') == model_name
        ]
        
        return {
            "model_name": model_name,
            "total_decisions": len(model_decisions),
            "decisions": model_decisions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model decisions: {str(e)}")


@router.get("/models/{model_name}/drift")
async def get_model_drift_analysis(
    model_name: str,
    days: int = Query(30, ge=1, le=365),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.VIEW_ANALYTICS))
):
    """Get model drift analysis over time"""
    
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        performance_data = await explainability_service.get_model_performance_history(
            model_name=model_name,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate drift metrics
        drift_scores = [p.get('drift_score') for p in performance_data if p.get('drift_score') is not None]
        
        if not drift_scores:
            return {
                "model_name": model_name,
                "period_days": days,
                "drift_analysis": {
                    "average_drift": 0.0,
                    "max_drift": 0.0,
                    "drift_trend": "stable",
                    "alert_level": "none"
                }
            }
        
        avg_drift = sum(drift_scores) / len(drift_scores)
        max_drift = max(drift_scores)
        
        # Determine trend (simplified)
        if len(drift_scores) >= 2:
            recent_drift = sum(drift_scores[-3:]) / min(3, len(drift_scores))
            older_drift = sum(drift_scores[:-3]) / max(1, len(drift_scores) - 3)
            trend = "increasing" if recent_drift > older_drift * 1.1 else "decreasing" if recent_drift < older_drift * 0.9 else "stable"
        else:
            trend = "stable"
        
        # Determine alert level
        if max_drift > 0.8:
            alert_level = "critical"
        elif max_drift > 0.6:
            alert_level = "high"
        elif max_drift > 0.4:
            alert_level = "medium"
        else:
            alert_level = "low"
        
        return {
            "model_name": model_name,
            "period_days": days,
            "drift_analysis": {
                "average_drift": avg_drift,
                "max_drift": max_drift,
                "drift_trend": trend,
                "alert_level": alert_level,
                "data_points": len(drift_scores)
            },
            "drift_history": [
                {"date": p["evaluation_date"], "drift_score": p.get("drift_score")}
                for p in performance_data if p.get("drift_score") is not None
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze model drift: {str(e)}")