"""
Anomaly Detection API Routes

This module provides REST API endpoints for the Statistical Anomaly Detection Service.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from ..models.anomaly import (
    Anomaly, AnomalySeverity, AnomalyType, AnomalyStatus,
    PatternAnomalies, RiskAssessment, AnomalyHistory,
    AnomalyCorrelation
)
from ..services.anomaly_service import (
    detect_metric_anomalies, analyze_pattern_anomalies, analyze_correlation_anomalies,
    get_anomaly_history, assess_anomaly_risk, manage_anomaly_lifecycle,
    update_anomaly_resolution, get_anomaly_dashboard_data, store_anomaly,
    classify_anomaly_severity, generate_anomaly_explanation
)

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


class MetricAnomalyRequest(BaseModel):
    """Request model for metric anomaly detection"""
    company: str
    metrics: List[str]
    lookback_periods: int = 20
    detection_methods: Optional[List[str]] = None


class PatternAnomalyRequest(BaseModel):
    """Request model for pattern anomaly analysis"""
    company: str
    metrics: Optional[List[str]] = None
    lookback_periods: int = 20


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment"""
    anomaly_id: str
    company_context: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None


class AnomalyUpdateRequest(BaseModel):
    """Request model for anomaly status updates"""
    status: AnomalyStatus
    notes: Optional[str] = None
    resolved_by: Optional[str] = None


@router.post("/detect/metrics", response_model=List[Anomaly])
async def detect_metric_anomalies_endpoint(request: MetricAnomalyRequest):
    """
    Detect anomalies in financial metrics using statistical methods
    
    - **company**: Company identifier (ticker or name)
    - **metrics**: List of metric names to analyze
    - **lookback_periods**: Number of historical periods to consider (default: 20)
    - **detection_methods**: Optional list of methods ['zscore', 'iqr', 'isolation_forest']
    """
    try:
        if not request.metrics:
            raise HTTPException(status_code=400, detail="At least one metric must be specified")
        
        anomalies = await detect_metric_anomalies(
            company=request.company,
            metrics=request.metrics,
            lookback_periods=request.lookback_periods
        )
        
        # Store detected anomalies
        for anomaly in anomalies:
            try:
                await store_anomaly(anomaly)
            except Exception as e:
                # Log but don't fail the request
                print(f"Failed to store anomaly {anomaly.id}: {e}")
        
        return anomalies
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metric anomaly detection failed: {str(e)}")


@router.post("/detect/patterns", response_model=PatternAnomalies)
async def detect_pattern_anomalies_endpoint(request: PatternAnomalyRequest):
    """
    Detect complex pattern anomalies using machine learning
    
    - **company**: Company identifier
    - **metrics**: Optional list of metrics (uses default set if not provided)
    - **lookback_periods**: Number of historical periods to analyze
    """
    try:
        pattern_anomalies = await analyze_pattern_anomalies(
            company=request.company,
            metrics=request.metrics,
            lookback_periods=request.lookback_periods
        )
        
        # Store detected pattern anomalies
        for anomaly in pattern_anomalies.anomalous_patterns:
            try:
                await store_anomaly(anomaly)
            except Exception as e:
                print(f"Failed to store pattern anomaly {anomaly.id}: {e}")
        
        return pattern_anomalies
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern anomaly detection failed: {str(e)}")


@router.get("/correlations/{company}", response_model=List[AnomalyCorrelation])
async def analyze_correlation_anomalies_endpoint(
    company: str,
    metrics: List[str] = Query(default=["revenue", "profit_margin", "cash_flow", "debt_ratio"]),
    correlation_threshold: float = Query(0.7, ge=0.1, le=1.0)
):
    """
    Analyze correlations between metrics to detect systemic anomalies
    
    - **company**: Company identifier
    - **metrics**: List of metrics to analyze for correlations
    - **correlation_threshold**: Minimum correlation strength to report (0.1-1.0)
    """
    try:
        correlations = await analyze_correlation_anomalies(
            company=company,
            metrics=metrics,
            correlation_threshold=correlation_threshold
        )
        
        return correlations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")


@router.post("/assess-risk", response_model=RiskAssessment)
async def assess_anomaly_risk_endpoint(request: RiskAssessmentRequest):
    """
    Assess risk for a detected anomaly
    
    - **anomaly_id**: ID of the anomaly to assess
    - **company_context**: Optional company context (size, industry, etc.)
    - **market_context**: Optional market conditions and external factors
    """
    try:
        # First, retrieve the anomaly from database
        from ..database.connection import get_database
        db = await get_database()
        
        query = """
            SELECT * FROM anomalies WHERE id = $1
        """
        anomaly_row = await db.fetchrow(query, request.anomaly_id)
        
        if not anomaly_row:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        # Convert to Anomaly object (simplified - in real implementation would use proper deserialization)
        anomaly = Anomaly(
            id=anomaly_row['id'],
            company=anomaly_row['company'],
            metric_name=anomaly_row['metric_name'],
            current_value=anomaly_row['current_value'],
            expected_value=anomaly_row['expected_value'],
            deviation_score=anomaly_row['deviation_score'],
            severity=AnomalySeverity(anomaly_row['severity']),
            anomaly_type=AnomalyType(anomaly_row['anomaly_type']),
            status=AnomalyStatus(anomaly_row['status']),
            explanation=anomaly_row['explanation'],
            detection_method=anomaly_row['detection_method'],
            confidence=anomaly_row['confidence'],
            created_at=anomaly_row['created_at']
        )
        
        # Assess risk
        risk_assessment = await assess_anomaly_risk(
            anomaly=anomaly,
            company_context=request.company_context,
            market_context=request.market_context
        )
        
        return risk_assessment
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.get("/history/{company}", response_model=AnomalyHistory)
async def get_anomaly_history_endpoint(
    company: str,
    days_back: int = Query(90, ge=7, le=365)
):
    """
    Get historical anomaly data for a company
    
    - **company**: Company identifier
    - **days_back**: Number of days to look back (7-365)
    """
    try:
        return await get_anomaly_history(company, days_back)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomaly history: {str(e)}")


@router.put("/update/{anomaly_id}")
async def update_anomaly_status_endpoint(
    anomaly_id: str,
    request: AnomalyUpdateRequest
):
    """
    Update anomaly status and resolution information
    
    - **anomaly_id**: ID of the anomaly to update
    - **status**: New status for the anomaly
    - **notes**: Optional resolution notes
    - **resolved_by**: Optional identifier of who resolved the anomaly
    """
    try:
        success = await update_anomaly_resolution(
            anomaly_id=anomaly_id,
            status=request.status,
            notes=request.notes,
            resolved_by=request.resolved_by
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update anomaly status")
        
        return {"message": "Anomaly status updated successfully", "anomaly_id": anomaly_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update anomaly: {str(e)}")


@router.get("/dashboard")
async def get_anomaly_dashboard_endpoint(
    company: Optional[str] = Query(None)
):
    """
    Get comprehensive anomaly dashboard data
    
    - **company**: Optional company filter (if not provided, shows all companies)
    """
    try:
        dashboard_data = await get_anomaly_dashboard_data(company)
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/active")
async def get_active_anomalies_endpoint(
    company: Optional[str] = Query(None),
    severity: Optional[List[AnomalySeverity]] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get list of active anomalies
    
    - **company**: Optional company filter
    - **severity**: Optional severity filter
    - **limit**: Maximum number of anomalies to return (1-200)
    """
    try:
        from ..services.anomaly_service import get_anomaly_manager
        
        manager = await get_anomaly_manager()
        active_anomalies = await manager.get_active_anomalies(company, severity)
        
        # Limit results
        limited_anomalies = active_anomalies[:limit]
        
        return {
            "total_count": len(active_anomalies),
            "returned_count": len(limited_anomalies),
            "anomalies": limited_anomalies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active anomalies: {str(e)}")


@router.get("/explain/{anomaly_id}")
async def explain_anomaly_endpoint(
    anomaly_id: str,
    include_technical: bool = Query(False)
):
    """
    Generate detailed explanation for an anomaly
    
    - **anomaly_id**: ID of the anomaly to explain
    - **include_technical**: Whether to include technical detection details
    """
    try:
        # Retrieve anomaly from database
        from ..database.connection import get_database
        db = await get_database()
        
        query = """
            SELECT * FROM anomalies WHERE id = $1
        """
        anomaly_row = await db.fetchrow(query, anomaly_id)
        
        if not anomaly_row:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        # Convert to Anomaly object
        anomaly = Anomaly(
            id=anomaly_row['id'],
            company=anomaly_row['company'],
            metric_name=anomaly_row['metric_name'],
            current_value=anomaly_row['current_value'],
            expected_value=anomaly_row['expected_value'],
            deviation_score=anomaly_row['deviation_score'],
            severity=AnomalySeverity(anomaly_row['severity']),
            anomaly_type=AnomalyType(anomaly_row['anomaly_type']),
            status=AnomalyStatus(anomaly_row['status']),
            explanation=anomaly_row['explanation'],
            historical_context=anomaly_row.get('historical_context'),
            potential_causes=anomaly_row.get('potential_causes', []),
            detection_method=anomaly_row['detection_method'],
            confidence=anomaly_row['confidence'],
            created_at=anomaly_row['created_at']
        )
        
        # Generate explanation
        explanation = await generate_anomaly_explanation(anomaly, include_technical)
        
        return {
            "anomaly_id": anomaly_id,
            "explanation": explanation,
            "anomaly_summary": {
                "company": anomaly.company,
                "metric": anomaly.metric_name,
                "severity": anomaly.severity.value,
                "type": anomaly.anomaly_type.value,
                "confidence": anomaly.confidence,
                "detection_date": anomaly.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain anomaly: {str(e)}")


@router.get("/severity/classify")
async def classify_severity_endpoint(
    anomaly_score: float = Query(..., ge=0.0),
    confidence: float = Query(..., ge=0.0, le=1.0),
    is_record_extreme: bool = Query(False),
    volatility_factor: float = Query(1.0, ge=0.1),
    breaks_trend: bool = Query(False)
):
    """
    Classify anomaly severity based on various factors
    
    - **anomaly_score**: Anomaly magnitude score
    - **confidence**: Detection confidence (0.0-1.0)
    - **is_record_extreme**: Whether this is a record high/low
    - **volatility_factor**: Volatility context factor
    - **breaks_trend**: Whether this breaks a long-term trend
    """
    try:
        historical_context = {
            'is_record_extreme': is_record_extreme,
            'volatility_factor': volatility_factor,
            'breaks_long_term_trend': breaks_trend
        }
        
        severity = await classify_anomaly_severity(
            anomaly_score=anomaly_score,
            confidence=confidence,
            historical_context=historical_context
        )
        
        return {
            "severity": severity.value,
            "severity_score": anomaly_score * confidence,
            "factors": historical_context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Severity classification failed: {str(e)}")


@router.get("/types")
async def get_anomaly_types():
    """Get list of available anomaly types"""
    return {
        "anomaly_types": [atype.value for atype in AnomalyType],
        "severity_levels": [severity.value for severity in AnomalySeverity],
        "status_options": [status.value for status in AnomalyStatus]
    }


@router.get("/health")
async def anomaly_service_health():
    """Check anomaly detection service health"""
    try:
        from ..services.anomaly_service import get_anomaly_engine, get_pattern_detector, get_risk_engine
        
        # Check if services are available
        engine = await get_anomaly_engine()
        pattern_detector = await get_pattern_detector()
        risk_engine = await get_risk_engine()
        
        return {
            "status": "healthy",
            "services": {
                "anomaly_engine": "available",
                "pattern_detector": "available", 
                "risk_engine": "available"
            },
            "detection_methods": ["zscore", "iqr", "isolation_forest"],
            "pattern_methods": ["trend_break", "seasonal", "volatility", "correlation", "clustering"],
            "supported_metrics": ["revenue", "profit_margin", "cash_flow", "debt_ratio", "expenses"],
            "service": "Statistical Anomaly Detection"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "Statistical Anomaly Detection"
        }


# Utility endpoints
@router.post("/simulate")
async def simulate_anomaly_detection(
    company: str = "TEST_COMPANY",
    metric: str = "revenue",
    periods: int = 20
):
    """
    Simulate anomaly detection with generated data (for testing)
    
    - **company**: Company identifier for simulation
    - **metric**: Metric name to simulate
    - **periods**: Number of periods to generate
    """
    try:
        # This will use the sample data generation in the service
        anomalies = await detect_metric_anomalies(
            company=company,
            metrics=[metric],
            lookback_periods=periods
        )
        
        return {
            "simulation_info": {
                "company": company,
                "metric": metric,
                "periods_generated": periods,
                "anomalies_detected": len(anomalies)
            },
            "detected_anomalies": anomalies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")