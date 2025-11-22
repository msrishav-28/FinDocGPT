"""
Anomaly detection data models
"""

from pydantic import Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin


class AnomalySeverity(str, Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Types of anomalies"""
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_DEVIATION = "pattern_deviation"
    TREND_BREAK = "trend_break"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_SPIKE = "volatility_spike"


class AnomalyStatus(str, Enum):
    """Anomaly resolution status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    EXPLAINED = "explained"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class Anomaly(UUIDMixin, TimestampMixin):
    """Detected anomaly in financial data"""
    company: str = Field(..., description="Company identifier")
    metric_name: str = Field(..., description="Name of the metric with anomaly")
    current_value: float = Field(..., description="Current anomalous value")
    expected_value: float = Field(..., description="Expected value based on baseline")
    deviation_score: float = Field(..., ge=0.0, description="Magnitude of deviation")
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    status: AnomalyStatus = Field(default=AnomalyStatus.DETECTED)
    
    # Context and explanation
    explanation: str = Field(..., description="Human-readable explanation of the anomaly")
    historical_context: Optional[str] = Field(None, description="Historical context for the anomaly")
    potential_causes: List[str] = Field(default_factory=list)
    
    # Detection details
    detection_method: str = Field(..., description="Method used to detect the anomaly")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    baseline_period: Optional[str] = Field(None, description="Period used for baseline calculation")
    
    # Resolution tracking
    investigated_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    @validator('deviation_score')
    def validate_deviation_score(cls, v):
        if v < 0:
            raise ValueError("Deviation score must be non-negative")
        return v


class PatternAnomalies(BaseModel):
    """Collection of pattern-based anomalies"""
    company: str
    analysis_period: str = Field(..., description="Period analyzed for patterns")
    detected_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    pattern_confidence: float = Field(..., ge=0.0, le=1.0)
    anomalous_patterns: List[Anomaly] = Field(default_factory=list)
    pattern_explanation: Optional[str] = None


class RiskAssessment(BaseModel):
    """Risk assessment for detected anomalies"""
    anomaly_id: UUID
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    financial_impact: Optional[str] = Field(None, description="Potential financial impact")
    probability_of_impact: float = Field(..., ge=0.0, le=1.0)
    time_horizon: str = Field(..., description="Time horizon for potential impact")
    
    # Risk factors
    market_risk: float = Field(..., ge=0.0, le=1.0)
    operational_risk: float = Field(..., ge=0.0, le=1.0)
    regulatory_risk: float = Field(..., ge=0.0, le=1.0)
    reputational_risk: float = Field(..., ge=0.0, le=1.0)
    
    # Mitigation
    recommended_actions: List[str] = Field(default_factory=list)
    monitoring_recommendations: List[str] = Field(default_factory=list)
    escalation_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class AnomalyHistory(BaseModel):
    """Historical anomaly data for a company"""
    company: str
    time_period: str = Field(..., description="Historical period covered")
    total_anomalies: int = Field(..., ge=0)
    anomalies_by_severity: Dict[AnomalySeverity, int] = Field(default_factory=dict)
    anomalies_by_type: Dict[AnomalyType, int] = Field(default_factory=dict)
    resolution_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of resolved anomalies")
    average_resolution_time: Optional[float] = Field(None, description="Average time to resolution in hours")
    recurring_patterns: List[str] = Field(default_factory=list)


class AnomalyCorrelation(BaseModel):
    """Correlation analysis between anomalies"""
    primary_anomaly_id: UUID
    correlated_anomaly_ids: List[UUID]
    correlation_strength: float = Field(..., ge=-1.0, le=1.0)
    correlation_type: str = Field(..., description="Type of correlation")
    time_lag: Optional[float] = Field(None, description="Time lag between anomalies in hours")
    statistical_significance: float = Field(..., ge=0.0, le=1.0)


class AnomalyBaseline(BaseModel):
    """Baseline model for anomaly detection"""
    company: str
    metric_name: str
    baseline_type: str = Field(..., description="Type of baseline model")
    baseline_parameters: Dict[str, Any] = Field(default_factory=dict)
    baseline_period: str = Field(..., description="Period used for baseline")
    last_updated: datetime
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    seasonal_adjustments: Optional[Dict[str, Any]] = None


class AnomalyAlert(UUIDMixin, TimestampMixin):
    """Alert generated for anomaly detection"""
    anomaly_id: UUID
    alert_level: AnomalySeverity
    recipient_groups: List[str] = Field(default_factory=list)
    message: str
    delivery_channels: List[str] = Field(default_factory=list)
    is_sent: bool = Field(default=False)
    sent_at: Optional[datetime] = None
    acknowledgment_required: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None