"""
Investment recommendation data models
"""

from pydantic import Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin
from pydantic import ConfigDict


class InvestmentSignal(str, Enum):
    """Investment recommendation signals"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(str, Enum):
    """Risk level categories"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeHorizon(str, Enum):
    """Investment time horizons"""
    SHORT_TERM = "short_term"  # < 3 months
    MEDIUM_TERM = "medium_term"  # 3-12 months
    LONG_TERM = "long_term"  # > 12 months


class RecommendationStatus(str, Enum):
    """Status of investment recommendations"""
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"


class InvestmentRecommendation(UUIDMixin, TimestampMixin):
    """Complete investment recommendation"""
    model_config = ConfigDict(protected_namespaces=())
    
    ticker: str = Field(..., description="Stock ticker symbol")
    signal: InvestmentSignal
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recommendation confidence")
    
    # Price and targets
    current_price: float = Field(..., gt=0, description="Current stock price")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    
    # Risk and sizing
    risk_level: RiskLevel
    position_size: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recommended position size as % of portfolio")
    max_position_size: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Reasoning and context
    reasoning: str = Field(..., description="Main reasoning for recommendation")
    supporting_factors: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    time_horizon: TimeHorizon
    
    # Analysis inputs
    document_insights: Optional[Dict[str, Any]] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    anomaly_flags: List[str] = Field(default_factory=list)
    forecast_data: Optional[Dict[str, Any]] = None
    
    # Metadata
    analyst_id: Optional[str] = None
    model_version: str = Field(..., description="Version of recommendation model")
    status: RecommendationStatus = Field(default=RecommendationStatus.ACTIVE)
    expires_at: Optional[datetime] = None
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()
    
    @validator('target_price', 'stop_loss')
    def validate_prices(cls, v, values):
        if v is not None and 'current_price' in values and v <= 0:
            raise ValueError("Prices must be positive")
        return v


class RecommendationExplanation(BaseModel):
    """Detailed explanation of investment recommendation"""
    model_config = ConfigDict(protected_namespaces=())
    
    recommendation_id: UUID
    executive_summary: str
    
    # Factor analysis
    fundamental_analysis: Optional[str] = None
    technical_analysis: Optional[str] = None
    sentiment_analysis: Optional[str] = None
    risk_analysis: str
    
    # Supporting data
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    peer_comparison: Optional[Dict[str, Any]] = None
    historical_performance: Optional[Dict[str, Any]] = None
    
    # Model insights
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    model_confidence_factors: List[str] = Field(default_factory=list)
    alternative_scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Disclaimers and limitations
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    data_quality_notes: Optional[str] = None


class Portfolio(BaseModel):
    """Investment portfolio model"""
    portfolio_id: UUID = Field(default_factory=uuid4)
    name: str
    owner_id: str
    
    # Holdings
    holdings: Dict[str, float] = Field(default_factory=dict, description="Ticker -> position size")
    cash_position: float = Field(default=0.0, ge=0.0)
    total_value: float = Field(..., gt=0)
    
    # Portfolio metrics
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = Field(None, ge=0.0)
    max_drawdown: Optional[float] = None
    
    # Risk management
    risk_tolerance: RiskLevel
    max_position_size: float = Field(default=0.1, ge=0.0, le=1.0)
    sector_limits: Dict[str, float] = Field(default_factory=dict)
    
    # Tracking
    benchmark: Optional[str] = Field(None, description="Benchmark index")
    inception_date: datetime
    last_rebalanced: Optional[datetime] = None


class PositionSizes(BaseModel):
    """Optimized position sizing recommendations"""
    portfolio_id: UUID
    recommendations: Dict[UUID, float] = Field(..., description="Recommendation ID -> position size")
    
    # Optimization details
    optimization_method: str = Field(..., description="Method used for optimization")
    risk_budget: float = Field(..., ge=0.0, le=1.0)
    expected_return: float
    expected_risk: float = Field(..., ge=0.0)
    
    # Constraints applied
    position_limits: Dict[str, float] = Field(default_factory=dict)
    sector_constraints: Dict[str, float] = Field(default_factory=dict)
    liquidity_constraints: Optional[Dict[str, Any]] = None
    
    # Performance projections
    projected_return: float
    projected_risk: float = Field(..., ge=0.0)
    diversification_ratio: float = Field(..., ge=0.0)


class AnalysisContext(BaseModel):
    """Context for generating recommendations"""
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    market_conditions: Optional[str] = None
    economic_indicators: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis scope
    include_sentiment: bool = Field(default=True)
    include_anomalies: bool = Field(default=True)
    include_forecasts: bool = Field(default=True)
    include_peer_analysis: bool = Field(default=False)
    
    # Risk parameters
    risk_free_rate: float = Field(default=0.02, ge=0.0)
    market_volatility: Optional[float] = Field(None, ge=0.0)
    correlation_lookback: int = Field(default=252, gt=0, description="Days for correlation calculation")
    
    # Model parameters
    model_weights: Dict[str, float] = Field(default_factory=dict)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_data_points: int = Field(default=30, gt=0)


class RecommendationPerformance(UUIDMixin, TimestampMixin):
    """Performance tracking for recommendations"""
    recommendation_id: UUID
    
    # Performance metrics
    actual_return: Optional[float] = None
    predicted_return: Optional[float] = None
    holding_period: Optional[int] = Field(None, description="Days held")
    
    # Risk metrics
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = Field(None, ge=0.0)
    sharpe_ratio: Optional[float] = None
    
    # Outcome tracking
    target_hit: Optional[bool] = None
    stop_loss_hit: Optional[bool] = None
    exit_reason: Optional[str] = None
    exit_date: Optional[datetime] = None
    
    # Attribution
    performance_attribution: Dict[str, float] = Field(default_factory=dict)
    factor_returns: Dict[str, float] = Field(default_factory=dict)


class RecommendationAlert(UUIDMixin, TimestampMixin):
    """Alert for recommendation updates or performance"""
    recommendation_id: UUID
    alert_type: str = Field(..., description="Type of recommendation alert")
    severity: str = Field(..., description="Alert severity")
    message: str
    
    # Alert triggers
    price_change: Optional[float] = None
    confidence_change: Optional[float] = None
    new_information: Optional[str] = None
    
    # Delivery
    recipient_ids: List[str] = Field(default_factory=list)
    delivery_channels: List[str] = Field(default_factory=list)
    is_delivered: bool = Field(default=False)
    delivered_at: Optional[datetime] = None