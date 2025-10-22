"""
Forecasting data models
"""

from pydantic import Field, validator
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin
from pydantic import ConfigDict


class ForecastHorizon(str, Enum):
    """Standard forecast horizons"""
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    SIX_MONTHS = "6M"
    TWELVE_MONTHS = "12M"
    TWO_YEARS = "2Y"


class ForecastType(str, Enum):
    """Types of forecasts"""
    STOCK_PRICE = "stock_price"
    REVENUE = "revenue"
    EARNINGS = "earnings"
    CASH_FLOW = "cash_flow"
    MARKET_CAP = "market_cap"
    VOLATILITY = "volatility"


class ModelType(str, Enum):
    """Forecasting model types"""
    PROPHET = "prophet"
    ARIMA = "arima"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"


class TimeSeriesData(BaseModel):
    """Time series data structure"""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamps: List[datetime]
    values: List[float]
    metric_name: str
    company: Optional[str] = None
    frequency: str = Field(..., description="Data frequency (daily, weekly, monthly)")
    
    @validator('timestamps', 'values')
    def validate_equal_length(cls, v, values):
        if 'timestamps' in values and len(v) != len(values['timestamps']):
            raise ValueError("Timestamps and values must have equal length")
        return v


class ForecastPoint(BaseModel):
    """Single forecast data point"""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    model_contribution: Optional[Dict[str, float]] = Field(None, description="Individual model contributions")


class StockForecast(UUIDMixin, TimestampMixin):
    """Stock price forecast"""
    model_config = ConfigDict(protected_namespaces=())
    
    ticker: str = Field(..., description="Stock ticker symbol")
    forecast_type: ForecastType = Field(default=ForecastType.STOCK_PRICE)
    forecasts: Dict[str, ForecastPoint] = Field(..., description="Horizon -> forecast point")
    model_used: ModelType
    model_version: str = Field(..., description="Version of the forecasting model")
    
    # Performance metrics
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    model_performance: Dict[str, float] = Field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Data sources
    data_sources: List[str] = Field(default_factory=list)
    training_period: Optional[str] = None
    last_training_date: Optional[datetime] = None
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return v.upper().strip()


class MetricForecast(UUIDMixin, TimestampMixin):
    """Financial metric forecast"""
    company: str = Field(..., description="Company identifier")
    metric_name: str = Field(..., description="Name of the forecasted metric")
    forecast_type: ForecastType
    forecasts: List[ForecastPoint] = Field(..., description="Forecast points over time")
    model_used: ModelType
    
    # Forecast metadata
    base_scenario: str = Field(default="base", description="Forecast scenario")
    assumptions: List[str] = Field(default_factory=list)
    external_factors: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    forecast_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainty_level: float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)


class ConfidenceMetrics(BaseModel):
    """Confidence and uncertainty metrics for forecasts"""
    model_config = ConfigDict(protected_namespaces=())
    
    forecast_id: UUID
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    horizon_confidence: Dict[str, float] = Field(default_factory=dict)
    uncertainty_sources: Dict[str, float] = Field(default_factory=dict)
    
    # Statistical measures
    prediction_intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    variance_explained: Optional[float] = Field(None, ge=0.0, le=1.0)
    residual_analysis: Optional[Dict[str, Any]] = None
    
    # Model reliability
    model_stability: float = Field(..., ge=0.0, le=1.0)
    out_of_sample_performance: Optional[float] = Field(None, ge=0.0, le=1.0)
    cross_validation_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ForecastEnsemble(BaseModel):
    """Ensemble forecast combining multiple models"""
    model_config = ConfigDict(protected_namespaces=())
    
    individual_forecasts: List[UUID] = Field(..., description="IDs of individual forecasts")
    ensemble_weights: Dict[str, float] = Field(..., description="Model -> weight")
    ensemble_forecast: ForecastPoint
    ensemble_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Ensemble metrics
    model_agreement: float = Field(..., ge=0.0, le=1.0, description="Agreement between models")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Diversity of ensemble models")
    ensemble_performance: Optional[Dict[str, float]] = None


class ForecastPerformance(UUIDMixin, TimestampMixin):
    """Historical performance tracking for forecasts"""
    forecast_id: UUID
    actual_values: List[float] = Field(..., description="Actual observed values")
    predicted_values: List[float] = Field(..., description="Predicted values")
    timestamps: List[datetime] = Field(..., description="Timestamps for values")
    
    # Performance metrics
    mae: float = Field(..., ge=0.0, description="Mean Absolute Error")
    rmse: float = Field(..., ge=0.0, description="Root Mean Square Error")
    mape: float = Field(..., ge=0.0, description="Mean Absolute Percentage Error")
    directional_accuracy: float = Field(..., ge=0.0, le=1.0)
    
    # Horizon-specific performance
    horizon_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    performance_trend: Optional[str] = Field(None, description="Improving/degrading/stable")


class ForecastAlert(UUIDMixin, TimestampMixin):
    """Alert for significant forecast changes or performance issues"""
    forecast_id: UUID
    alert_type: str = Field(..., description="Type of forecast alert")
    severity: str = Field(..., description="Alert severity level")
    message: str
    
    # Alert details
    current_forecast: float
    previous_forecast: Optional[float] = None
    change_magnitude: Optional[float] = None
    confidence_change: Optional[float] = None
    
    # Resolution
    is_acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class ExternalDataSource(BaseModel):
    """External data source configuration"""
    model_config = ConfigDict(protected_namespaces=())
    
    source_name: str = Field(..., description="Name of the data source")
    api_endpoint: str
    api_key_required: bool = Field(default=True)
    rate_limit: Optional[int] = Field(None, description="Requests per minute")
    data_types: List[str] = Field(default_factory=list)
    last_updated: Optional[datetime] = None
    is_active: bool = Field(default=True)
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)