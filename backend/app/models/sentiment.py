"""
Sentiment analysis data models
"""

from pydantic import Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin
from pydantic import ConfigDict


class SentimentPolarity(str, Enum):
    """Sentiment polarity categories"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class SentimentTopic(str, Enum):
    """Financial sentiment topics"""
    MANAGEMENT_OUTLOOK = "management_outlook"
    FINANCIAL_PERFORMANCE = "financial_performance"
    COMPETITIVE_POSITION = "competitive_position"
    MARKET_CONDITIONS = "market_conditions"
    REGULATORY_ENVIRONMENT = "regulatory_environment"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    GROWTH_PROSPECTS = "growth_prospects"
    RISK_FACTORS = "risk_factors"


class SentimentAnalysis(UUIDMixin, TimestampMixin):
    """Comprehensive sentiment analysis results"""
    model_config = ConfigDict(protected_namespaces=())
    
    document_id: Optional[UUID] = None
    text_snippet: Optional[str] = Field(None, description="Analyzed text snippet")
    overall_sentiment: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    polarity: SentimentPolarity
    topic_sentiments: Dict[str, float] = Field(default_factory=dict)
    sentiment_explanation: Optional[str] = None
    model_used: str = Field(..., description="Model used for analysis")
    processing_time: Optional[float] = None
    
    @validator('overall_sentiment')
    def validate_sentiment_range(cls, v):
        if not -1.0 <= v <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        return v
    
    @validator('topic_sentiments')
    def validate_topic_sentiments(cls, v):
        for topic, score in v.items():
            if not -1.0 <= score <= 1.0:
                raise ValueError(f"Topic sentiment score for {topic} must be between -1.0 and 1.0")
        return v


class TopicSentiment(BaseModel):
    """Sentiment analysis for specific topics"""
    model_config = ConfigDict(protected_namespaces=())
    
    topic: SentimentTopic
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_phrases: List[str] = Field(default_factory=list)
    context: Optional[str] = None


class SentimentTrend(BaseModel):
    """Sentiment trend data point"""
    model_config = ConfigDict(protected_namespaces=())
    
    date: datetime
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    document_count: int = Field(..., ge=0)
    topic: Optional[SentimentTopic] = None


class SentimentTrends(BaseModel):
    """Historical sentiment trends"""
    model_config = ConfigDict(protected_namespaces=())
    
    company: str
    topic: Optional[SentimentTopic] = None
    time_period: str = Field(..., description="Time period (e.g., '1Y', '6M', '3M')")
    trends: List[SentimentTrend]
    average_sentiment: float = Field(..., ge=-1.0, le=1.0)
    volatility: float = Field(..., ge=0.0, description="Sentiment volatility measure")
    trend_direction: str = Field(..., description="Overall trend direction")


class SentimentComparison(BaseModel):
    """Sentiment comparison between companies"""
    model_config = ConfigDict(protected_namespaces=())
    
    companies: List[str]
    comparison_date: datetime
    topic: Optional[SentimentTopic] = None
    sentiment_scores: Dict[str, float] = Field(..., description="Company -> sentiment score")
    confidence_scores: Dict[str, float] = Field(..., description="Company -> confidence score")
    relative_rankings: Dict[str, int] = Field(..., description="Company -> rank (1 = most positive)")
    analysis_summary: Optional[str] = None


class SentimentAlert(UUIDMixin, TimestampMixin):
    """Sentiment-based alert"""
    company: str
    alert_type: str = Field(..., description="Type of sentiment alert")
    current_sentiment: float = Field(..., ge=-1.0, le=1.0)
    previous_sentiment: float = Field(..., ge=-1.0, le=1.0)
    change_magnitude: float = Field(..., description="Magnitude of sentiment change")
    significance_level: float = Field(..., ge=0.0, le=1.0)
    description: str
    is_resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = None


class SentimentModelPerformance(BaseModel):
    """Performance metrics for sentiment models"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    last_evaluated: datetime
    evaluation_dataset_size: int = Field(..., ge=0)
    confidence_calibration: Optional[float] = Field(None, ge=0.0, le=1.0)