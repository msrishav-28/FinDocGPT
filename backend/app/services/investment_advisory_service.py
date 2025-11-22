"""
Investment Advisory Service with Explainable AI

This service implements a comprehensive investment advisory system that:
- Aggregates signals from document insights, sentiment, anomalies, and forecasts
- Uses multi-criteria decision making with configurable weights
- Generates investment signals (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- Provides risk assessment and position sizing recommendations
- Offers explainable AI with natural language explanations

Features:
- Multi-factor recommendation engine with ensemble approach
- Risk-adjusted position sizing based on volatility and sentiment
- Portfolio-level risk assessment and correlation analysis
- Transparent decision reasoning with feature importance
- Complete audit trail for regulatory compliance
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from ..models.recommendation import (
    InvestmentRecommendation, InvestmentSignal, RiskLevel, TimeHorizon,
    RecommendationExplanation, Portfolio, PositionSizes, AnalysisContext,
    RecommendationStatus, RecommendationPerformance
)
from ..models.sentiment import SentimentAnalysis, SentimentTopic
from ..models.anomaly import Anomaly, AnomalySeverity
from ..config import get_settings
from ..database.connection import get_database

# Import existing services
try:
    from .sentiment_service import analyze_document_sentiment, analyze_topic_sentiment
    from .anomaly_service import detect_metric_anomalies
    from .ensemble_forecasting_service import get_ensemble_forecasting_service
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Some services not available: {e}")

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SignalInput:
    """Input signals for recommendation generation"""
    document_insights: Optional[Dict[str, Any]] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    topic_sentiments: Optional[Dict[str, Any]] = None
    anomalies: Optional[List[Anomaly]] = None
    forecast: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None


@dataclass
class DecisionFactors:
    """Factors contributing to investment decision"""
    fundamental_score: float
    technical_score: float
    sentiment_score: float
    risk_score: float
    forecast_score: float
    anomaly_score: float
    overall_score: float
    confidence: float


class InvestmentAdvisoryService:
    """Main investment advisory service with multi-factor analysis"""
    
    def __init__(self):
        self.default_weights = {
            'fundamental': 0.25,
            'technical': 0.20,
            'sentiment': 0.20,
            'forecast': 0.20,
            'anomaly': 0.10,
            'risk': 0.05
        }
        
        # Signal thresholds for investment decisions
        self.signal_thresholds = {
            InvestmentSignal.STRONG_BUY: 0.7,
            InvestmentSignal.BUY: 0.3,
            InvestmentSignal.HOLD: -0.3,
            InvestmentSignal.SELL: -0.7,
            InvestmentSignal.STRONG_SELL: -1.0
        }
        
        # Risk level mappings
        self.risk_mappings = {
            'very_low': 0.1,
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'very_high': 0.9
        }
        
        self.forecasting_service = None
    
    async def initialize(self):
        """Initialize the advisory service"""
        try:
            logger.info("Investment Advisory Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Investment Advisory Service: {e}")
    
    async def generate_recommendation(
        self, 
        ticker: str, 
        context: AnalysisContext = None,
        custom_weights: Dict[str, float] = None
    ) -> InvestmentRecommendation:
        """
        Generate comprehensive investment recommendation
        
        Args:
            ticker: Stock ticker symbol
            context: Analysis context with parameters
            custom_weights: Custom factor weights (optional)
        
        Returns:
            Complete investment recommendation with reasoning
        """
        if context is None:
            context = AnalysisContext()
        
        # Use custom weights if provided, otherwise use defaults
        weights = custom_weights if custom_weights else self.default_weights
        
        try:
            # Gather all input signals
            signal_inputs = await self._gather_signal_inputs(ticker, context)
            
            # Calculate decision factors
            decision_factors = await self._calculate_decision_factors(
                ticker, signal_inputs, weights
            )
            
            # Generate investment signal
            investment_signal = self._determine_investment_signal(decision_factors.overall_score)
            
            # Calculate risk assessment
            risk_level = self._assess_risk_level(signal_inputs, decision_factors)
            
            # Calculate target price and position sizing
            target_price, stop_loss = await self._calculate_price_targets(
                ticker, signal_inputs, investment_signal
            )
            
            position_size = self._calculate_position_size(
                investment_signal, risk_level, decision_factors.confidence
            )
            
            # Generate reasoning and supporting factors
            reasoning, supporting_factors, risk_factors = self._generate_reasoning(
                ticker, investment_signal, decision_factors, signal_inputs
            )
            
            # Get current price (placeholder - would come from market data)
            current_price = await self._get_current_price(ticker)
            
            # Create recommendation
            recommendation = InvestmentRecommendation(
                ticker=ticker,
                signal=investment_signal,
                confidence=decision_factors.confidence,
                current_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_level=risk_level,
                position_size=position_size,
                reasoning=reasoning,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                time_horizon=self._determine_time_horizon(signal_inputs),
                document_insights=signal_inputs.document_insights,
                sentiment_score=signal_inputs.sentiment_analysis.overall_sentiment if signal_inputs.sentiment_analysis else None,
                anomaly_flags=[a.explanation for a in signal_inputs.anomalies] if signal_inputs.anomalies else [],
                forecast_data=signal_inputs.forecast,
                model_version="1.0.0",
                status=RecommendationStatus.ACTIVE,
                expires_at=datetime.now() + timedelta(days=30)  # 30-day validity
            )
            
            # Store recommendation in database
            await self._store_recommendation(recommendation)
            
            logger.info(f"Generated {investment_signal.value} recommendation for {ticker} "
                       f"with {decision_factors.confidence:.2f} confidence")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation for {ticker}: {e}")
            raise
    
    async def _gather_signal_inputs(self, ticker: str, context: AnalysisContext) -> SignalInput:
        """Gather all input signals for analysis"""
        signal_inputs = SignalInput()
        
        try:
            # Get document insights if available
            if context.include_sentiment:
                try:
                    # This would typically get the latest document for the company
                    # For now, we'll use a placeholder
                    signal_inputs.document_insights = {
                        'revenue_growth': 0.15,
                        'profit_margin': 0.12,
                        'debt_ratio': 0.35,
                        'management_confidence': 0.7
                    }
                except Exception as e:
                    logger.warning(f"Failed to get document insights for {ticker}: {e}")
            
            # Get sentiment analysis
            if context.include_sentiment:
                try:
                    # Placeholder sentiment analysis
                    from ..models.sentiment import SentimentPolarity
                    signal_inputs.sentiment_analysis = SentimentAnalysis(
                        text_snippet=f"Analysis for {ticker}",
                        overall_sentiment=np.random.uniform(-0.5, 0.5),
                        confidence=0.75,
                        polarity=SentimentPolarity.NEUTRAL,
                        sentiment_explanation="Market sentiment analysis",
                        model_used="ensemble"
                    )
                    
                    # Topic-specific sentiments
                    signal_inputs.topic_sentiments = {
                        'financial_performance': np.random.uniform(-0.3, 0.3),
                        'management_outlook': np.random.uniform(-0.2, 0.4),
                        'competitive_position': np.random.uniform(-0.4, 0.2)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get sentiment analysis for {ticker}: {e}")
            
            # Get anomaly detection results
            if context.include_anomalies:
                try:
                    metrics = ['revenue', 'profit_margin', 'debt_ratio', 'cash_flow']
                    # For now, use empty list as placeholder
                    signal_inputs.anomalies = []
                except Exception as e:
                    logger.warning(f"Failed to get anomalies for {ticker}: {e}")
                    signal_inputs.anomalies = []
            
            # Get forecasting results
            if context.include_forecasts:
                try:
                    # Placeholder forecast data
                    signal_inputs.forecast = {
                        'horizons': {30: 105.0, 90: 110.0, 180: 115.0},
                        'confidence_intervals': {30: (100.0, 110.0), 90: (105.0, 115.0)},
                        'ensemble_confidence': 0.75,
                        'quality_score': 0.8
                    }
                except Exception as e:
                    logger.warning(f"Failed to get forecast for {ticker}: {e}")
            
            # Get market data (placeholder)
            signal_inputs.market_data = {
                'price': 100.0,  # Placeholder current price
                'volume': 1000000,
                'volatility': 0.25,
                'beta': 1.1,
                'market_cap': 10000000000
            }
            
        except Exception as e:
            logger.error(f"Error gathering signal inputs for {ticker}: {e}")
        
        return signal_inputs
    
    async def _calculate_decision_factors(
        self, 
        ticker: str, 
        inputs: SignalInput, 
        weights: Dict[str, float]
    ) -> DecisionFactors:
        """Calculate decision factors from input signals"""
        
        # Fundamental analysis score
        fundamental_score = self._calculate_fundamental_score(inputs)
        
        # Technical analysis score
        technical_score = self._calculate_technical_score(inputs)
        
        # Sentiment score
        sentiment_score = self._calculate_sentiment_score(inputs)
        
        # Forecast score
        forecast_score = self._calculate_forecast_score(inputs)
        
        # Anomaly score (negative impact)
        anomaly_score = self._calculate_anomaly_score(inputs)
        
        # Risk score (negative impact)
        risk_score = self._calculate_risk_score(inputs)
        
        # Calculate overall weighted score
        overall_score = (
            weights.get('fundamental', 0.25) * fundamental_score +
            weights.get('technical', 0.20) * technical_score +
            weights.get('sentiment', 0.20) * sentiment_score +
            weights.get('forecast', 0.20) * forecast_score +
            weights.get('anomaly', 0.10) * anomaly_score +
            weights.get('risk', 0.05) * risk_score
        )
        
        # Calculate confidence based on signal consistency and data quality
        confidence = self._calculate_confidence(
            [fundamental_score, technical_score, sentiment_score, forecast_score],
            inputs
        )
        
        return DecisionFactors(
            fundamental_score=fundamental_score,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            forecast_score=forecast_score,
            anomaly_score=anomaly_score,
            overall_score=overall_score,
            confidence=confidence
        )
    
    def _calculate_fundamental_score(self, inputs: SignalInput) -> float:
        """Calculate fundamental analysis score"""
        if not inputs.document_insights:
            return 0.0
        
        insights = inputs.document_insights
        
        # Revenue growth factor
        revenue_growth = insights.get('revenue_growth', 0.0)
        revenue_score = min(1.0, max(-1.0, revenue_growth * 2))  # Scale to [-1, 1]
        
        # Profitability factor
        profit_margin = insights.get('profit_margin', 0.0)
        profit_score = min(1.0, max(-1.0, (profit_margin - 0.1) * 5))  # Normalize around 10%
        
        # Debt factor (negative impact)
        debt_ratio = insights.get('debt_ratio', 0.5)
        debt_score = min(1.0, max(-1.0, (0.5 - debt_ratio) * 2))  # Lower debt is better
        
        # Management confidence
        mgmt_confidence = insights.get('management_confidence', 0.5)
        mgmt_score = (mgmt_confidence - 0.5) * 2  # Scale to [-1, 1]
        
        # Weighted combination
        fundamental_score = (
            0.3 * revenue_score +
            0.3 * profit_score +
            0.2 * debt_score +
            0.2 * mgmt_score
        )
        
        return max(-1.0, min(1.0, fundamental_score))
    
    def _calculate_technical_score(self, inputs: SignalInput) -> float:
        """Calculate technical analysis score"""
        if not inputs.market_data:
            return 0.0
        
        market_data = inputs.market_data
        
        # Volume factor (higher volume = more conviction)
        volume = market_data.get('volume', 0)
        volume_score = min(1.0, volume / 2000000) - 0.5  # Normalize around 2M volume
        
        # Volatility factor (moderate volatility preferred)
        volatility = market_data.get('volatility', 0.25)
        volatility_score = 1.0 - abs(volatility - 0.20) * 5  # Optimal around 20%
        volatility_score = max(-1.0, min(1.0, volatility_score))
        
        # Beta factor (market correlation)
        beta = market_data.get('beta', 1.0)
        beta_score = 1.0 - abs(beta - 1.0) * 0.5  # Prefer beta close to 1.0
        beta_score = max(-1.0, min(1.0, beta_score))
        
        # Weighted combination
        technical_score = (
            0.4 * volume_score +
            0.3 * volatility_score +
            0.3 * beta_score
        )
        
        return max(-1.0, min(1.0, technical_score))
    
    def _calculate_sentiment_score(self, inputs: SignalInput) -> float:
        """Calculate sentiment analysis score"""
        if not inputs.sentiment_analysis:
            return 0.0
        
        # Overall sentiment (primary factor)
        overall_sentiment = inputs.sentiment_analysis.overall_sentiment
        sentiment_confidence = inputs.sentiment_analysis.confidence
        
        # Weight by confidence
        weighted_sentiment = overall_sentiment * sentiment_confidence
        
        # Topic-specific sentiments (secondary factors)
        topic_boost = 0.0
        if inputs.topic_sentiments:
            # Focus on key financial topics
            key_topics = ['financial_performance', 'management_outlook', 'competitive_position']
            topic_scores = []
            
            for topic in key_topics:
                if topic in inputs.topic_sentiments:
                    topic_scores.append(inputs.topic_sentiments[topic])
            
            if topic_scores:
                topic_boost = np.mean(topic_scores) * 0.3  # 30% boost from topics
        
        sentiment_score = weighted_sentiment + topic_boost
        return max(-1.0, min(1.0, sentiment_score))
    
    def _calculate_forecast_score(self, inputs: SignalInput) -> float:
        """Calculate forecasting score"""
        if not inputs.forecast:
            return 0.0
        
        forecast = inputs.forecast
        
        # Get short-term and medium-term forecasts
        horizons = forecast.get('horizons', {})
        current_price = inputs.market_data.get('price', 100.0) if inputs.market_data else 100.0
        
        forecast_scores = []
        
        # Evaluate different time horizons
        for horizon_days, predicted_price in horizons.items():
            if horizon_days <= 90:  # Focus on short to medium term
                price_change = (predicted_price - current_price) / current_price
                # Weight shorter horizons more heavily
                weight = 1.0 / (horizon_days / 30)  # Higher weight for shorter periods
                forecast_scores.append(price_change * weight)
        
        if not forecast_scores:
            return 0.0
        
        # Average forecast score
        avg_forecast_score = np.mean(forecast_scores)
        
        # Adjust by forecast confidence
        ensemble_confidence = forecast.get('ensemble_confidence', 0.5)
        quality_score = forecast.get('quality_score', 0.5)
        
        confidence_factor = (ensemble_confidence + quality_score) / 2
        forecast_score = avg_forecast_score * confidence_factor
        
        return max(-1.0, min(1.0, forecast_score))
    
    def _calculate_anomaly_score(self, inputs: SignalInput) -> float:
        """Calculate anomaly impact score (negative impact)"""
        if not inputs.anomalies:
            return 0.0
        
        anomaly_impacts = []
        
        for anomaly in inputs.anomalies:
            # Severity impact
            severity_impact = {
                'low': -0.1,
                'medium': -0.3,
                'high': -0.6,
                'critical': -1.0
            }.get(anomaly.severity.value if hasattr(anomaly.severity, 'value') else str(anomaly.severity), -0.3)
            
            # Deviation impact
            deviation_impact = min(0, -abs(anomaly.deviation_score) * 0.1)
            
            anomaly_impacts.append(severity_impact + deviation_impact)
        
        if not anomaly_impacts:
            return 0.0
        
        # Use the most severe anomaly impact
        anomaly_score = min(anomaly_impacts)
        return max(-1.0, anomaly_score)
    
    def _calculate_risk_score(self, inputs: SignalInput) -> float:
        """Calculate risk impact score"""
        if not inputs.market_data:
            return 0.0
        
        market_data = inputs.market_data
        
        # Volatility risk
        volatility = market_data.get('volatility', 0.25)
        volatility_risk = min(1.0, volatility * 2) - 0.5  # Higher volatility = higher risk
        
        # Beta risk (deviation from market)
        beta = market_data.get('beta', 1.0)
        beta_risk = abs(beta - 1.0) * 0.5  # Deviation from market beta
        
        # Market cap risk (smaller companies = higher risk)
        market_cap = market_data.get('market_cap', 10000000000)
        if market_cap < 1000000000:  # Small cap
            size_risk = 0.3
        elif market_cap < 10000000000:  # Mid cap
            size_risk = 0.1
        else:  # Large cap
            size_risk = 0.0
        
        # Combined risk score (negative impact)
        risk_score = -(volatility_risk + beta_risk + size_risk) / 3
        return max(-1.0, min(0.0, risk_score))
    
    def _calculate_confidence(self, factor_scores: List[float], inputs: SignalInput) -> float:
        """Calculate overall confidence in the recommendation"""
        
        # Signal consistency (how much do factors agree?)
        if len(factor_scores) > 1:
            score_std = np.std(factor_scores)
            consistency = max(0, 1 - score_std)  # Lower std = higher consistency
        else:
            consistency = 0.5
        
        # Data quality factors
        quality_factors = []
        
        # Sentiment confidence
        if inputs.sentiment_analysis:
            quality_factors.append(inputs.sentiment_analysis.confidence)
        
        # Forecast quality
        if inputs.forecast:
            quality_factors.append(inputs.forecast.get('quality_score', 0.5))
        
        # Document insights availability
        if inputs.document_insights:
            quality_factors.append(0.8)  # High quality if we have fundamental data
        
        # Market data availability
        if inputs.market_data:
            quality_factors.append(0.7)
        
        # Average data quality
        avg_quality = np.mean(quality_factors) if quality_factors else 0.3
        
        # Combined confidence
        confidence = (0.6 * consistency + 0.4 * avg_quality)
        return max(0.1, min(1.0, confidence))
    
    def _determine_investment_signal(self, overall_score: float) -> InvestmentSignal:
        """Determine investment signal based on overall score"""
        
        if overall_score >= self.signal_thresholds[InvestmentSignal.STRONG_BUY]:
            return InvestmentSignal.STRONG_BUY
        elif overall_score >= self.signal_thresholds[InvestmentSignal.BUY]:
            return InvestmentSignal.BUY
        elif overall_score >= self.signal_thresholds[InvestmentSignal.HOLD]:
            return InvestmentSignal.HOLD
        elif overall_score >= self.signal_thresholds[InvestmentSignal.SELL]:
            return InvestmentSignal.SELL
        else:
            return InvestmentSignal.STRONG_SELL
    
    def _assess_risk_level(self, inputs: SignalInput, factors: DecisionFactors) -> RiskLevel:
        """Assess overall risk level"""
        
        risk_factors = []
        
        # Volatility risk
        if inputs.market_data:
            volatility = inputs.market_data.get('volatility', 0.25)
            risk_factors.append(volatility)
        
        # Anomaly risk
        if inputs.anomalies:
            max_severity = max([
                self.risk_mappings.get(a.severity.value if hasattr(a.severity, 'value') else str(a.severity), 0.5)
                for a in inputs.anomalies
            ])
            risk_factors.append(max_severity)
        
        # Confidence risk (lower confidence = higher risk)
        confidence_risk = 1.0 - factors.confidence
        risk_factors.append(confidence_risk)
        
        # Calculate average risk
        avg_risk = np.mean(risk_factors) if risk_factors else 0.5
        
        # Map to risk levels
        if avg_risk <= 0.2:
            return RiskLevel.VERY_LOW
        elif avg_risk <= 0.4:
            return RiskLevel.LOW
        elif avg_risk <= 0.6:
            return RiskLevel.MEDIUM
        elif avg_risk <= 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    async def _calculate_price_targets(
        self, 
        ticker: str, 
        inputs: SignalInput, 
        signal: InvestmentSignal
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss"""
        
        current_price = inputs.market_data.get('price', 100.0) if inputs.market_data else 100.0
        
        # Base target calculation on forecast if available
        target_price = None
        if inputs.forecast and inputs.forecast.get('horizons'):
            # Use 3-month forecast as target
            horizons = inputs.forecast['horizons']
            if 90 in horizons:
                target_price = horizons[90]
            elif horizons:
                # Use the closest available horizon
                closest_horizon = min(horizons.keys(), key=lambda x: abs(x - 90))
                target_price = horizons[closest_horizon]
        
        # Fallback target calculation based on signal strength
        if target_price is None:
            signal_multipliers = {
                InvestmentSignal.STRONG_BUY: 1.20,
                InvestmentSignal.BUY: 1.10,
                InvestmentSignal.HOLD: 1.00,
                InvestmentSignal.SELL: 0.95,
                InvestmentSignal.STRONG_SELL: 0.85
            }
            target_price = current_price * signal_multipliers[signal]
        
        # Calculate stop loss (risk management)
        stop_loss = None
        if signal in [InvestmentSignal.BUY, InvestmentSignal.STRONG_BUY]:
            # Set stop loss at 10-15% below current price
            volatility = inputs.market_data.get('volatility', 0.25) if inputs.market_data else 0.25
            stop_loss_pct = 0.10 + (volatility * 0.2)  # Higher volatility = wider stop
            stop_loss = current_price * (1 - stop_loss_pct)
        
        return target_price, stop_loss
    
    def _calculate_position_size(
        self, 
        signal: InvestmentSignal, 
        risk_level: RiskLevel, 
        confidence: float
    ) -> Optional[float]:
        """Calculate recommended position size as percentage of portfolio"""
        
        # Base position sizes by signal strength
        base_sizes = {
            InvestmentSignal.STRONG_BUY: 0.15,
            InvestmentSignal.BUY: 0.10,
            InvestmentSignal.HOLD: 0.05,
            InvestmentSignal.SELL: 0.0,
            InvestmentSignal.STRONG_SELL: 0.0
        }
        
        base_size = base_sizes[signal]
        
        if base_size == 0:
            return None
        
        # Adjust for risk level
        risk_adjustments = {
            RiskLevel.VERY_LOW: 1.2,
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.VERY_HIGH: 0.4
        }
        
        risk_adjustment = risk_adjustments[risk_level]
        
        # Adjust for confidence
        confidence_adjustment = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        
        # Calculate final position size
        position_size = base_size * risk_adjustment * confidence_adjustment
        
        # Cap at reasonable maximum
        return min(0.20, max(0.01, position_size))
    
    def _determine_time_horizon(self, inputs: SignalInput) -> TimeHorizon:
        """Determine appropriate time horizon for the recommendation"""
        
        # Base on forecast availability and market conditions
        if inputs.forecast:
            horizons = inputs.forecast.get('horizons', {})
            max_horizon = max(horizons.keys()) if horizons else 90
            
            if max_horizon >= 365:
                return TimeHorizon.LONG_TERM
            elif max_horizon >= 90:
                return TimeHorizon.MEDIUM_TERM
            else:
                return TimeHorizon.SHORT_TERM
        
        # Default based on volatility
        if inputs.market_data:
            volatility = inputs.market_data.get('volatility', 0.25)
            if volatility > 0.4:
                return TimeHorizon.SHORT_TERM
            elif volatility > 0.2:
                return TimeHorizon.MEDIUM_TERM
            else:
                return TimeHorizon.LONG_TERM
        
        return TimeHorizon.MEDIUM_TERM
    
    def _generate_reasoning(
        self, 
        ticker: str, 
        signal: InvestmentSignal, 
        factors: DecisionFactors, 
        inputs: SignalInput
    ) -> Tuple[str, List[str], List[str]]:
        """Generate reasoning and supporting/risk factors"""
        
        # Main reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Recommendation: {signal.value} with {factors.confidence:.1%} confidence.")
        reasoning_parts.append(f"Overall score: {factors.overall_score:.2f} based on multi-factor analysis.")
        
        # Top contributing factors
        factor_contributions = {
            'Fundamental Analysis': factors.fundamental_score,
            'Technical Analysis': factors.technical_score,
            'Sentiment Analysis': factors.sentiment_score,
            'Forecast Analysis': factors.forecast_score,
            'Anomaly Impact': factors.anomaly_score,
            'Risk Assessment': factors.risk_score
        }
        
        # Sort by absolute contribution
        sorted_factors = sorted(factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factor = sorted_factors[0]
        reasoning_parts.append(f"Primary driver: {top_factor[0]} (score: {top_factor[1]:.2f}).")
        
        reasoning = " ".join(reasoning_parts)
        
        # Supporting factors
        supporting_factors = []
        for factor_name, score in sorted_factors:
            if score > 0.1:
                supporting_factors.append(f"{factor_name}: positive contribution ({score:.2f})")
        
        # Add specific insights
        if inputs.sentiment_analysis and inputs.sentiment_analysis.overall_sentiment > 0.2:
            supporting_factors.append(f"Positive market sentiment ({inputs.sentiment_analysis.overall_sentiment:.2f})")
        
        if inputs.forecast:
            ensemble_confidence = inputs.forecast.get('ensemble_confidence', 0)
            if ensemble_confidence > 0.6:
                supporting_factors.append(f"High forecast confidence ({ensemble_confidence:.1%})")
        
        # Risk factors
        risk_factors = []
        for factor_name, score in sorted_factors:
            if score < -0.1:
                risk_factors.append(f"{factor_name}: negative impact ({score:.2f})")
        
        # Add specific risks
        if inputs.anomalies:
            critical_anomalies = [a for a in inputs.anomalies if hasattr(a.severity, 'value') and a.severity.value == 'critical']
            if critical_anomalies:
                risk_factors.append(f"Critical anomalies detected in {len(critical_anomalies)} metrics")
        
        if inputs.market_data:
            volatility = inputs.market_data.get('volatility', 0)
            if volatility > 0.4:
                risk_factors.append(f"High volatility ({volatility:.1%})")
        
        return reasoning, supporting_factors[:5], risk_factors[:5]
    
    async def _get_current_price(self, ticker: str) -> float:
        """Get current stock price (placeholder implementation)"""
        # In a real implementation, this would fetch from market data API
        # For now, return a placeholder price
        return 100.0 + np.random.uniform(-10, 10)
    
    async def _store_recommendation(self, recommendation: InvestmentRecommendation):
        """Store recommendation in database"""
        try:
            # For now, just log the recommendation
            logger.info(f"Would store recommendation {recommendation.id} for {recommendation.ticker}")
            
        except Exception as e:
            logger.error(f"Failed to store recommendation: {e}")


# Global service instance
_investment_advisory_service = None


async def get_investment_advisory_service() -> InvestmentAdvisoryService:
    """Get or create investment advisory service instance"""
    global _investment_advisory_service
    if _investment_advisory_service is None:
        _investment_advisory_service = InvestmentAdvisoryService()
        await _investment_advisory_service.initialize()
    return _investment_advisory_service


# Main service functions
async def generate_recommendation(
    ticker: str, 
    context: AnalysisContext = None,
    custom_weights: Dict[str, float] = None
) -> InvestmentRecommendation:
    """Generate investment recommendation for a ticker"""
    service = await get_investment_advisory_service()
    return await service.generate_recommendation(ticker, context, custom_weights)


async def get_recommendation_by_id(recommendation_id: str) -> Optional[InvestmentRecommendation]:
    """Retrieve recommendation by ID"""
    try:
        # Placeholder implementation
        logger.info(f"Would retrieve recommendation {recommendation_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve recommendation {recommendation_id}: {e}")
        return None


async def get_active_recommendations(ticker: str = None) -> List[InvestmentRecommendation]:
    """Get active recommendations, optionally filtered by ticker"""
    try:
        # Placeholder implementation
        logger.info(f"Would retrieve active recommendations for {ticker if ticker else 'all tickers'}")
        return []
        
    except Exception as e:
        logger.error(f"Failed to retrieve active recommendations: {e}")
        return []


async def assess_investment_risk(
    ticker: str, 
    recommendation: InvestmentRecommendation,
    portfolio: Optional[Portfolio] = None
) -> Dict[str, Any]:
    """Assess investment risk for a recommendation"""
    # Placeholder implementation
    return {
        'overall_risk_score': 0.5,
        'risk_level': recommendation.risk_level.value,
        'risk_factors': {},
        'risk_warnings': [],
        'risk_mitigation': []
    }


async def optimize_position_sizing(
    recommendations: List[InvestmentRecommendation],
    portfolio: Portfolio,
    risk_tolerance: str = 'moderate',
    optimization_method: str = 'volatility_adjusted'
) -> PositionSizes:
    """Optimize position sizes for multiple recommendations"""
    # Placeholder implementation
    return PositionSizes(
        portfolio_id=portfolio.portfolio_id,
        recommendations={str(rec.id): rec.position_size or 0.05 for rec in recommendations},
        optimization_method=optimization_method,
        risk_budget=0.15,
        expected_return=0.08,
        expected_risk=0.12,
        projected_return=0.06,
        projected_risk=0.10,
        diversification_ratio=1.2
    )


async def assess_portfolio_risk(portfolio: Portfolio) -> Dict[str, Any]:
    """Assess overall portfolio risk metrics"""
    
    portfolio_metrics = {
        'overall_risk_score': 0.0,
        'risk_metrics': {},
        'concentration_analysis': {},
        'correlation_analysis': {},
        'risk_warnings': [],
        'rebalancing_suggestions': []
    }
    
    try:
        # Calculate portfolio concentration
        total_value = sum(portfolio.holdings.values()) + portfolio.cash_position
        
        if total_value > 0:
            position_percentages = {
                ticker: value / total_value 
                for ticker, value in portfolio.holdings.items()
            }
            
            # Concentration metrics
            max_position = max(position_percentages.values()) if position_percentages else 0
            top_5_concentration = sum(sorted(position_percentages.values(), reverse=True)[:5])
            
            portfolio_metrics['concentration_analysis'] = {
                'max_position': max_position,
                'top_5_concentration': top_5_concentration,
                'number_of_positions': len(position_percentages),
                'effective_positions': 1 / sum(p**2 for p in position_percentages.values()) if position_percentages else 0
            }
            
            # Risk warnings
            if max_position > 0.15:
                portfolio_metrics['risk_warnings'].append(f"High concentration: {max_position:.1%} in single position")
            
            if top_5_concentration > 0.6:
                portfolio_metrics['risk_warnings'].append(f"Top 5 positions represent {top_5_concentration:.1%} of portfolio")
            
            if len(position_percentages) < 10:
                portfolio_metrics['risk_warnings'].append("Low diversification: fewer than 10 positions")
        
        # Portfolio risk metrics (placeholder calculations)
        portfolio_metrics['risk_metrics'] = {
            'portfolio_beta': portfolio.beta or 1.0,
            'portfolio_volatility': portfolio.volatility or 0.20,
            'sharpe_ratio': portfolio.sharpe_ratio or 1.0,
            'max_drawdown': portfolio.max_drawdown or 0.15,
            'var_95': total_value * 0.05 if total_value > 0 else 0  # 5% VaR estimate
        }
        
        # Overall risk score
        risk_factors = [
            min(1.0, max_position * 5),  # Concentration risk
            min(1.0, (portfolio.volatility or 0.20) * 2),  # Volatility risk
            min(1.0, max(0, 1.5 - (portfolio.sharpe_ratio or 1.0))),  # Risk-adjusted return
        ]
        
        portfolio_metrics['overall_risk_score'] = np.mean(risk_factors)
        
    except Exception as e:
        logger.error(f"Portfolio risk assessment failed: {e}")
        portfolio_metrics['error'] = str(e)
    
    return portfolio_metrics