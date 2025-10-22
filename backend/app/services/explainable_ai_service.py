"""
Explainable AI Service for Investment Recommendations

This service provides transparency and explainability for investment recommendations:
- Natural language explanation generation
- Feature importance ranking and visualization
- Decision factor analysis and reasoning
- Alternative scenario analysis
- Complete audit trail with decision history
- Model interpretability and confidence factors

Features:
- Comprehensive recommendation explanations
- Visual decision tree representations
- Risk factor analysis and mitigation strategies
- Data quality assessment and limitations
- Regulatory compliance documentation
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
from scipy import stats

from ..models.recommendation import (
    InvestmentRecommendation, InvestmentSignal, RiskLevel, TimeHorizon,
    RecommendationExplanation, Portfolio, PositionSizes, AnalysisContext,
    RecommendationStatus, RecommendationPerformance
)
from ..models.sentiment import SentimentAnalysis, SentimentTopic
from ..models.anomaly import Anomaly, AnomalySeverity
from ..config import get_settings
from ..database.connection import get_database

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


class ExplainableAIEngine:
    """Explainable AI engine for investment recommendation transparency"""
    
    def __init__(self):
        self.explanation_templates = {
            'executive_summary': self._generate_executive_summary,
            'risk_explanation': self._generate_risk_explanation,
            'alternative_scenarios': self._generate_alternative_scenarios,
            'confidence_factors': self._generate_confidence_factors
        }
    
    async def explain_recommendation(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput,
        risk_assessment: Optional[Dict[str, Any]] = None
    ) -> RecommendationExplanation:
        """
        Generate comprehensive explanation for investment recommendation
        
        Args:
            recommendation: The investment recommendation to explain
            decision_factors: Decision factors used in recommendation
            signal_inputs: Input signals and data
            risk_assessment: Risk assessment results (optional)
        
        Returns:
            Detailed recommendation explanation with transparency
        """
        
        try:
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                recommendation, decision_factors, signal_inputs
            )
            
            # Generate detailed factor analysis
            fundamental_analysis = await self._generate_fundamental_analysis(
                recommendation, decision_factors, signal_inputs
            )
            
            technical_analysis = await self._generate_technical_analysis(
                recommendation, decision_factors, signal_inputs
            )
            
            sentiment_analysis = await self._generate_sentiment_explanation(
                recommendation, decision_factors, signal_inputs
            )
            
            risk_analysis = await self._generate_risk_explanation(
                recommendation, decision_factors, signal_inputs, risk_assessment
            )
            
            # Extract key metrics
            key_metrics = await self._extract_key_metrics(
                recommendation, decision_factors, signal_inputs
            )
            
            # Generate feature importance
            feature_importance = await self._calculate_feature_importance(
                decision_factors, signal_inputs
            )
            
            # Generate model confidence factors
            confidence_factors = await self._generate_confidence_factors(
                recommendation, decision_factors, signal_inputs
            )
            
            # Generate alternative scenarios
            alternative_scenarios = await self._generate_alternative_scenarios(
                recommendation, decision_factors, signal_inputs
            )
            
            # Generate assumptions and limitations
            assumptions = await self._generate_assumptions(signal_inputs)
            limitations = await self._generate_limitations(signal_inputs)
            
            # Data quality assessment
            data_quality_notes = await self._assess_data_quality(signal_inputs)
            
            return RecommendationExplanation(
                recommendation_id=recommendation.id,
                executive_summary=executive_summary,
                fundamental_analysis=fundamental_analysis,
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                risk_analysis=risk_analysis,
                key_metrics=key_metrics,
                feature_importance=feature_importance,
                model_confidence_factors=confidence_factors,
                alternative_scenarios=alternative_scenarios,
                assumptions=assumptions,
                limitations=limitations,
                data_quality_notes=data_quality_notes
            )
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for recommendation {recommendation.id}: {e}")
            raise
    
    async def _generate_executive_summary(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> str:
        """Generate executive summary of the recommendation"""
        
        summary_parts = []
        
        # Main recommendation
        summary_parts.append(
            f"Our analysis recommends a {recommendation.signal.value} position in {recommendation.ticker} "
            f"with {recommendation.confidence:.0%} confidence."
        )
        
        # Key driver
        factor_scores = {
            'fundamental': decision_factors.fundamental_score,
            'technical': decision_factors.technical_score,
            'sentiment': decision_factors.sentiment_score,
            'forecast': decision_factors.forecast_score
        }
        
        primary_driver = max(factor_scores.items(), key=lambda x: abs(x[1]))
        if abs(primary_driver[1]) > 0.2:
            direction = "positive" if primary_driver[1] > 0 else "negative"
            summary_parts.append(
                f"The primary driver is {direction} {primary_driver[0]} analysis "
                f"(score: {primary_driver[1]:.2f})."
            )
        
        # Price target
        if recommendation.target_price:
            upside = (recommendation.target_price - recommendation.current_price) / recommendation.current_price
            summary_parts.append(
                f"Price target of ${recommendation.target_price:.2f} implies {upside:.1%} "
                f"{'upside' if upside > 0 else 'downside'} from current price of ${recommendation.current_price:.2f}."
            )
        
        # Risk assessment
        summary_parts.append(
            f"Risk level is assessed as {recommendation.risk_level.value.replace('_', ' ').title()} "
            f"with recommended position size of {recommendation.position_size:.1%} of portfolio."
        )
        
        # Time horizon
        summary_parts.append(
            f"This recommendation is suitable for {recommendation.time_horizon.value.replace('_', ' ')} investors."
        )
        
        return " ".join(summary_parts)
    
    async def _generate_fundamental_analysis(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> Optional[str]:
        """Generate fundamental analysis explanation"""
        
        if not signal_inputs.document_insights:
            return None
        
        insights = signal_inputs.document_insights
        analysis_parts = []
        
        analysis_parts.append(
            f"Fundamental analysis score: {decision_factors.fundamental_score:.2f}. "
        )
        
        # Revenue analysis
        revenue_growth = insights.get('revenue_growth', 0)
        if revenue_growth > 0.1:
            analysis_parts.append(f"Strong revenue growth of {revenue_growth:.1%} indicates healthy business expansion.")
        elif revenue_growth < 0:
            analysis_parts.append(f"Revenue decline of {abs(revenue_growth):.1%} raises concerns about business performance.")
        else:
            analysis_parts.append(f"Modest revenue growth of {revenue_growth:.1%} suggests stable but limited expansion.")
        
        # Profitability analysis
        profit_margin = insights.get('profit_margin', 0)
        if profit_margin > 0.15:
            analysis_parts.append(f"Excellent profit margins of {profit_margin:.1%} demonstrate strong operational efficiency.")
        elif profit_margin < 0.05:
            analysis_parts.append(f"Low profit margins of {profit_margin:.1%} indicate potential operational challenges.")
        else:
            analysis_parts.append(f"Profit margins of {profit_margin:.1%} are within industry norms.")
        
        # Debt analysis
        debt_ratio = insights.get('debt_ratio', 0.5)
        if debt_ratio > 0.6:
            analysis_parts.append(f"High debt ratio of {debt_ratio:.1%} presents financial leverage risks.")
        elif debt_ratio < 0.3:
            analysis_parts.append(f"Conservative debt ratio of {debt_ratio:.1%} provides financial stability.")
        else:
            analysis_parts.append(f"Moderate debt ratio of {debt_ratio:.1%} maintains balanced capital structure.")
        
        # Management confidence
        mgmt_confidence = insights.get('management_confidence', 0.5)
        if mgmt_confidence > 0.7:
            analysis_parts.append("Management expresses high confidence in future prospects.")
        elif mgmt_confidence < 0.3:
            analysis_parts.append("Management commentary suggests cautious outlook.")
        else:
            analysis_parts.append("Management maintains neutral stance on future performance.")
        
        return " ".join(analysis_parts)
    
    async def _generate_technical_analysis(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> Optional[str]:
        """Generate technical analysis explanation"""
        
        if not signal_inputs.market_data:
            return None
        
        market_data = signal_inputs.market_data
        analysis_parts = []
        
        analysis_parts.append(
            f"Technical analysis score: {decision_factors.technical_score:.2f}. "
        )
        
        # Volume analysis
        volume = market_data.get('volume', 0)
        if volume > 2000000:
            analysis_parts.append(f"High trading volume of {volume:,} shares indicates strong market interest.")
        elif volume < 500000:
            analysis_parts.append(f"Low trading volume of {volume:,} shares suggests limited market participation.")
        else:
            analysis_parts.append(f"Average trading volume of {volume:,} shares shows normal market activity.")
        
        # Volatility analysis
        volatility = market_data.get('volatility', 0.25)
        if volatility > 0.35:
            analysis_parts.append(f"High volatility of {volatility:.1%} indicates significant price uncertainty.")
        elif volatility < 0.15:
            analysis_parts.append(f"Low volatility of {volatility:.1%} suggests stable price behavior.")
        else:
            analysis_parts.append(f"Moderate volatility of {volatility:.1%} is typical for this asset class.")
        
        # Beta analysis
        beta = market_data.get('beta', 1.0)
        if beta > 1.3:
            analysis_parts.append(f"High beta of {beta:.2f} indicates amplified market movements.")
        elif beta < 0.7:
            analysis_parts.append(f"Low beta of {beta:.2f} suggests defensive characteristics.")
        else:
            analysis_parts.append(f"Beta of {beta:.2f} shows typical market correlation.")
        
        return " ".join(analysis_parts)
    
    async def _generate_sentiment_explanation(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> Optional[str]:
        """Generate sentiment analysis explanation"""
        
        if not signal_inputs.sentiment_analysis:
            return None
        
        sentiment = signal_inputs.sentiment_analysis
        analysis_parts = []
        
        analysis_parts.append(
            f"Sentiment analysis score: {decision_factors.sentiment_score:.2f}. "
        )
        
        # Overall sentiment
        overall_sentiment = sentiment.overall_sentiment
        if overall_sentiment > 0.3:
            analysis_parts.append(f"Positive market sentiment ({overall_sentiment:.2f}) supports bullish outlook.")
        elif overall_sentiment < -0.3:
            analysis_parts.append(f"Negative market sentiment ({overall_sentiment:.2f}) creates headwinds.")
        else:
            analysis_parts.append(f"Neutral market sentiment ({overall_sentiment:.2f}) provides balanced perspective.")
        
        # Confidence in sentiment
        confidence = sentiment.confidence
        if confidence > 0.8:
            analysis_parts.append(f"High confidence ({confidence:.1%}) in sentiment analysis strengthens signal.")
        elif confidence < 0.5:
            analysis_parts.append(f"Low confidence ({confidence:.1%}) in sentiment analysis suggests uncertainty.")
        else:
            analysis_parts.append(f"Moderate confidence ({confidence:.1%}) in sentiment assessment.")
        
        # Topic-specific sentiments
        if signal_inputs.topic_sentiments:
            positive_topics = []
            negative_topics = []
            
            for topic, score in signal_inputs.topic_sentiments.items():
                if score > 0.2:
                    positive_topics.append(topic.replace('_', ' '))
                elif score < -0.2:
                    negative_topics.append(topic.replace('_', ' '))
            
            if positive_topics:
                analysis_parts.append(f"Positive sentiment in: {', '.join(positive_topics)}.")
            
            if negative_topics:
                analysis_parts.append(f"Negative sentiment in: {', '.join(negative_topics)}.")
        
        return " ".join(analysis_parts)
    
    async def _generate_risk_explanation(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput,
        risk_assessment: Optional[Dict[str, Any]]
    ) -> str:
        """Generate risk analysis explanation"""
        
        analysis_parts = []
        
        # Overall risk assessment
        analysis_parts.append(
            f"Risk assessment: {recommendation.risk_level.value.replace('_', ' ').title()} risk level "
            f"with overall risk score of {decision_factors.risk_score:.2f}."
        )
        
        # Risk factors from recommendation
        if recommendation.risk_factors:
            analysis_parts.append(f"Key risk factors include: {', '.join(recommendation.risk_factors[:3])}.")
        
        # Detailed risk assessment if available
        if risk_assessment:
            overall_risk = risk_assessment.get('overall_risk_score', 0)
            analysis_parts.append(f"Comprehensive risk analysis yields {overall_risk:.1%} risk score.")
            
            # Top risk factors
            risk_factors = risk_assessment.get('risk_factors', {})
            top_risks = sorted(
                [(name, data.get('score', 0)) for name, data in risk_factors.items()],
                key=lambda x: x[1],
                reverse=True
            )[:2]
            
            if top_risks:
                risk_names = [name.replace('_', ' ') for name, _ in top_risks]
                analysis_parts.append(f"Primary risk sources: {', '.join(risk_names)}.")
            
            # Risk warnings
            warnings = risk_assessment.get('risk_warnings', [])
            if warnings:
                analysis_parts.append(f"Risk warnings: {warnings[0]}")
        
        # Position sizing rationale
        if recommendation.position_size:
            analysis_parts.append(
                f"Recommended position size of {recommendation.position_size:.1%} balances "
                f"opportunity with risk management."
            )
        
        # Stop loss rationale
        if recommendation.stop_loss:
            stop_loss_pct = (recommendation.current_price - recommendation.stop_loss) / recommendation.current_price
            analysis_parts.append(
                f"Stop loss at ${recommendation.stop_loss:.2f} ({stop_loss_pct:.1%} below current price) "
                f"limits downside risk."
            )
        
        return " ".join(analysis_parts)
    
    async def _extract_key_metrics(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> Dict[str, Any]:
        """Extract key metrics for the recommendation"""
        
        metrics = {}
        
        # Recommendation metrics
        metrics['signal'] = recommendation.signal.value
        metrics['confidence'] = recommendation.confidence
        metrics['overall_score'] = decision_factors.overall_score
        
        # Price metrics
        metrics['current_price'] = recommendation.current_price
        if recommendation.target_price:
            metrics['target_price'] = recommendation.target_price
            metrics['upside_potential'] = (recommendation.target_price - recommendation.current_price) / recommendation.current_price
        
        # Risk metrics
        metrics['risk_level'] = recommendation.risk_level.value
        metrics['position_size'] = recommendation.position_size
        
        # Factor scores
        metrics['fundamental_score'] = decision_factors.fundamental_score
        metrics['technical_score'] = decision_factors.technical_score
        metrics['sentiment_score'] = decision_factors.sentiment_score
        metrics['forecast_score'] = decision_factors.forecast_score
        
        # Market data metrics
        if signal_inputs.market_data:
            market_data = signal_inputs.market_data
            metrics['volatility'] = market_data.get('volatility')
            metrics['beta'] = market_data.get('beta')
            metrics['volume'] = market_data.get('volume')
            metrics['market_cap'] = market_data.get('market_cap')
        
        # Sentiment metrics
        if signal_inputs.sentiment_analysis:
            metrics['sentiment_score'] = signal_inputs.sentiment_analysis.overall_sentiment
            metrics['sentiment_confidence'] = signal_inputs.sentiment_analysis.confidence
        
        # Forecast metrics
        if signal_inputs.forecast:
            forecast = signal_inputs.forecast
            metrics['forecast_confidence'] = forecast.get('ensemble_confidence')
            metrics['forecast_quality'] = forecast.get('quality_score')
        
        return metrics
    
    async def _calculate_feature_importance(
        self, 
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> Dict[str, float]:
        """Calculate feature importance for the recommendation"""
        
        # Base feature importance on factor scores and their contribution
        factor_scores = {
            'fundamental_analysis': abs(decision_factors.fundamental_score),
            'technical_analysis': abs(decision_factors.technical_score),
            'sentiment_analysis': abs(decision_factors.sentiment_score),
            'forecast_analysis': abs(decision_factors.forecast_score),
            'anomaly_detection': abs(decision_factors.anomaly_score),
            'risk_assessment': abs(decision_factors.risk_score)
        }
        
        # Normalize to sum to 1.0
        total_importance = sum(factor_scores.values())
        if total_importance > 0:
            feature_importance = {
                factor: score / total_importance 
                for factor, score in factor_scores.items()
            }
        else:
            # Equal importance if no clear signals
            feature_importance = {
                factor: 1.0 / len(factor_scores) 
                for factor in factor_scores.keys()
            }
        
        # Add sub-feature importance
        if signal_inputs.document_insights:
            insights = signal_inputs.document_insights
            feature_importance['revenue_growth'] = min(0.1, abs(insights.get('revenue_growth', 0)) * 0.5)
            feature_importance['profit_margin'] = min(0.1, abs(insights.get('profit_margin', 0)) * 0.5)
            feature_importance['debt_ratio'] = min(0.1, abs(insights.get('debt_ratio', 0.5) - 0.5) * 0.5)
        
        if signal_inputs.market_data:
            market_data = signal_inputs.market_data
            feature_importance['volatility'] = min(0.1, market_data.get('volatility', 0.25) * 0.2)
            feature_importance['volume'] = min(0.1, market_data.get('volume', 1000000) / 10000000)
        
        return feature_importance
    
    async def _generate_confidence_factors(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> List[str]:
        """Generate factors that contribute to model confidence"""
        
        confidence_factors = []
        
        # Data quality factors
        if signal_inputs.document_insights:
            confidence_factors.append("Comprehensive fundamental data available")
        
        if signal_inputs.sentiment_analysis and signal_inputs.sentiment_analysis.confidence > 0.7:
            confidence_factors.append(f"High sentiment analysis confidence ({signal_inputs.sentiment_analysis.confidence:.1%})")
        
        if signal_inputs.forecast and signal_inputs.forecast.get('quality_score', 0) > 0.6:
            confidence_factors.append(f"High-quality forecasting models (quality: {signal_inputs.forecast['quality_score']:.1%})")
        
        # Signal consistency factors
        factor_scores = [
            decision_factors.fundamental_score,
            decision_factors.technical_score,
            decision_factors.sentiment_score,
            decision_factors.forecast_score
        ]
        
        # Check for signal agreement
        positive_signals = sum(1 for score in factor_scores if score > 0.2)
        negative_signals = sum(1 for score in factor_scores if score < -0.2)
        
        if positive_signals >= 3:
            confidence_factors.append("Multiple positive signals align")
        elif negative_signals >= 3:
            confidence_factors.append("Multiple negative signals align")
        
        # Model performance factors
        confidence_factors.append(f"Overall model confidence: {decision_factors.confidence:.1%}")
        
        # Risk assessment factors
        if recommendation.risk_level in [RiskLevel.LOW, RiskLevel.VERY_LOW]:
            confidence_factors.append("Low risk profile increases confidence")
        
        # Time horizon factors
        if recommendation.time_horizon == TimeHorizon.MEDIUM_TERM:
            confidence_factors.append("Medium-term horizon provides balanced perspective")
        
        return confidence_factors[:5]  # Return top 5 factors
    
    async def _generate_alternative_scenarios(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput
    ) -> List[Dict[str, Any]]:
        """Generate alternative scenarios and their implications"""
        
        scenarios = []
        
        # Bull case scenario
        bull_case = {
            'scenario': 'Bull Case',
            'probability': 0.3,
            'description': 'All positive factors materialize as expected',
            'price_target': recommendation.target_price * 1.2 if recommendation.target_price else recommendation.current_price * 1.3,
            'key_drivers': []
        }
        
        if decision_factors.fundamental_score > 0:
            bull_case['key_drivers'].append('Strong fundamental performance continues')
        if decision_factors.sentiment_score > 0:
            bull_case['key_drivers'].append('Positive sentiment momentum accelerates')
        if decision_factors.forecast_score > 0:
            bull_case['key_drivers'].append('Forecasting models prove accurate')
        
        scenarios.append(bull_case)
        
        # Base case scenario (current recommendation)
        base_case = {
            'scenario': 'Base Case',
            'probability': 0.4,
            'description': 'Current analysis proves accurate',
            'price_target': recommendation.target_price or recommendation.current_price * 1.1,
            'key_drivers': [
                'Recommendation factors perform as expected',
                'Market conditions remain stable',
                'No major surprises in fundamentals'
            ]
        }
        
        scenarios.append(base_case)
        
        # Bear case scenario
        bear_case = {
            'scenario': 'Bear Case',
            'probability': 0.3,
            'description': 'Negative factors dominate performance',
            'price_target': recommendation.current_price * 0.8,
            'key_drivers': []
        }
        
        if decision_factors.risk_score < -0.2:
            bear_case['key_drivers'].append('Risk factors materialize')
        if signal_inputs.anomalies:
            bear_case['key_drivers'].append('Anomalies indicate underlying problems')
        if decision_factors.sentiment_score < 0:
            bear_case['key_drivers'].append('Negative sentiment persists')
        
        if not bear_case['key_drivers']:
            bear_case['key_drivers'] = [
                'Market conditions deteriorate',
                'Unexpected negative developments',
                'Broader economic headwinds'
            ]
        
        scenarios.append(bear_case)
        
        return scenarios
    
    async def _generate_assumptions(self, signal_inputs: SignalInput) -> List[str]:
        """Generate key assumptions underlying the analysis"""
        
        assumptions = []
        
        # Market assumptions
        assumptions.append("Market conditions remain relatively stable")
        assumptions.append("No major regulatory changes affecting the sector")
        
        # Data assumptions
        if signal_inputs.document_insights:
            assumptions.append("Financial data accurately reflects company performance")
            assumptions.append("Management guidance is reliable and achievable")
        
        if signal_inputs.sentiment_analysis:
            assumptions.append("Market sentiment analysis captures true investor mood")
            assumptions.append("Sentiment trends continue in near term")
        
        if signal_inputs.forecast:
            assumptions.append("Historical patterns remain relevant for future predictions")
            assumptions.append("Forecasting models maintain their accuracy")
        
        # Economic assumptions
        assumptions.append("No major economic disruptions or black swan events")
        assumptions.append("Interest rate environment remains relatively stable")
        
        # Company-specific assumptions
        assumptions.append("Company maintains current business model and strategy")
        assumptions.append("Key management team remains in place")
        
        return assumptions[:6]  # Return top 6 assumptions
    
    async def _generate_limitations(self, signal_inputs: SignalInput) -> List[str]:
        """Generate limitations of the analysis"""
        
        limitations = []
        
        # Data limitations
        if not signal_inputs.document_insights:
            limitations.append("Limited fundamental data available for analysis")
        
        if not signal_inputs.sentiment_analysis or signal_inputs.sentiment_analysis.confidence < 0.6:
            limitations.append("Sentiment analysis has moderate confidence levels")
        
        if not signal_inputs.forecast:
            limitations.append("No forecasting data available for price predictions")
        
        # Model limitations
        limitations.append("Models based on historical data may not predict future events")
        limitations.append("Analysis cannot account for unforeseen market disruptions")
        limitations.append("Recommendation accuracy depends on data quality and completeness")
        
        # Market limitations
        limitations.append("Market conditions can change rapidly, affecting recommendation validity")
        limitations.append("Individual stock performance may deviate from model predictions")
        
        # Time limitations
        limitations.append("Analysis reflects point-in-time assessment and may become outdated")
        
        return limitations[:5]  # Return top 5 limitations
    
    async def _assess_data_quality(self, signal_inputs: SignalInput) -> Optional[str]:
        """Assess and report on data quality"""
        
        quality_factors = []
        quality_score = 0
        total_factors = 0
        
        # Document insights quality
        if signal_inputs.document_insights:
            quality_factors.append("Fundamental data: Available")
            quality_score += 1
        else:
            quality_factors.append("Fundamental data: Limited")
        total_factors += 1
        
        # Sentiment analysis quality
        if signal_inputs.sentiment_analysis:
            confidence = signal_inputs.sentiment_analysis.confidence
            if confidence > 0.7:
                quality_factors.append(f"Sentiment analysis: High quality ({confidence:.1%} confidence)")
                quality_score += 1
            elif confidence > 0.5:
                quality_factors.append(f"Sentiment analysis: Moderate quality ({confidence:.1%} confidence)")
                quality_score += 0.5
            else:
                quality_factors.append(f"Sentiment analysis: Low quality ({confidence:.1%} confidence)")
        else:
            quality_factors.append("Sentiment analysis: Not available")
        total_factors += 1
        
        # Forecast quality
        if signal_inputs.forecast:
            forecast_quality = signal_inputs.forecast.get('quality_score', 0.5)
            if forecast_quality > 0.7:
                quality_factors.append(f"Forecast data: High quality ({forecast_quality:.1%})")
                quality_score += 1
            elif forecast_quality > 0.5:
                quality_factors.append(f"Forecast data: Moderate quality ({forecast_quality:.1%})")
                quality_score += 0.5
            else:
                quality_factors.append(f"Forecast data: Low quality ({forecast_quality:.1%})")
        else:
            quality_factors.append("Forecast data: Not available")
        total_factors += 1
        
        # Market data quality
        if signal_inputs.market_data:
            quality_factors.append("Market data: Available and current")
            quality_score += 1
        else:
            quality_factors.append("Market data: Limited")
        total_factors += 1
        
        # Overall quality assessment
        overall_quality = quality_score / total_factors if total_factors > 0 else 0
        
        quality_assessment = f"Overall data quality: {overall_quality:.1%}. "
        quality_assessment += " | ".join(quality_factors)
        
        if overall_quality < 0.5:
            quality_assessment += " | WARNING: Low data quality may affect recommendation accuracy."
        
        return quality_assessment


class RecommendationAuditTrail:
    """Audit trail system for investment recommendations"""
    
    def __init__(self):
        self.audit_events = []
    
    async def log_recommendation_creation(
        self, 
        recommendation: InvestmentRecommendation,
        decision_factors: DecisionFactors,
        signal_inputs: SignalInput,
        user_id: Optional[str] = None
    ):
        """Log recommendation creation event"""
        
        audit_event = {
            'event_type': 'recommendation_created',
            'timestamp': datetime.now(),
            'recommendation_id': str(recommendation.id),
            'ticker': recommendation.ticker,
            'signal': recommendation.signal.value,
            'confidence': recommendation.confidence,
            'user_id': user_id,
            'decision_factors': asdict(decision_factors),
            'input_data_sources': self._extract_data_sources(signal_inputs),
            'model_version': recommendation.model_version
        }
        
        await self._store_audit_event(audit_event)
    
    async def log_recommendation_update(
        self, 
        recommendation_id: str,
        old_recommendation: InvestmentRecommendation,
        new_recommendation: InvestmentRecommendation,
        reason: str,
        user_id: Optional[str] = None
    ):
        """Log recommendation update event"""
        
        changes = self._detect_changes(old_recommendation, new_recommendation)
        
        audit_event = {
            'event_type': 'recommendation_updated',
            'timestamp': datetime.now(),
            'recommendation_id': recommendation_id,
            'ticker': new_recommendation.ticker,
            'reason': reason,
            'user_id': user_id,
            'changes': changes,
            'old_signal': old_recommendation.signal.value,
            'new_signal': new_recommendation.signal.value,
            'old_confidence': old_recommendation.confidence,
            'new_confidence': new_recommendation.confidence
        }
        
        await self._store_audit_event(audit_event)
    
    async def log_recommendation_performance(
        self, 
        recommendation_id: str,
        actual_return: float,
        holding_period: int,
        exit_reason: str,
        user_id: Optional[str] = None
    ):
        """Log recommendation performance tracking"""
        
        audit_event = {
            'event_type': 'recommendation_performance',
            'timestamp': datetime.now(),
            'recommendation_id': recommendation_id,
            'actual_return': actual_return,
            'holding_period_days': holding_period,
            'exit_reason': exit_reason,
            'user_id': user_id
        }
        
        await self._store_audit_event(audit_event)
    
    async def log_model_decision(
        self, 
        recommendation_id: str,
        decision_point: str,
        decision_logic: str,
        input_values: Dict[str, Any],
        output_value: Any
    ):
        """Log individual model decisions for transparency"""
        
        audit_event = {
            'event_type': 'model_decision',
            'timestamp': datetime.now(),
            'recommendation_id': recommendation_id,
            'decision_point': decision_point,
            'decision_logic': decision_logic,
            'input_values': input_values,
            'output_value': output_value
        }
        
        await self._store_audit_event(audit_event)
    
    def _extract_data_sources(self, signal_inputs: SignalInput) -> List[str]:
        """Extract data sources used in analysis"""
        
        sources = []
        
        if signal_inputs.document_insights:
            sources.append('document_analysis')
        
        if signal_inputs.sentiment_analysis:
            sources.append('sentiment_analysis')
        
        if signal_inputs.anomalies:
            sources.append('anomaly_detection')
        
        if signal_inputs.forecast:
            sources.append('forecasting_models')
        
        if signal_inputs.market_data:
            sources.append('market_data')
        
        return sources
    
    def _detect_changes(
        self, 
        old_rec: InvestmentRecommendation, 
        new_rec: InvestmentRecommendation
    ) -> Dict[str, Dict[str, Any]]:
        """Detect changes between recommendation versions"""
        
        changes = {}
        
        # Check signal change
        if old_rec.signal != new_rec.signal:
            changes['signal'] = {
                'old': old_rec.signal.value,
                'new': new_rec.signal.value
            }
        
        # Check confidence change
        if abs(old_rec.confidence - new_rec.confidence) > 0.05:
            changes['confidence'] = {
                'old': old_rec.confidence,
                'new': new_rec.confidence
            }
        
        # Check target price change
        if old_rec.target_price != new_rec.target_price:
            changes['target_price'] = {
                'old': old_rec.target_price,
                'new': new_rec.target_price
            }
        
        # Check position size change
        if old_rec.position_size != new_rec.position_size:
            changes['position_size'] = {
                'old': old_rec.position_size,
                'new': new_rec.position_size
            }
        
        # Check risk level change
        if old_rec.risk_level != new_rec.risk_level:
            changes['risk_level'] = {
                'old': old_rec.risk_level.value,
                'new': new_rec.risk_level.value
            }
        
        return changes
    
    async def _store_audit_event(self, audit_event: Dict[str, Any]):
        """Store audit event in database"""
        
        try:
            db = await get_database()
            
            query = """
                INSERT INTO recommendation_audit_trail (
                    event_type, timestamp, recommendation_id, ticker, user_id, event_data
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            # Convert datetime objects to strings for JSON serialization
            serializable_event = audit_event.copy()
            if 'timestamp' in serializable_event:
                serializable_event['timestamp'] = serializable_event['timestamp'].isoformat()
            
            await db.execute(
                query,
                audit_event['event_type'],
                audit_event['timestamp'],
                audit_event.get('recommendation_id'),
                audit_event.get('ticker'),
                audit_event.get('user_id'),
                json.dumps(serializable_event, default=str)
            )
            
            logger.info(f"Logged audit event: {audit_event['event_type']} for {audit_event.get('recommendation_id')}")
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
    
    async def get_recommendation_history(self, recommendation_id: str) -> List[Dict[str, Any]]:
        """Get complete audit history for a recommendation"""
        
        try:
            db = await get_database()
            
            query = """
                SELECT * FROM recommendation_audit_trail 
                WHERE recommendation_id = $1 
                ORDER BY timestamp ASC
            """
            
            rows = await db.fetch(query, recommendation_id)
            
            history = []
            for row in rows:
                event_data = json.loads(row['event_data'])
                history.append(event_data)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve recommendation history: {e}")
            return []


# Global service functions
async def explain_recommendation(
    recommendation: InvestmentRecommendation,
    decision_factors: DecisionFactors,
    signal_inputs: SignalInput,
    risk_assessment: Optional[Dict[str, Any]] = None
) -> RecommendationExplanation:
    """Generate comprehensive explanation for investment recommendation"""
    explainer = ExplainableAIEngine()
    return await explainer.explain_recommendation(
        recommendation, decision_factors, signal_inputs, risk_assessment
    )


async def get_recommendation_audit_trail(recommendation_id: str) -> List[Dict[str, Any]]:
    """Get audit trail for a recommendation"""
    audit_trail = RecommendationAuditTrail()
    return await audit_trail.get_recommendation_history(recommendation_id)


async def generate_feature_importance_visualization(
    feature_importance: Dict[str, float]
) -> Dict[str, Any]:
    """Generate data for feature importance visualization"""
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    visualization_data = {
        'chart_type': 'horizontal_bar',
        'title': 'Feature Importance in Investment Recommendation',
        'data': {
            'labels': [feature.replace('_', ' ').title() for feature, _ in sorted_features],
            'values': [importance for _, importance in sorted_features],
            'colors': ['#1f77b4' if importance > 0.1 else '#aec7e8' for _, importance in sorted_features]
        },
        'options': {
            'xlabel': 'Importance Score',
            'ylabel': 'Features',
            'show_values': True,
            'value_format': '.2%'
        }
    }
    
    return visualization_data


async def generate_decision_tree_explanation(
    recommendation: InvestmentRecommendation,
    decision_factors: DecisionFactors
) -> Dict[str, Any]:
    """Generate decision tree explanation for recommendation logic"""
    
    decision_tree = {
        'root': {
            'question': f'Overall Score: {decision_factors.overall_score:.2f}',
            'children': []
        }
    }
    
    # Add factor branches
    factors = [
        ('Fundamental Analysis', decision_factors.fundamental_score),
        ('Technical Analysis', decision_factors.technical_score),
        ('Sentiment Analysis', decision_factors.sentiment_score),
        ('Forecast Analysis', decision_factors.forecast_score)
    ]
    
    for factor_name, score in factors:
        if abs(score) > 0.1:  # Only include significant factors
            branch = {
                'question': f'{factor_name}: {score:.2f}',
                'decision': 'Positive' if score > 0 else 'Negative',
                'impact': 'High' if abs(score) > 0.5 else 'Medium' if abs(score) > 0.2 else 'Low'
            }
            decision_tree['root']['children'].append(branch)
    
    # Final decision
    decision_tree['final_decision'] = {
        'signal': recommendation.signal.value,
        'confidence': recommendation.confidence,
        'reasoning': f'Based on weighted combination of factors, recommendation is {recommendation.signal.value}'
    }
    
    return decision_tree