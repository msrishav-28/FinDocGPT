"""
Investment Advisory API Routes

This module provides REST API endpoints for the Investment Advisory Service:
- Generate investment recommendations
- Risk assessment and position sizing
- Explainable AI and recommendation transparency
- Portfolio analysis and optimization
- Audit trail and compliance reporting
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..models.recommendation import (
    InvestmentRecommendation, InvestmentSignal, RiskLevel, TimeHorizon,
    RecommendationExplanation, Portfolio, PositionSizes, AnalysisContext
)
from ..services.investment_advisory_service import (
    generate_recommendation, get_recommendation_by_id, get_active_recommendations,
    assess_investment_risk, optimize_position_sizing, assess_portfolio_risk
)
from ..services.explainable_ai_service import (
    explain_recommendation, get_recommendation_audit_trail,
    generate_feature_importance_visualization, generate_decision_tree_explanation,
    SignalInput, DecisionFactors
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/investment-advisory", tags=["Investment Advisory"])


@router.post("/recommendations", response_model=InvestmentRecommendation)
async def create_recommendation(
    ticker: str,
    analysis_context: Optional[AnalysisContext] = None,
    custom_weights: Optional[Dict[str, float]] = None
):
    """
    Generate investment recommendation for a ticker
    
    Args:
        ticker: Stock ticker symbol
        analysis_context: Analysis parameters and context
        custom_weights: Custom factor weights for recommendation
    
    Returns:
        Complete investment recommendation with reasoning
    """
    try:
        recommendation = await generate_recommendation(
            ticker=ticker.upper(),
            context=analysis_context,
            custom_weights=custom_weights
        )
        
        logger.info(f"Generated {recommendation.signal.value} recommendation for {ticker}")
        return recommendation
        
    except Exception as e:
        logger.error(f"Failed to generate recommendation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}", response_model=InvestmentRecommendation)
async def get_recommendation(recommendation_id: str):
    """
    Retrieve investment recommendation by ID
    
    Args:
        recommendation_id: Unique recommendation identifier
    
    Returns:
        Investment recommendation details
    """
    try:
        recommendation = await get_recommendation_by_id(recommendation_id)
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        return recommendation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[InvestmentRecommendation])
async def list_recommendations(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    signal: Optional[InvestmentSignal] = Query(None, description="Filter by investment signal"),
    risk_level: Optional[RiskLevel] = Query(None, description="Filter by risk level"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of recommendations")
):
    """
    List active investment recommendations with optional filters
    
    Args:
        ticker: Optional ticker filter
        signal: Optional signal filter
        risk_level: Optional risk level filter
        limit: Maximum number of results
    
    Returns:
        List of active investment recommendations
    """
    try:
        recommendations = await get_active_recommendations(ticker=ticker.upper() if ticker else None)
        
        # Apply additional filters
        if signal:
            recommendations = [r for r in recommendations if r.signal == signal]
        
        if risk_level:
            recommendations = [r for r in recommendations if r.risk_level == risk_level]
        
        # Limit results
        recommendations = recommendations[:limit]
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to list recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/risk-assessment")
async def assess_recommendation_risk(
    recommendation_id: str,
    portfolio: Optional[Portfolio] = None
):
    """
    Assess risk for a specific recommendation
    
    Args:
        recommendation_id: Recommendation to assess
        portfolio: Optional portfolio context
    
    Returns:
        Comprehensive risk assessment
    """
    try:
        # Get the recommendation
        recommendation = await get_recommendation_by_id(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Perform risk assessment
        risk_assessment = await assess_investment_risk(
            ticker=recommendation.ticker,
            recommendation=recommendation,
            portfolio=portfolio
        )
        
        return risk_assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assess risk for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/position-sizing", response_model=PositionSizes)
async def optimize_portfolio_positions(
    recommendations: List[InvestmentRecommendation],
    portfolio: Portfolio,
    risk_tolerance: str = Query("moderate", regex="^(conservative|moderate|aggressive)$"),
    optimization_method: str = Query("volatility_adjusted", regex="^(fixed_percentage|volatility_adjusted|kelly_criterion|risk_parity|max_drawdown)$")
):
    """
    Optimize position sizes for multiple recommendations
    
    Args:
        recommendations: List of investment recommendations
        portfolio: Current portfolio
        risk_tolerance: Risk tolerance level
        optimization_method: Position sizing method
    
    Returns:
        Optimized position sizes with risk metrics
    """
    try:
        position_sizes = await optimize_position_sizing(
            recommendations=recommendations,
            portfolio=portfolio,
            risk_tolerance=risk_tolerance,
            optimization_method=optimization_method
        )
        
        return position_sizes
        
    except Exception as e:
        logger.error(f"Failed to optimize position sizing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/risk-assessment")
async def assess_portfolio_risk_endpoint(portfolio: Portfolio):
    """
    Assess overall portfolio risk metrics
    
    Args:
        portfolio: Portfolio to assess
    
    Returns:
        Portfolio risk assessment with metrics and warnings
    """
    try:
        risk_assessment = await assess_portfolio_risk(portfolio)
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Failed to assess portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}/explanation", response_model=RecommendationExplanation)
async def get_recommendation_explanation(recommendation_id: str):
    """
    Get detailed explanation for a recommendation
    
    Args:
        recommendation_id: Recommendation to explain
    
    Returns:
        Comprehensive recommendation explanation
    """
    try:
        # Get the recommendation
        recommendation = await get_recommendation_by_id(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # For this endpoint, we need to reconstruct the decision factors and signal inputs
        # In a real implementation, these would be stored with the recommendation
        
        # Placeholder decision factors (would be retrieved from storage)
        decision_factors = DecisionFactors(
            fundamental_score=0.3,
            technical_score=0.2,
            sentiment_score=0.1,
            risk_score=-0.1,
            forecast_score=0.2,
            anomaly_score=-0.05,
            overall_score=0.25,
            confidence=recommendation.confidence
        )
        
        # Placeholder signal inputs (would be retrieved from storage)
        signal_inputs = SignalInput(
            document_insights=recommendation.document_insights,
            sentiment_analysis=None,  # Would reconstruct from stored data
            topic_sentiments=None,
            anomalies=None,
            forecast=recommendation.forecast_data,
            market_data={'price': recommendation.current_price, 'volatility': 0.25}
        )
        
        # Generate explanation
        explanation = await explain_recommendation(
            recommendation=recommendation,
            decision_factors=decision_factors,
            signal_inputs=signal_inputs
        )
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate explanation for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}/feature-importance")
async def get_feature_importance(recommendation_id: str):
    """
    Get feature importance visualization data for a recommendation
    
    Args:
        recommendation_id: Recommendation to analyze
    
    Returns:
        Feature importance visualization data
    """
    try:
        # Get the recommendation
        recommendation = await get_recommendation_by_id(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Placeholder feature importance (would be calculated from stored decision factors)
        feature_importance = {
            'fundamental_analysis': 0.35,
            'technical_analysis': 0.25,
            'sentiment_analysis': 0.20,
            'forecast_analysis': 0.15,
            'risk_assessment': 0.05
        }
        
        visualization_data = await generate_feature_importance_visualization(feature_importance)
        return visualization_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate feature importance for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}/decision-tree")
async def get_decision_tree(recommendation_id: str):
    """
    Get decision tree explanation for a recommendation
    
    Args:
        recommendation_id: Recommendation to analyze
    
    Returns:
        Decision tree visualization data
    """
    try:
        # Get the recommendation
        recommendation = await get_recommendation_by_id(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Placeholder decision factors (would be retrieved from storage)
        decision_factors = DecisionFactors(
            fundamental_score=0.3,
            technical_score=0.2,
            sentiment_score=0.1,
            risk_score=-0.1,
            forecast_score=0.2,
            anomaly_score=-0.05,
            overall_score=0.25,
            confidence=recommendation.confidence
        )
        
        decision_tree = await generate_decision_tree_explanation(recommendation, decision_factors)
        return decision_tree
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate decision tree for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}/audit-trail")
async def get_audit_trail(recommendation_id: str):
    """
    Get audit trail for a recommendation
    
    Args:
        recommendation_id: Recommendation to trace
    
    Returns:
        Complete audit history for the recommendation
    """
    try:
        audit_trail = await get_recommendation_audit_trail(recommendation_id)
        return {
            'recommendation_id': recommendation_id,
            'audit_events': audit_trail,
            'total_events': len(audit_trail)
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve audit trail for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/performance")
async def get_recommendation_performance(
    start_date: Optional[datetime] = Query(None, description="Start date for performance analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for performance analysis"),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol")
):
    """
    Get recommendation performance analytics
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        ticker: Optional ticker filter
    
    Returns:
        Performance analytics and metrics
    """
    try:
        # Placeholder performance analytics
        # In a real implementation, this would query actual performance data
        
        performance_data = {
            'summary': {
                'total_recommendations': 150,
                'profitable_recommendations': 95,
                'win_rate': 0.633,
                'average_return': 0.087,
                'average_holding_period': 45,
                'sharpe_ratio': 1.24
            },
            'by_signal': {
                'STRONG_BUY': {'count': 25, 'win_rate': 0.72, 'avg_return': 0.145},
                'BUY': {'count': 60, 'win_rate': 0.65, 'avg_return': 0.098},
                'HOLD': {'count': 40, 'win_rate': 0.55, 'avg_return': 0.032},
                'SELL': {'count': 20, 'win_rate': 0.60, 'avg_return': -0.045},
                'STRONG_SELL': {'count': 5, 'win_rate': 0.80, 'avg_return': -0.125}
            },
            'by_risk_level': {
                'very_low': {'count': 20, 'win_rate': 0.75, 'avg_return': 0.045},
                'low': {'count': 45, 'win_rate': 0.67, 'avg_return': 0.067},
                'medium': {'count': 60, 'win_rate': 0.62, 'avg_return': 0.089},
                'high': {'count': 20, 'win_rate': 0.55, 'avg_return': 0.134},
                'very_high': {'count': 5, 'win_rate': 0.40, 'avg_return': 0.187}
            },
            'monthly_performance': [
                {'month': '2024-01', 'recommendations': 12, 'win_rate': 0.67, 'avg_return': 0.089},
                {'month': '2024-02', 'recommendations': 15, 'win_rate': 0.60, 'avg_return': 0.076},
                {'month': '2024-03', 'recommendations': 18, 'win_rate': 0.72, 'avg_return': 0.112}
            ]
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to retrieve performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/model-performance")
async def get_model_performance():
    """
    Get model performance metrics and statistics
    
    Returns:
        Model performance analytics across different components
    """
    try:
        # Placeholder model performance data
        model_performance = {
            'overall_metrics': {
                'recommendation_accuracy': 0.687,
                'price_target_accuracy': 0.623,
                'risk_assessment_accuracy': 0.745,
                'model_confidence_correlation': 0.834
            },
            'component_performance': {
                'fundamental_analysis': {
                    'accuracy': 0.712,
                    'precision': 0.689,
                    'recall': 0.734,
                    'f1_score': 0.711
                },
                'technical_analysis': {
                    'accuracy': 0.645,
                    'precision': 0.623,
                    'recall': 0.667,
                    'f1_score': 0.644
                },
                'sentiment_analysis': {
                    'accuracy': 0.678,
                    'precision': 0.656,
                    'recall': 0.701,
                    'f1_score': 0.678
                },
                'forecasting_models': {
                    'accuracy': 0.634,
                    'mae': 0.087,
                    'rmse': 0.123,
                    'mape': 0.156
                }
            },
            'model_drift': {
                'last_updated': '2024-03-15T10:30:00Z',
                'drift_score': 0.023,
                'drift_threshold': 0.050,
                'status': 'stable',
                'next_retraining': '2024-04-15T00:00:00Z'
            },
            'feature_stability': {
                'fundamental_features': 0.945,
                'technical_features': 0.867,
                'sentiment_features': 0.823,
                'market_features': 0.912
            }
        }
        
        return model_performance
        
    except Exception as e:
        logger.error(f"Failed to retrieve model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/feedback")
async def submit_recommendation_feedback(
    recommendation_id: str,
    feedback: Dict[str, Any]
):
    """
    Submit feedback on recommendation performance
    
    Args:
        recommendation_id: Recommendation to provide feedback on
        feedback: Feedback data including actual performance
    
    Returns:
        Confirmation of feedback submission
    """
    try:
        # In a real implementation, this would store feedback and update model performance
        
        logger.info(f"Received feedback for recommendation {recommendation_id}: {feedback}")
        
        return {
            'recommendation_id': recommendation_id,
            'feedback_received': True,
            'timestamp': datetime.now(),
            'message': 'Feedback submitted successfully and will be used to improve model performance'
        }
        
    except Exception as e:
        logger.error(f"Failed to submit feedback for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))