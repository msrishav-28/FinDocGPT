"""
Sentiment Analysis API Routes

This module provides REST API endpoints for the Multi-Dimensional Sentiment Analysis Service.
"""

from fastapi import APIRouter, HTTPException, Form, Query
from typing import List, Optional
from pydantic import BaseModel

from ..models.sentiment import (
    SentimentAnalysis, TopicSentiment, SentimentTrends, SentimentComparison,
    SentimentAlert, SentimentTopic
)
from ..services.sentiment_service import (
    analyze_document_sentiment, analyze_text_sentiment, analyze_topic_sentiment,
    get_sentiment_trends, compare_company_sentiment, detect_sentiment_deviation,
    generate_sentiment_explanation, store_sentiment_analysis
)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class TextSentimentRequest(BaseModel):
    """Request model for text sentiment analysis"""
    text: str
    topics: Optional[List[SentimentTopic]] = None


class CompanySentimentRequest(BaseModel):
    """Request model for company sentiment comparison"""
    companies: List[str]
    topic: Optional[SentimentTopic] = None
    days_back: int = 30


@router.post("/analyze/text", response_model=SentimentAnalysis)
async def analyze_text_sentiment_endpoint(request: TextSentimentRequest):
    """
    Analyze sentiment of arbitrary text using ensemble models
    
    - **text**: Text to analyze for sentiment
    - **topics**: Optional list of specific topics to analyze
    """
    try:
        # Analyze overall sentiment
        sentiment_result = await analyze_text_sentiment(request.text)
        
        # Analyze topic-specific sentiment if requested
        if request.topics:
            topic_results = await analyze_topic_sentiment(request.text, request.topics)
            sentiment_result.topic_sentiments = {
                topic: result.sentiment_score 
                for topic, result in topic_results.items()
            }
        
        return sentiment_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.get("/analyze/document/{doc_id}", response_model=SentimentAnalysis)
async def analyze_document_sentiment_endpoint(doc_id: str):
    """
    Analyze sentiment of a specific document
    
    - **doc_id**: Document identifier to analyze
    """
    try:
        return await analyze_document_sentiment(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document sentiment analysis failed: {str(e)}")


@router.post("/analyze/topics", response_model=dict)
async def analyze_topic_sentiment_endpoint(request: TextSentimentRequest):
    """
    Analyze sentiment for specific financial topics
    
    - **text**: Text to analyze
    - **topics**: List of topics to analyze (if empty, analyzes all topics)
    """
    try:
        topics = request.topics or list(SentimentTopic)
        topic_results = await analyze_topic_sentiment(request.text, topics)
        
        return {
            "text_snippet": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "topic_sentiments": {
                topic: {
                    "sentiment_score": result.sentiment_score,
                    "confidence": result.confidence,
                    "supporting_phrases": result.supporting_phrases,
                    "context": result.context
                }
                for topic, result in topic_results.items()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic sentiment analysis failed: {str(e)}")


@router.get("/trends/{company}", response_model=SentimentTrends)
async def get_sentiment_trends_endpoint(
    company: str,
    time_period: str = Query("6M", regex="^(1M|3M|6M|1Y)$"),
    topic: Optional[SentimentTopic] = None
):
    """
    Get historical sentiment trends for a company
    
    - **company**: Company ticker or name
    - **time_period**: Time period (1M, 3M, 6M, 1Y)
    - **topic**: Optional specific topic to analyze
    """
    try:
        return await get_sentiment_trends(company, time_period, topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment trends: {str(e)}")


@router.post("/compare", response_model=SentimentComparison)
async def compare_company_sentiment_endpoint(request: CompanySentimentRequest):
    """
    Compare sentiment across multiple companies
    
    - **companies**: List of company tickers/names to compare
    - **topic**: Optional specific topic to compare
    - **days_back**: Number of days to look back for comparison (default: 30)
    """
    try:
        if len(request.companies) < 2:
            raise HTTPException(status_code=400, detail="At least 2 companies required for comparison")
        
        return await compare_company_sentiment(request.companies, request.topic, request.days_back)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Company sentiment comparison failed: {str(e)}")


@router.get("/alerts/{company}", response_model=Optional[SentimentAlert])
async def detect_sentiment_deviation_endpoint(
    company: str,
    threshold: float = Query(0.3, ge=0.1, le=1.0),
    lookback_days: int = Query(90, ge=30, le=365),
    recent_days: int = Query(7, ge=1, le=30)
):
    """
    Detect significant sentiment deviations for a company
    
    - **company**: Company ticker or name
    - **threshold**: Minimum change magnitude to trigger alert (0.1-1.0)
    - **lookback_days**: Days to look back for historical baseline (30-365)
    - **recent_days**: Recent period to compare against baseline (1-30)
    """
    try:
        return await detect_sentiment_deviation(company, threshold, lookback_days, recent_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment deviation detection failed: {str(e)}")


@router.post("/explain")
async def explain_sentiment_endpoint(
    sentiment_analysis: SentimentAnalysis,
    topic_sentiments: Optional[dict] = None
):
    """
    Generate detailed explanation for sentiment analysis results
    
    - **sentiment_analysis**: Sentiment analysis results to explain
    - **topic_sentiments**: Optional topic-specific sentiment results
    """
    try:
        # Convert dict to TopicSentiment objects if provided
        topic_sentiment_objects = None
        if topic_sentiments:
            topic_sentiment_objects = {}
            for topic_name, topic_data in topic_sentiments.items():
                if isinstance(topic_data, dict):
                    topic_sentiment_objects[topic_name] = TopicSentiment(
                        topic=SentimentTopic(topic_name),
                        sentiment_score=topic_data.get('sentiment_score', 0.0),
                        confidence=topic_data.get('confidence', 0.0),
                        supporting_phrases=topic_data.get('supporting_phrases', []),
                        context=topic_data.get('context', '')
                    )
        
        explanation = await generate_sentiment_explanation(sentiment_analysis, topic_sentiment_objects)
        
        return {
            "explanation": explanation,
            "sentiment_summary": {
                "overall_sentiment": sentiment_analysis.overall_sentiment,
                "polarity": sentiment_analysis.polarity.value,
                "confidence": sentiment_analysis.confidence,
                "model_used": sentiment_analysis.model_used
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment explanation generation failed: {str(e)}")


@router.get("/topics", response_model=List[str])
async def get_available_topics():
    """
    Get list of available sentiment analysis topics
    """
    return [topic.value for topic in SentimentTopic]


@router.get("/health")
async def sentiment_service_health():
    """
    Check sentiment analysis service health
    """
    try:
        from ..services.sentiment_service import get_sentiment_ensemble
        
        ensemble = await get_sentiment_ensemble()
        
        return {
            "status": "healthy",
            "models_initialized": ensemble.is_initialized,
            "available_models": list(ensemble.models.keys()) if ensemble.models else [],
            "model_weights": ensemble.model_weights,
            "topics_available": len(SentimentTopic),
            "service": "Multi-Dimensional Sentiment Analysis"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "Multi-Dimensional Sentiment Analysis"
        }


# Legacy endpoint for backward compatibility
@router.get("/legacy/{doc_id}")
async def legacy_sentiment_endpoint(doc_id: str):
    """
    Legacy sentiment analysis endpoint (backward compatibility)
    Returns simple sentiment score for a document
    """
    try:
        from ..services.sentiment_service import analyze_sentiment
        score = await analyze_sentiment(doc_id)
        return {"sentiment_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legacy sentiment analysis failed: {str(e)}")