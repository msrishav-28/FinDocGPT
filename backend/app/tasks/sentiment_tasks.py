"""
Sentiment analysis tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List
from celery import current_task
from ..celery_app import celery_app
from ..services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.sentiment_tasks.analyze_document_sentiment")
def analyze_document_sentiment(self, document_id: str) -> Dict[str, Any]:
    """
    Analyze sentiment for a document asynchronously
    
    Args:
        document_id: Document identifier
        
    Returns:
        Dictionary with sentiment analysis results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Starting sentiment analysis"})
        
        sentiment_service = SentimentService()
        
        # Perform sentiment analysis
        self.update_state(state="PROCESSING", meta={"status": "Analyzing overall sentiment"})
        sentiment_result = sentiment_service.analyze_document_sentiment(document_id)
        
        # Analyze topic-specific sentiment
        self.update_state(state="PROCESSING", meta={"status": "Analyzing topic sentiment"})
        topic_sentiment = sentiment_service.analyze_topic_sentiment(document_id)
        
        # Generate sentiment trends
        self.update_state(state="PROCESSING", meta={"status": "Generating sentiment trends"})
        trends = sentiment_service.get_sentiment_trends(document_id)
        
        return {
            "document_id": document_id,
            "overall_sentiment": sentiment_result,
            "topic_sentiment": topic_sentiment,
            "trends": trends,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Sentiment analysis failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.sentiment_tasks.batch_sentiment_analysis")
def batch_sentiment_analysis(self, document_ids: List[str]) -> Dict[str, Any]:
    """
    Perform sentiment analysis on multiple documents
    
    Args:
        document_ids: List of document identifiers
        
    Returns:
        Dictionary with batch sentiment analysis results
    """
    try:
        total_docs = len(document_ids)
        results = []
        failed_analyses = []
        
        for i, doc_id in enumerate(document_ids):
            try:
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Analyzing sentiment for document {i+1}/{total_docs}",
                        "progress": (i / total_docs) * 100
                    }
                )
                
                # Analyze individual document sentiment
                result = analyze_document_sentiment.apply_async(args=[doc_id]).get()
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze sentiment for document {doc_id}: {str(e)}")
                failed_analyses.append({"document_id": doc_id, "error": str(e)})
        
        return {
            "total_documents": total_docs,
            "successful_analyses": len(results),
            "failed_analyses": len(failed_analyses),
            "results": results,
            "failed_docs": failed_analyses,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Batch sentiment analysis failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.sentiment_tasks.update_sentiment_trends")
def update_sentiment_trends(self, company: str) -> Dict[str, Any]:
    """
    Update sentiment trends for a company
    
    Args:
        company: Company ticker symbol
        
    Returns:
        Dictionary with trend update results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Updating sentiment trends"})
        
        sentiment_service = SentimentService()
        
        # Update trends for company
        updated_trends = sentiment_service.update_company_sentiment_trends(company)
        
        return {
            "company": company,
            "updated_trends": updated_trends,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Sentiment trends update failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Sentiment trends update failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.sentiment_tasks.compare_company_sentiment")
def compare_company_sentiment(self, companies: List[str], timeframe: str) -> Dict[str, Any]:
    """
    Compare sentiment across multiple companies
    
    Args:
        companies: List of company ticker symbols
        timeframe: Time period for comparison
        
    Returns:
        Dictionary with sentiment comparison results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Comparing company sentiment"})
        
        sentiment_service = SentimentService()
        
        # Perform sentiment comparison
        comparison_results = sentiment_service.compare_companies_sentiment(companies, timeframe)
        
        return {
            "companies": companies,
            "timeframe": timeframe,
            "comparison_results": comparison_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Company sentiment comparison failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Sentiment comparison failed"}
        )
        raise