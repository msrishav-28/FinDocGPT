"""
Multi-Dimensional Sentiment Analysis Service

This service implements ensemble sentiment analysis using multiple models:
- FinBERT: Financial domain-specific BERT model
- RoBERTa: Robust optimized BERT approach
- Custom financial sentiment models

Features:
- Ensemble model combination with dynamic weighting
- Confidence scoring and uncertainty quantification
- Topic-specific sentiment extraction
- Historical sentiment tracking and trend analysis
"""

import os
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

from ..models.sentiment import (
    SentimentAnalysis, TopicSentiment, SentimentTrend, SentimentTrends,
    SentimentComparison, SentimentAlert, SentimentModelPerformance,
    SentimentPolarity, SentimentTopic
)
from ..config import get_settings
from .doc_store import get_doc

logger = logging.getLogger(__name__)
settings = get_settings()


class SentimentEnsemble:
    """Ensemble sentiment analysis with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.is_initialized = False
        
        # Financial topic keywords for aspect-based analysis
        self.topic_keywords = {
            SentimentTopic.MANAGEMENT_OUTLOOK: [
                "management", "outlook", "guidance", "forecast", "expectations", 
                "leadership", "strategy", "vision", "confident", "optimistic"
            ],
            SentimentTopic.FINANCIAL_PERFORMANCE: [
                "revenue", "earnings", "profit", "margin", "cash flow", "ebitda",
                "growth", "performance", "results", "financial", "income"
            ],
            SentimentTopic.COMPETITIVE_POSITION: [
                "competition", "competitive", "market share", "advantage", 
                "differentiation", "positioning", "rivals", "competitors"
            ],
            SentimentTopic.MARKET_CONDITIONS: [
                "market", "economy", "economic", "demand", "supply", "conditions",
                "environment", "trends", "macro", "industry"
            ],
            SentimentTopic.REGULATORY_ENVIRONMENT: [
                "regulation", "regulatory", "compliance", "policy", "government",
                "legal", "rules", "requirements", "oversight"
            ],
            SentimentTopic.OPERATIONAL_EFFICIENCY: [
                "operations", "efficiency", "productivity", "costs", "expenses",
                "optimization", "streamline", "process", "automation"
            ],
            SentimentTopic.GROWTH_PROSPECTS: [
                "growth", "expansion", "opportunities", "potential", "prospects",
                "future", "pipeline", "development", "innovation"
            ],
            SentimentTopic.RISK_FACTORS: [
                "risk", "risks", "uncertainty", "challenges", "concerns", 
                "threats", "volatility", "exposure", "headwinds"
            ]
        }
    
    async def initialize(self):
        """Initialize all sentiment models"""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing sentiment analysis ensemble...")
            
            # Check if model downloads are allowed
            allow_download = os.environ.get("ALLOW_MODEL_DOWNLOAD", "0") in ("1", "true", "True")
            
            if not allow_download:
                logger.warning("Model downloads disabled. Using fallback sentiment analysis.")
                self.is_initialized = True
                return
            
            # Initialize FinBERT
            await self._initialize_finbert()
            
            # Initialize RoBERTa
            await self._initialize_roberta()
            
            # Initialize custom financial model (fallback to DistilBERT)
            await self._initialize_custom_model()
            
            # Set initial model weights based on expected performance
            self._initialize_model_weights()
            
            self.is_initialized = True
            logger.info("Sentiment analysis ensemble initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment ensemble: {e}")
            self.is_initialized = True  # Allow fallback operation
    
    async def _initialize_finbert(self):
        """Initialize FinBERT model"""
        try:
            self.models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
    
    async def _initialize_roberta(self):
        """Initialize RoBERTa model"""
        try:
            # Using a financial-tuned RoBERTa model or general RoBERTa
            self.models['roberta'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("RoBERTa model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load RoBERTa: {e}")
    
    async def _initialize_custom_model(self):
        """Initialize custom financial sentiment model (fallback to DistilBERT)"""
        try:
            self.models['custom'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Custom sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load custom model: {e}")
    
    def _initialize_model_weights(self):
        """Initialize model weights based on expected performance"""
        available_models = list(self.models.keys())
        
        if not available_models:
            return
        
        # Default weights favoring financial-specific models
        default_weights = {
            'finbert': 0.5,    # Highest weight for financial domain
            'roberta': 0.3,    # Good general performance
            'custom': 0.2      # Fallback model
        }
        
        # Normalize weights for available models only
        total_weight = sum(default_weights.get(model, 0) for model in available_models)
        
        for model in available_models:
            self.model_weights[model] = default_weights.get(model, 1.0) / total_weight
        
        logger.info(f"Model weights initialized: {self.model_weights}")
    
    def _normalize_sentiment_score(self, result: Dict, model_name: str) -> Tuple[float, float]:
        """Normalize sentiment scores from different models to [-1, 1] range"""
        label = result.get('label', '').lower()
        score = result.get('score', 0.0)
        
        if model_name == 'finbert':
            # FinBERT: positive, negative, neutral
            if 'positive' in label:
                return score, score
            elif 'negative' in label:
                return -score, score
            else:  # neutral
                return 0.0, score
        
        elif model_name == 'roberta':
            # RoBERTa: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
            if 'label_2' in label or 'positive' in label:
                return score, score
            elif 'label_0' in label or 'negative' in label:
                return -score, score
            else:  # neutral
                return 0.0, score
        
        else:  # custom/distilbert
            # DistilBERT: POSITIVE, NEGATIVE
            if 'positive' in label:
                return score, score
            elif 'negative' in label:
                return -score, score
            else:
                return 0.0, score
    
    async def analyze_text(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment using ensemble of models"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        # Fallback to keyword-based analysis if no models available
        if not self.models:
            return await self._fallback_analysis(text, start_time)
        
        # Truncate text to manageable length
        max_length = 512
        text_truncated = text[:max_length] if len(text) > max_length else text
        
        model_results = {}
        model_confidences = {}
        
        # Run analysis with each available model
        for model_name, model in self.models.items():
            try:
                result = model(text_truncated)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                sentiment_score, confidence = self._normalize_sentiment_score(result, model_name)
                model_results[model_name] = sentiment_score
                model_confidences[model_name] = confidence
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        # If no models succeeded, use fallback
        if not model_results:
            return await self._fallback_analysis(text, start_time)
        
        # Calculate ensemble prediction with dynamic weighting
        ensemble_sentiment, ensemble_confidence = self._calculate_ensemble_prediction(
            model_results, model_confidences
        )
        
        # Determine polarity
        polarity = self._determine_polarity(ensemble_sentiment)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentAnalysis(
            text_snippet=text_truncated,
            overall_sentiment=ensemble_sentiment,
            confidence=ensemble_confidence,
            polarity=polarity,
            sentiment_explanation=self._generate_explanation(
                ensemble_sentiment, model_results, text_truncated
            ),
            model_used=f"ensemble({','.join(model_results.keys())})",
            processing_time=processing_time
        )
    
    def _calculate_ensemble_prediction(
        self, 
        model_results: Dict[str, float], 
        model_confidences: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate ensemble prediction with confidence-based weighting"""
        
        if not model_results:
            return 0.0, 0.0
        
        # Dynamic weighting based on confidence and model weights
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for model_name, sentiment in model_results.items():
            model_weight = self.model_weights.get(model_name, 1.0)
            confidence = model_confidences.get(model_name, 0.5)
            
            # Combine model weight with confidence
            dynamic_weight = model_weight * confidence
            
            weighted_sum += sentiment * dynamic_weight
            total_weight += dynamic_weight
            confidence_scores.append(confidence)
        
        # Calculate ensemble sentiment
        ensemble_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate ensemble confidence (weighted average with agreement bonus)
        base_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Agreement bonus: higher confidence when models agree
        sentiment_values = list(model_results.values())
        if len(sentiment_values) > 1:
            agreement = 1.0 - (np.std(sentiment_values) / 2.0)  # Normalize std to [0,1]
            ensemble_confidence = min(base_confidence * (1.0 + 0.2 * agreement), 1.0)
        else:
            ensemble_confidence = base_confidence
        
        return float(np.clip(ensemble_sentiment, -1.0, 1.0)), float(np.clip(ensemble_confidence, 0.0, 1.0))
    
    def _determine_polarity(self, sentiment_score: float) -> SentimentPolarity:
        """Determine sentiment polarity category"""
        if sentiment_score >= 0.6:
            return SentimentPolarity.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            return SentimentPolarity.POSITIVE
        elif sentiment_score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif sentiment_score <= -0.2:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL
    
    def _generate_explanation(
        self, 
        sentiment_score: float, 
        model_results: Dict[str, float], 
        text: str
    ) -> str:
        """Generate explanation for sentiment analysis"""
        polarity = self._determine_polarity(sentiment_score)
        
        explanation = f"Overall sentiment is {polarity.value} (score: {sentiment_score:.3f}). "
        
        if len(model_results) > 1:
            model_agreements = []
            for model, score in model_results.items():
                model_polarity = self._determine_polarity(score)
                model_agreements.append(f"{model}: {model_polarity.value} ({score:.3f})")
            
            explanation += f"Model consensus: {', '.join(model_agreements)}. "
        
        # Add key phrases that influenced the sentiment
        key_phrases = self._extract_key_phrases(text, sentiment_score)
        if key_phrases:
            explanation += f"Key phrases: {', '.join(key_phrases[:3])}."
        
        return explanation
    
    def _extract_key_phrases(self, text: str, sentiment_score: float) -> List[str]:
        """Extract key phrases that likely influenced sentiment"""
        # Simple keyword-based extraction
        positive_keywords = [
            "optimistic", "growth", "improved", "increase", "strong", "record", 
            "beat", "positive", "confident", "successful", "excellent", "outstanding"
        ]
        negative_keywords = [
            "concern", "decline", "decrease", "weak", "loss", "negative", 
            "pressure", "risk", "challenging", "difficult", "disappointing"
        ]
        
        text_lower = text.lower()
        found_phrases = []
        
        keywords = positive_keywords if sentiment_score > 0 else negative_keywords
        
        for keyword in keywords:
            if keyword in text_lower:
                # Find the phrase containing the keyword
                pattern = rf'\b\w*{keyword}\w*\b[^.!?]*'
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found_phrases.extend(matches[:2])  # Limit to 2 matches per keyword
        
        return found_phrases[:5]  # Return top 5 phrases
    
    async def analyze_topic_sentiment(self, text: str, topic: SentimentTopic) -> TopicSentiment:
        """Analyze sentiment for a specific financial topic"""
        # Extract topic-relevant contexts
        contexts = await self._extract_topic_contexts(text, topic)
        
        if not contexts:
            # If no topic-specific context found, return neutral sentiment
            return TopicSentiment(
                topic=topic,
                sentiment_score=0.0,
                confidence=0.1,
                supporting_phrases=[],
                context="No relevant context found for this topic"
            )
        
        # Analyze sentiment of topic-specific contexts
        topic_sentiments = []
        topic_confidences = []
        supporting_phrases = []
        
        for context in contexts:
            try:
                context_analysis = await self.analyze_text(context)
                topic_sentiments.append(context_analysis.overall_sentiment)
                topic_confidences.append(context_analysis.confidence)
                
                # Extract key phrases from this context
                key_phrases = self._extract_key_phrases(context, context_analysis.overall_sentiment)
                supporting_phrases.extend(key_phrases[:2])
                
            except Exception as e:
                logger.warning(f"Failed to analyze topic context: {e}")
                continue
        
        if not topic_sentiments:
            return TopicSentiment(
                topic=topic,
                sentiment_score=0.0,
                confidence=0.1,
                supporting_phrases=[],
                context="Failed to analyze topic contexts"
            )
        
        # Calculate weighted average sentiment for the topic
        weights = np.array(topic_confidences)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            topic_sentiment = np.average(topic_sentiments, weights=weights)
            topic_confidence = np.mean(topic_confidences)
        else:
            topic_sentiment = np.mean(topic_sentiments)
            topic_confidence = 0.3
        
        return TopicSentiment(
            topic=topic,
            sentiment_score=float(np.clip(topic_sentiment, -1.0, 1.0)),
            confidence=float(np.clip(topic_confidence, 0.0, 1.0)),
            supporting_phrases=list(set(supporting_phrases[:5])),  # Remove duplicates, limit to 5
            context=f"Analyzed {len(contexts)} relevant text segments"
        )
    
    async def _extract_topic_contexts(self, text: str, topic: SentimentTopic, window_size: int = 150) -> List[str]:
        """Extract text contexts relevant to a specific topic"""
        keywords = self.topic_keywords.get(topic, [])
        
        if not keywords:
            return []
        
        contexts = []
        text_lower = text.lower()
        sentences = self._split_into_sentences(text)
        
        # Find sentences containing topic keywords
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword in sentence_lower:
                    # Include surrounding context
                    sentence_idx = sentences.index(sentence)
                    start_idx = max(0, sentence_idx - 1)
                    end_idx = min(len(sentences), sentence_idx + 2)
                    
                    context = " ".join(sentences[start_idx:end_idx]).strip()
                    if context and len(context) > 20 and context not in contexts:
                        contexts.append(context)
                    break
        
        return contexts[:10]  # Limit to top 10 contexts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        # Simple sentence splitting - could be enhanced with NLTK or spaCy
        import re
        
        # Split on sentence endings, but be careful with abbreviations and numbers
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    async def _fallback_analysis(self, text: str, start_time: datetime) -> SentimentAnalysis:
        """Fallback keyword-based sentiment analysis"""
        pos_words = [
            "optimistic", "growth", "improved", "increase", "strong", "record", 
            "beat", "positive", "confident", "successful", "excellent", "outstanding",
            "revenue growth", "profit", "margin expansion", "cash flow"
        ]
        neg_words = [
            "concern", "decline", "decrease", "weak", "loss", "negative", 
            "pressure", "risk", "challenging", "difficult", "disappointing",
            "headwinds", "uncertainty", "volatility", "costs"
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in pos_words if word in text_lower)
        neg_count = sum(1 for word in neg_words if word in text_lower)
        
        total_count = pos_count + neg_count
        if total_count == 0:
            sentiment_score = 0.0
            confidence = 0.3
        else:
            sentiment_score = (pos_count - neg_count) / total_count
            sentiment_score = max(min(sentiment_score, 1.0), -1.0) * 0.6  # Scale down for conservatism
            confidence = min(0.7, 0.3 + 0.1 * total_count)  # Higher confidence with more keywords
        
        polarity = self._determine_polarity(sentiment_score)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentAnalysis(
            text_snippet=text[:512],
            overall_sentiment=sentiment_score,
            confidence=confidence,
            polarity=polarity,
            sentiment_explanation=f"Keyword-based analysis found {pos_count} positive and {neg_count} negative indicators.",
            model_used="keyword_fallback",
            processing_time=processing_time
        )


# Global sentiment ensemble instance
_sentiment_ensemble = None


async def get_sentiment_ensemble() -> SentimentEnsemble:
    """Get or create sentiment ensemble instance"""
    global _sentiment_ensemble
    if _sentiment_ensemble is None:
        _sentiment_ensemble = SentimentEnsemble()
        await _sentiment_ensemble.initialize()
    return _sentiment_ensemble


# Legacy function for backward compatibility
async def analyze_sentiment(doc_id: str) -> float:
    """Legacy sentiment analysis function"""
    text = get_doc(doc_id)
    if not text:
        return 0.0
    
    ensemble = await get_sentiment_ensemble()
    result = await ensemble.analyze_text(text)
    return result.overall_sentiment


# New enhanced functions
async def analyze_document_sentiment(doc_id: str) -> SentimentAnalysis:
    """Analyze sentiment of a document with full details"""
    text = get_doc(doc_id)
    if not text:
        raise ValueError(f"Document {doc_id} not found")
    
    ensemble = await get_sentiment_ensemble()
    result = await ensemble.analyze_text(text)
    result.document_id = doc_id
    return result


async def analyze_text_sentiment(text: str) -> SentimentAnalysis:
    """Analyze sentiment of arbitrary text"""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    ensemble = await get_sentiment_ensemble()
    return await ensemble.analyze_text(text)


async def analyze_topic_sentiment(text: str, topics: List[SentimentTopic] = None) -> Dict[str, TopicSentiment]:
    """Analyze sentiment for specific financial topics"""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    ensemble = await get_sentiment_ensemble()
    
    # Use all topics if none specified
    if topics is None:
        topics = list(SentimentTopic)
    
    topic_results = {}
    
    for topic in topics:
        topic_sentiment = await ensemble.analyze_topic_sentiment(text, topic)
        topic_results[topic.value] = topic_sentiment
    
    return topic_results


async def extract_topic_contexts(text: str, topic: SentimentTopic, window_size: int = 100) -> List[str]:
    """Extract text contexts relevant to a specific topic"""
    ensemble = await get_sentiment_ensemble()
    keywords = ensemble.topic_keywords.get(topic, [])
    
    if not keywords:
        return []
    
    contexts = []
    text_lower = text.lower()
    
    for keyword in keywords:
        # Find all occurrences of the keyword
        start = 0
        while True:
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break
            
            # Extract context window around the keyword
            context_start = max(0, pos - window_size)
            context_end = min(len(text), pos + len(keyword) + window_size)
            context = text[context_start:context_end].strip()
            
            if context and context not in contexts:
                contexts.append(context)
            
            start = pos + 1
    
    return contexts[:5]  # Return top 5 contexts


async def generate_sentiment_explanation(
    sentiment_analysis: SentimentAnalysis, 
    topic_sentiments: Dict[str, TopicSentiment] = None
) -> str:
    """Generate detailed explanation for sentiment analysis results"""
    explanation_parts = []
    
    # Overall sentiment explanation
    polarity = sentiment_analysis.polarity.value.replace('_', ' ').title()
    explanation_parts.append(
        f"Overall sentiment is {polarity} with a score of {sentiment_analysis.overall_sentiment:.3f} "
        f"(confidence: {sentiment_analysis.confidence:.3f})"
    )
    
    # Model information
    if sentiment_analysis.model_used:
        explanation_parts.append(f"Analysis performed using {sentiment_analysis.model_used}")
    
    # Topic-specific insights
    if topic_sentiments:
        positive_topics = []
        negative_topics = []
        neutral_topics = []
        
        for topic_name, topic_sentiment in topic_sentiments.items():
            if topic_sentiment.sentiment_score > 0.2:
                positive_topics.append((topic_name, topic_sentiment.sentiment_score))
            elif topic_sentiment.sentiment_score < -0.2:
                negative_topics.append((topic_name, topic_sentiment.sentiment_score))
            else:
                neutral_topics.append((topic_name, topic_sentiment.sentiment_score))
        
        if positive_topics:
            top_positive = sorted(positive_topics, key=lambda x: x[1], reverse=True)[:2]
            topics_str = ", ".join([f"{topic.replace('_', ' ')}" for topic, _ in top_positive])
            explanation_parts.append(f"Most positive aspects: {topics_str}")
        
        if negative_topics:
            top_negative = sorted(negative_topics, key=lambda x: x[1])[:2]
            topics_str = ", ".join([f"{topic.replace('_', ' ')}" for topic, _ in top_negative])
            explanation_parts.append(f"Most concerning aspects: {topics_str}")
    
    # Processing details
    if sentiment_analysis.processing_time:
        explanation_parts.append(f"Analysis completed in {sentiment_analysis.processing_time:.3f} seconds")
    
    return ". ".join(explanation_parts) + "."


async def get_sentiment_trends(
    company: str, 
    time_period: str = "6M", 
    topic: Optional[SentimentTopic] = None
) -> SentimentTrends:
    """Get historical sentiment trends for a company"""
    from ..database.connection import get_database
    
    db = await get_database()
    
    # Calculate date range based on time period
    end_date = datetime.now()
    if time_period == "1M":
        start_date = end_date - timedelta(days=30)
    elif time_period == "3M":
        start_date = end_date - timedelta(days=90)
    elif time_period == "6M":
        start_date = end_date - timedelta(days=180)
    elif time_period == "1Y":
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=180)  # Default to 6M
    
    # Query sentiment data from database
    query = """
        SELECT 
            DATE_TRUNC('week', sa.created_at) as week_date,
            AVG(sa.overall_sentiment) as avg_sentiment,
            AVG(sa.confidence) as avg_confidence,
            COUNT(*) as document_count,
            sa.topic_sentiments
        FROM sentiment_analysis sa
        JOIN documents d ON sa.document_id = d.id
        WHERE d.company = $1 
        AND sa.created_at >= $2 
        AND sa.created_at <= $3
    """
    
    params = [company, start_date, end_date]
    
    if topic:
        query += " AND sa.topic_sentiments ? $4"
        params.append(topic.value)
    
    query += """
        GROUP BY DATE_TRUNC('week', sa.created_at), sa.topic_sentiments
        ORDER BY week_date
    """
    
    try:
        rows = await db.fetch(query, *params)
        
        trends = []
        sentiment_scores = []
        
        for row in rows:
            sentiment_score = float(row['avg_sentiment'])
            confidence = float(row['avg_confidence'])
            
            # If analyzing specific topic, extract topic sentiment
            if topic and row['topic_sentiments']:
                topic_sentiments = row['topic_sentiments']
                if topic.value in topic_sentiments:
                    sentiment_score = float(topic_sentiments[topic.value])
            
            trend = SentimentTrend(
                date=row['week_date'],
                sentiment_score=sentiment_score,
                confidence=confidence,
                document_count=int(row['document_count']),
                topic=topic
            )
            trends.append(trend)
            sentiment_scores.append(sentiment_score)
        
        # Calculate statistics
        if sentiment_scores:
            average_sentiment = float(np.mean(sentiment_scores))
            volatility = float(np.std(sentiment_scores))
            
            # Determine trend direction
            if len(sentiment_scores) >= 2:
                recent_avg = np.mean(sentiment_scores[-3:])  # Last 3 data points
                earlier_avg = np.mean(sentiment_scores[:3])   # First 3 data points
                
                if recent_avg > earlier_avg + 0.1:
                    trend_direction = "improving"
                elif recent_avg < earlier_avg - 0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "insufficient_data"
        else:
            average_sentiment = 0.0
            volatility = 0.0
            trend_direction = "no_data"
        
        return SentimentTrends(
            company=company,
            topic=topic,
            time_period=time_period,
            trends=trends,
            average_sentiment=average_sentiment,
            volatility=volatility,
            trend_direction=trend_direction
        )
        
    except Exception as e:
        logger.error(f"Failed to get sentiment trends for {company}: {e}")
        # Return empty trends on error
        return SentimentTrends(
            company=company,
            topic=topic,
            time_period=time_period,
            trends=[],
            average_sentiment=0.0,
            volatility=0.0,
            trend_direction="error"
        )


async def compare_company_sentiment(
    companies: List[str], 
    topic: Optional[SentimentTopic] = None,
    days_back: int = 30
) -> SentimentComparison:
    """Compare sentiment across multiple companies"""
    from ..database.connection import get_database
    
    db = await get_database()
    
    comparison_date = datetime.now()
    start_date = comparison_date - timedelta(days=days_back)
    
    sentiment_scores = {}
    confidence_scores = {}
    
    for company in companies:
        query = """
            SELECT 
                AVG(sa.overall_sentiment) as avg_sentiment,
                AVG(sa.confidence) as avg_confidence,
                sa.topic_sentiments
            FROM sentiment_analysis sa
            JOIN documents d ON sa.document_id = d.id
            WHERE d.company = $1 
            AND sa.created_at >= $2 
            AND sa.created_at <= $3
        """
        
        params = [company, start_date, comparison_date]
        
        if topic:
            query += " AND sa.topic_sentiments ? $4"
            params.append(topic.value)
        
        query += " GROUP BY sa.topic_sentiments"
        
        try:
            rows = await db.fetch(query, *params)
            
            if rows:
                # Calculate weighted average across all documents
                total_sentiment = 0.0
                total_confidence = 0.0
                total_weight = 0.0
                
                for row in rows:
                    sentiment = float(row['avg_sentiment'])
                    confidence = float(row['avg_confidence'])
                    
                    # If analyzing specific topic, extract topic sentiment
                    if topic and row['topic_sentiments']:
                        topic_sentiments = row['topic_sentiments']
                        if topic.value in topic_sentiments:
                            sentiment = float(topic_sentiments[topic.value])
                    
                    weight = confidence  # Weight by confidence
                    total_sentiment += sentiment * weight
                    total_confidence += confidence * weight
                    total_weight += weight
                
                if total_weight > 0:
                    sentiment_scores[company] = total_sentiment / total_weight
                    confidence_scores[company] = total_confidence / total_weight
                else:
                    sentiment_scores[company] = 0.0
                    confidence_scores[company] = 0.0
            else:
                sentiment_scores[company] = 0.0
                confidence_scores[company] = 0.0
                
        except Exception as e:
            logger.error(f"Failed to get sentiment for {company}: {e}")
            sentiment_scores[company] = 0.0
            confidence_scores[company] = 0.0
    
    # Calculate relative rankings
    sorted_companies = sorted(companies, key=lambda c: sentiment_scores[c], reverse=True)
    relative_rankings = {company: rank + 1 for rank, company in enumerate(sorted_companies)}
    
    # Generate analysis summary
    best_company = sorted_companies[0] if sorted_companies else None
    worst_company = sorted_companies[-1] if sorted_companies else None
    
    if best_company and worst_company and len(companies) > 1:
        best_score = sentiment_scores[best_company]
        worst_score = sentiment_scores[worst_company]
        
        analysis_summary = (
            f"{best_company} shows the most positive sentiment ({best_score:.3f}), "
            f"while {worst_company} has the most negative sentiment ({worst_score:.3f}). "
            f"Sentiment spread: {best_score - worst_score:.3f}"
        )
    else:
        analysis_summary = "Insufficient data for meaningful comparison"
    
    return SentimentComparison(
        companies=companies,
        comparison_date=comparison_date,
        topic=topic,
        sentiment_scores=sentiment_scores,
        confidence_scores=confidence_scores,
        relative_rankings=relative_rankings,
        analysis_summary=analysis_summary
    )


async def detect_sentiment_deviation(
    company: str, 
    threshold: float = 0.3,
    lookback_days: int = 90,
    recent_days: int = 7
) -> Optional[SentimentAlert]:
    """Detect significant sentiment deviations"""
    from ..database.connection import get_database
    
    db = await get_database()
    
    current_date = datetime.now()
    recent_start = current_date - timedelta(days=recent_days)
    historical_start = current_date - timedelta(days=lookback_days)
    historical_end = recent_start
    
    try:
        # Get recent sentiment
        recent_query = """
            SELECT AVG(sa.overall_sentiment) as avg_sentiment
            FROM sentiment_analysis sa
            JOIN documents d ON sa.document_id = d.id
            WHERE d.company = $1 
            AND sa.created_at >= $2 
            AND sa.created_at <= $3
        """
        
        recent_result = await db.fetchrow(recent_query, company, recent_start, current_date)
        recent_sentiment = float(recent_result['avg_sentiment']) if recent_result and recent_result['avg_sentiment'] else 0.0
        
        # Get historical sentiment
        historical_query = """
            SELECT 
                AVG(sa.overall_sentiment) as avg_sentiment,
                STDDEV(sa.overall_sentiment) as sentiment_stddev
            FROM sentiment_analysis sa
            JOIN documents d ON sa.document_id = d.id
            WHERE d.company = $1 
            AND sa.created_at >= $2 
            AND sa.created_at <= $3
        """
        
        historical_result = await db.fetchrow(historical_query, company, historical_start, historical_end)
        
        if not historical_result or historical_result['avg_sentiment'] is None:
            return None
        
        historical_sentiment = float(historical_result['avg_sentiment'])
        sentiment_stddev = float(historical_result['sentiment_stddev']) if historical_result['sentiment_stddev'] else 0.1
        
        # Calculate change magnitude and significance
        change_magnitude = abs(recent_sentiment - historical_sentiment)
        significance_level = change_magnitude / max(sentiment_stddev, 0.1)  # Prevent division by zero
        
        # Check if deviation exceeds threshold
        if change_magnitude >= threshold and significance_level >= 2.0:  # 2 standard deviations
            alert_type = "positive_surge" if recent_sentiment > historical_sentiment else "negative_decline"
            
            description = (
                f"Significant sentiment change detected for {company}. "
                f"Recent sentiment ({recent_sentiment:.3f}) differs from historical average "
                f"({historical_sentiment:.3f}) by {change_magnitude:.3f} points "
                f"({significance_level:.1f} standard deviations)."
            )
            
            return SentimentAlert(
                company=company,
                alert_type=alert_type,
                current_sentiment=recent_sentiment,
                previous_sentiment=historical_sentiment,
                change_magnitude=change_magnitude,
                significance_level=min(significance_level, 1.0),  # Cap at 1.0
                description=description
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to detect sentiment deviation for {company}: {e}")
        return None


async def store_sentiment_analysis(sentiment_analysis: SentimentAnalysis) -> str:
    """Store sentiment analysis results in database"""
    from ..database.connection import get_database
    
    db = await get_database()
    
    try:
        query = """
            INSERT INTO sentiment_analysis 
            (id, document_id, text_snippet, overall_sentiment, confidence, polarity, 
             topic_sentiments, sentiment_explanation, model_used, processing_time, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """
        
        result = await db.fetchrow(
            query,
            sentiment_analysis.id,
            sentiment_analysis.document_id,
            sentiment_analysis.text_snippet,
            sentiment_analysis.overall_sentiment,
            sentiment_analysis.confidence,
            sentiment_analysis.polarity.value,
            sentiment_analysis.topic_sentiments,
            sentiment_analysis.sentiment_explanation,
            sentiment_analysis.model_used,
            sentiment_analysis.processing_time,
            sentiment_analysis.created_at
        )
        
        return str(result['id'])
        
    except Exception as e:
        logger.error(f"Failed to store sentiment analysis: {e}")
        raise
