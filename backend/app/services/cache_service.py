"""
Redis caching service for the Financial Intelligence System

This service provides comprehensive caching capabilities including:
- Frequently accessed data caching
- Cache invalidation strategies
- Cache warming for improved response times
- Cache analytics and monitoring
"""

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import aioredis
from aioredis import Redis
from pydantic import BaseModel

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheKey:
    """Cache key constants and generators"""
    
    # Document processing cache keys
    DOCUMENT_CONTENT = "doc:content:{doc_id}"
    DOCUMENT_EMBEDDINGS = "doc:embeddings:{doc_id}"
    DOCUMENT_QA = "doc:qa:{doc_id}:{question_hash}"
    DOCUMENT_INSIGHTS = "doc:insights:{doc_id}"
    
    # Sentiment analysis cache keys
    SENTIMENT_ANALYSIS = "sentiment:{doc_id}"
    SENTIMENT_TRENDS = "sentiment:trends:{company}:{timeframe}"
    SENTIMENT_COMPARISON = "sentiment:comparison:{companies_hash}"
    
    # Anomaly detection cache keys
    ANOMALY_DETECTION = "anomaly:{company}:{metrics_hash}"
    ANOMALY_PATTERNS = "anomaly:patterns:{data_hash}"
    ANOMALY_HISTORY = "anomaly:history:{company}"
    
    # Forecasting cache keys
    STOCK_FORECAST = "forecast:stock:{ticker}:{horizons_hash}"
    METRIC_FORECAST = "forecast:metrics:{company}:{metrics_hash}"
    FORECAST_CONFIDENCE = "forecast:confidence:{forecast_id}"
    MODEL_PERFORMANCE = "forecast:performance:{model_id}"
    
    # Investment advisory cache keys
    INVESTMENT_RECOMMENDATION = "investment:recommendation:{ticker}:{context_hash}"
    PORTFOLIO_RISK = "investment:portfolio_risk:{portfolio_hash}"
    POSITION_SIZING = "investment:position_sizing:{recommendations_hash}"
    
    # Market data cache keys
    MARKET_QUOTE = "market:quote:{ticker}"
    MARKET_HISTORICAL = "market:historical:{ticker}:{period}:{interval}"
    MARKET_NEWS = "market:news:{ticker}"
    
    # External API cache keys
    YAHOO_FINANCE = "api:yahoo:{endpoint}:{params_hash}"
    ALPHA_VANTAGE = "api:alpha:{endpoint}:{params_hash}"
    QUANDL = "api:quandl:{endpoint}:{params_hash}"
    
    # Model cache keys
    MODEL_PREDICTIONS = "model:predictions:{model_name}:{input_hash}"
    MODEL_EMBEDDINGS = "model:embeddings:{model_name}:{text_hash}"
    
    @staticmethod
    def generate_hash(data: Any) -> str:
        """Generate hash for cache key from data"""
        import hashlib
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]


class CacheStats(BaseModel):
    """Cache statistics model"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheService:
    """Comprehensive Redis caching service"""
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.stats = CacheStats()
        self._connection_pool = None
        self._cache_warming_tasks: Dict[str, asyncio.Task] = {}
        
        # Cache TTL settings (in seconds)
        self.default_ttl = 3600  # 1 hour
        self.ttl_settings = {
            # Document processing - longer TTL as documents don't change often
            "document": 86400,  # 24 hours
            "embeddings": 86400,  # 24 hours
            "qa": 3600,  # 1 hour
            
            # Sentiment analysis - medium TTL as sentiment can change
            "sentiment": 7200,  # 2 hours
            "sentiment_trends": 1800,  # 30 minutes
            
            # Anomaly detection - shorter TTL for real-time detection
            "anomaly": 900,  # 15 minutes
            "anomaly_patterns": 1800,  # 30 minutes
            
            # Forecasting - medium TTL as forecasts update daily
            "forecast": 14400,  # 4 hours
            "model_performance": 86400,  # 24 hours
            
            # Investment recommendations - shorter TTL for timely decisions
            "investment": 1800,  # 30 minutes
            "portfolio": 3600,  # 1 hour
            
            # Market data - very short TTL for real-time data
            "market": 300,  # 5 minutes
            "market_historical": 3600,  # 1 hour
            
            # External APIs - respect rate limits
            "api": 600,  # 10 minutes
            
            # Model predictions - longer TTL for expensive computations
            "model": 7200,  # 2 hours
        }
    
    async def initialize(self):
        """Initialize Redis connection and cache warming"""
        try:
            # Create connection pool
            self._connection_pool = aioredis.ConnectionPool.from_url(
                settings.database.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.redis_client = aioredis.Redis(
                connection_pool=self._connection_pool,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache service initialized successfully")
            
            # Start cache warming for critical data
            await self._start_cache_warming()
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup Redis connections and background tasks"""
        try:
            # Cancel cache warming tasks
            for task in self._cache_warming_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Redis cache service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cache service cleanup: {e}")
    
    def _get_ttl(self, cache_type: str) -> int:
        """Get TTL for cache type"""
        return self.ttl_settings.get(cache_type, self.default_ttl)
    
    def _extract_cache_type(self, key: str) -> str:
        """Extract cache type from key"""
        if ":" in key:
            return key.split(":")[0]
        return "default"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            if not self.redis_client:
                return default
            
            value = await self.redis_client.get(key)
            if value is None:
                self.stats.misses += 1
                return default
            
            self.stats.hits += 1
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats.errors += 1
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        cache_type: Optional[str] = None
    ) -> bool:
        """Set value in cache"""
        try:
            if not self.redis_client:
                return False
            
            # Determine TTL
            if ttl is None:
                if cache_type:
                    ttl = self._get_ttl(cache_type)
                else:
                    ttl = self._get_ttl(self._extract_cache_type(key))
            
            # Serialize value if needed
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await self.redis_client.setex(key, ttl, value)
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if not self.redis_client:
                return False
            
            result = await self.redis_client.delete(key)
            if result > 0:
                self.stats.deletes += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            if not self.redis_client:
                return 0
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                self.stats.deletes += result
                return result
            return 0
            
        except Exception as e:
            logger.error(f"Cache delete pattern error for pattern {pattern}: {e}")
            self.stats.errors += 1
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if not self.redis_client:
                return False
            
            return await self.redis_client.exists(key) > 0
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key"""
        try:
            if not self.redis_client:
                return False
            
            return await self.redis_client.expire(key, ttl)
            
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        try:
            if not self.redis_client or not keys:
                return {}
            
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = value
                    self.stats.hits += 1
                else:
                    self.stats.misses += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            self.stats.errors += 1
            return {}
    
    async def set_many(
        self, 
        mapping: Dict[str, Any], 
        ttl: Optional[int] = None,
        cache_type: Optional[str] = None
    ) -> bool:
        """Set multiple values in cache"""
        try:
            if not self.redis_client or not mapping:
                return False
            
            # Determine TTL
            if ttl is None:
                ttl = self._get_ttl(cache_type or "default")
            
            # Prepare pipeline
            pipe = self.redis_client.pipeline()
            
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                pipe.setex(key, ttl, value)
            
            await pipe.execute()
            self.stats.sets += len(mapping)
            return True
            
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            self.stats.errors += 1
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        try:
            if not self.redis_client:
                return 0
            
            return await self.redis_client.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            self.stats.errors += 1
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = {}
            if self.redis_client:
                redis_info = await self.redis_client.info()
                info = {
                    "redis_version": redis_info.get("redis_version"),
                    "used_memory": redis_info.get("used_memory_human"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "total_commands_processed": redis_info.get("total_commands_processed"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                }
            
            return {
                "service_stats": self.stats.dict(),
                "redis_info": info,
                "cache_warming_tasks": len(self._cache_warming_tasks),
                "ttl_settings": self.ttl_settings
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def clear_all(self) -> bool:
        """Clear all cache (use with caution)"""
        try:
            if not self.redis_client:
                return False
            
            await self.redis_client.flushdb()
            logger.warning("All cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear_all error: {e}")
            self.stats.errors += 1
            return False
    
    # Cache invalidation strategies
    async def invalidate_document_cache(self, doc_id: str):
        """Invalidate all document-related cache"""
        patterns = [
            f"doc:*:{doc_id}",
            f"sentiment:{doc_id}",
            f"doc:*:{doc_id}:*"
        ]
        
        for pattern in patterns:
            await self.delete_pattern(pattern)
    
    async def invalidate_company_cache(self, company: str):
        """Invalidate all company-related cache"""
        patterns = [
            f"*:{company}",
            f"*:{company}:*",
            f"sentiment:*:{company}:*",
            f"anomaly:{company}:*",
            f"forecast:*:{company}:*"
        ]
        
        for pattern in patterns:
            await self.delete_pattern(pattern)
    
    async def invalidate_market_cache(self, ticker: str):
        """Invalidate market data cache for ticker"""
        patterns = [
            f"market:*:{ticker}",
            f"market:*:{ticker}:*",
            f"forecast:stock:{ticker}:*",
            f"investment:*:{ticker}:*"
        ]
        
        for pattern in patterns:
            await self.delete_pattern(pattern)
    
    # Cache warming strategies
    async def _start_cache_warming(self):
        """Start cache warming background tasks"""
        try:
            # Warm popular market data
            self._cache_warming_tasks["market_data"] = asyncio.create_task(
                self._warm_market_data()
            )
            
            # Warm model embeddings
            self._cache_warming_tasks["model_embeddings"] = asyncio.create_task(
                self._warm_model_embeddings()
            )
            
            logger.info("Cache warming tasks started")
            
        except Exception as e:
            logger.error(f"Error starting cache warming: {e}")
    
    async def _warm_market_data(self):
        """Warm cache with popular market data"""
        popular_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        while True:
            try:
                # This would integrate with market data service
                # For now, just log the warming attempt
                logger.debug(f"Warming market data cache for {len(popular_tickers)} tickers")
                
                # Wait 5 minutes before next warming
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data cache warming: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _warm_model_embeddings(self):
        """Warm cache with common model embeddings"""
        common_queries = [
            "revenue growth",
            "profit margin",
            "debt ratio",
            "cash flow",
            "market outlook",
            "competitive position"
        ]
        
        while True:
            try:
                # This would integrate with NLP models
                # For now, just log the warming attempt
                logger.debug(f"Warming model embeddings cache for {len(common_queries)} queries")
                
                # Wait 1 hour before next warming
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model embeddings cache warming: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    # Context managers for cache operations
    @asynccontextmanager
    async def cached_operation(self, key: str, ttl: Optional[int] = None):
        """Context manager for cached operations"""
        # Check cache first
        cached_result = await self.get(key)
        if cached_result is not None:
            yield cached_result
            return
        
        # Execute operation and cache result
        class CacheContext:
            def __init__(self, cache_service, cache_key, cache_ttl):
                self.cache_service = cache_service
                self.cache_key = cache_key
                self.cache_ttl = cache_ttl
                self.result = None
            
            async def set_result(self, result):
                self.result = result
                await self.cache_service.set(self.cache_key, result, self.cache_ttl)
        
        context = CacheContext(self, key, ttl)
        yield context


# Global cache service instance
cache_service = CacheService()


async def get_cache_service() -> CacheService:
    """Get cache service instance"""
    return cache_service


# Decorator for caching function results
def cached(
    key_template: str,
    ttl: Optional[int] = None,
    cache_type: Optional[str] = None
):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {"args": args, "kwargs": kwargs}
            key_hash = CacheKey.generate_hash(key_data)
            cache_key = key_template.format(hash=key_hash, **kwargs)
            
            # Check cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_service.set(cache_key, result, ttl, cache_type)
            
            return result
        
        return wrapper
    return decorator