"""
Multi-source data integration service for the Ensemble Forecasting Engine.
Handles Yahoo Finance, Quandl, and Alpha Vantage API integrations with
data normalization, quality validation, caching, and rate limiting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import yfinance as yf
import aioredis
import json
from dataclasses import dataclass, asdict
from enum import Enum
import time
import hashlib

try:
    import quandl
except ImportError:
    quandl = None

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
except ImportError:
    TimeSeries = None
    FundamentalData = None

logger = logging.getLogger(__name__)


class DataSource(Enum):
    YAHOO_FINANCE = "yahoo_finance"
    QUANDL = "quandl"
    ALPHA_VANTAGE = "alpha_vantage"


class DataType(Enum):
    STOCK_PRICE = "stock_price"
    FUNDAMENTAL = "fundamental"
    ECONOMIC = "economic"
    MARKET_INDEX = "market_index"


@dataclass
class DataPoint:
    """Standardized data point structure"""
    timestamp: datetime
    value: float
    source: DataSource
    data_type: DataType
    symbol: str
    metric: str
    metadata: Dict[str, Any] = None


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float  # Percentage of non-null values
    consistency: float   # Consistency score across sources
    timeliness: float   # How recent the data is
    accuracy: float     # Estimated accuracy based on validation
    total_points: int
    missing_points: int
    outliers: int


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make an API call"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)


class DataCache:
    """Redis-based caching for external data"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    def _generate_key(self, source: DataSource, symbol: str, data_type: DataType, 
                     start_date: datetime, end_date: datetime) -> str:
        """Generate cache key"""
        key_data = f"{source.value}:{symbol}:{data_type.value}:{start_date.isoformat()}:{end_date.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, source: DataSource, symbol: str, data_type: DataType,
                  start_date: datetime, end_date: datetime) -> Optional[List[DataPoint]]:
        """Get cached data"""
        if not self.redis:
            return None
        
        try:
            key = self._generate_key(source, symbol, data_type, start_date, end_date)
            cached_data = await self.redis.get(key)
            if cached_data:
                data_list = json.loads(cached_data)
                return [DataPoint(**item) for item in data_list]
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def set(self, source: DataSource, symbol: str, data_type: DataType,
                  start_date: datetime, end_date: datetime, data: List[DataPoint],
                  ttl: int = 3600):
        """Cache data with TTL"""
        if not self.redis:
            return
        
        try:
            key = self._generate_key(source, symbol, data_type, start_date, end_date)
            # Convert DataPoint objects to dict for JSON serialization
            data_dict = []
            for point in data:
                point_dict = asdict(point)
                point_dict['timestamp'] = point.timestamp.isoformat()
                point_dict['source'] = point.source.value
                point_dict['data_type'] = point.data_type.value
                data_dict.append(point_dict)
            
            await self.redis.setex(key, ttl, json.dumps(data_dict))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")


class DataValidator:
    """Data quality validation and normalization"""
    
    @staticmethod
    def validate_data_point(data_point: DataPoint) -> bool:
        """Validate individual data point"""
        if data_point.value is None or pd.isna(data_point.value):
            return False
        
        if not isinstance(data_point.timestamp, datetime):
            return False
        
        # Check for reasonable value ranges based on data type
        if data_point.data_type == DataType.STOCK_PRICE:
            if data_point.value <= 0 or data_point.value > 100000:  # Reasonable stock price range
                return False
        
        return True
    
    @staticmethod
    def detect_outliers(data: List[DataPoint], threshold: float = 3.0) -> List[int]:
        """Detect outliers using z-score method"""
        if len(data) < 3:
            return []
        
        values = [point.value for point in data]
        df = pd.Series(values)
        z_scores = abs((df - df.mean()) / df.std())
        return [i for i, z in enumerate(z_scores) if z > threshold]
    
    @staticmethod
    def assess_quality(data: List[DataPoint]) -> DataQualityMetrics:
        """Assess data quality metrics"""
        total_points = len(data)
        if total_points == 0:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Completeness
        valid_points = [point for point in data if DataValidator.validate_data_point(point)]
        missing_points = total_points - len(valid_points)
        completeness = len(valid_points) / total_points
        
        # Timeliness (based on most recent data point)
        if valid_points:
            latest_timestamp = max(point.timestamp for point in valid_points)
            hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600
            timeliness = max(0, 1 - (hours_old / 168))  # Decay over a week
        else:
            timeliness = 0
        
        # Outliers
        outliers = DataValidator.detect_outliers(valid_points)
        
        # Consistency and accuracy (simplified metrics)
        consistency = max(0, 1 - (len(outliers) / max(1, len(valid_points))))
        accuracy = completeness * consistency * timeliness
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            timeliness=timeliness,
            accuracy=accuracy,
            total_points=total_points,
            missing_points=missing_points,
            outliers=len(outliers)
        )


class DataIntegrationService:
    """Main service for multi-source data integration"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, 
                 quandl_key: Optional[str] = None,
                 redis_url: str = "redis://localhost:6379"):
        self.alpha_vantage_key = alpha_vantage_key
        self.quandl_key = quandl_key
        
        # Initialize rate limiters for each source
        self.rate_limiters = {
            DataSource.YAHOO_FINANCE: RateLimiter(calls_per_minute=60),
            DataSource.QUANDL: RateLimiter(calls_per_minute=50),
            DataSource.ALPHA_VANTAGE: RateLimiter(calls_per_minute=5)  # Free tier limit
        }
        
        # Initialize cache
        self.cache = DataCache(redis_url)
        
        # Initialize API clients
        self._init_api_clients()
    
    def _init_api_clients(self):
        """Initialize API clients"""
        if self.alpha_vantage_key and TimeSeries:
            self.alpha_vantage_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.alpha_vantage_fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.alpha_vantage_ts = None
            self.alpha_vantage_fd = None
        
        if self.quandl_key and quandl:
            quandl.ApiConfig.api_key = self.quandl_key
    
    async def initialize(self):
        """Initialize the service"""
        await self.cache.connect()
    
    async def get_stock_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, sources: List[DataSource] = None) -> Dict[DataSource, List[DataPoint]]:
        """Get stock price data from multiple sources"""
        if sources is None:
            sources = [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE]
        
        results = {}
        
        for source in sources:
            try:
                # Check cache first
                cached_data = await self.cache.get(source, symbol, DataType.STOCK_PRICE, start_date, end_date)
                if cached_data:
                    results[source] = cached_data
                    continue
                
                # Acquire rate limit permission
                await self.rate_limiters[source].acquire()
                
                # Fetch data from source
                if source == DataSource.YAHOO_FINANCE:
                    data = await self._fetch_yahoo_finance_data(symbol, start_date, end_date)
                elif source == DataSource.ALPHA_VANTAGE:
                    data = await self._fetch_alpha_vantage_data(symbol, start_date, end_date)
                else:
                    continue
                
                if data:
                    results[source] = data
                    # Cache the data
                    await self.cache.set(source, symbol, DataType.STOCK_PRICE, start_date, end_date, data)
                
            except Exception as e:
                logger.error(f"Error fetching data from {source.value} for {symbol}: {e}")
                continue
        
        return results
    
    async def _fetch_yahoo_finance_data(self, symbol: str, start_date: datetime, 
                                      end_date: datetime) -> List[DataPoint]:
        """Fetch data from Yahoo Finance"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, 
                lambda: yf.download(symbol, start=start_date.date(), end=end_date.date(), progress=False)
            )
            
            if data.empty:
                return []
            
            data_points = []
            for timestamp, row in data.iterrows():
                if pd.notna(row['Close']):
                    data_points.append(DataPoint(
                        timestamp=timestamp.to_pydatetime(),
                        value=float(row['Close']),
                        source=DataSource.YAHOO_FINANCE,
                        data_type=DataType.STOCK_PRICE,
                        symbol=symbol,
                        metric='close_price',
                        metadata={
                            'open': float(row['Open']) if pd.notna(row['Open']) else None,
                            'high': float(row['High']) if pd.notna(row['High']) else None,
                            'low': float(row['Low']) if pd.notna(row['Low']) else None,
                            'volume': int(row['Volume']) if pd.notna(row['Volume']) else None
                        }
                    ))
            
            return data_points
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error for {symbol}: {e}")
            return []
    
    async def _fetch_alpha_vantage_data(self, symbol: str, start_date: datetime, 
                                      end_date: datetime) -> List[DataPoint]:
        """Fetch data from Alpha Vantage"""
        if not self.alpha_vantage_ts:
            return []
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data, meta_data = await loop.run_in_executor(
                None,
                lambda: self.alpha_vantage_ts.get_daily_adjusted(symbol=symbol, outputsize='full')
            )
            
            if data.empty:
                return []
            
            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            data_points = []
            for timestamp, row in data.iterrows():
                if pd.notna(row['5. adjusted close']):
                    data_points.append(DataPoint(
                        timestamp=timestamp.to_pydatetime(),
                        value=float(row['5. adjusted close']),
                        source=DataSource.ALPHA_VANTAGE,
                        data_type=DataType.STOCK_PRICE,
                        symbol=symbol,
                        metric='adjusted_close_price',
                        metadata={
                            'open': float(row['1. open']) if pd.notna(row['1. open']) else None,
                            'high': float(row['2. high']) if pd.notna(row['2. high']) else None,
                            'low': float(row['3. low']) if pd.notna(row['3. low']) else None,
                            'close': float(row['4. close']) if pd.notna(row['4. close']) else None,
                            'volume': int(row['6. volume']) if pd.notna(row['6. volume']) else None
                        }
                    ))
            
            return data_points
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error for {symbol}: {e}")
            return []
    
    async def get_economic_data(self, dataset: str, start_date: datetime, 
                              end_date: datetime) -> List[DataPoint]:
        """Get economic data from Quandl"""
        if not quandl or not self.quandl_key:
            return []
        
        try:
            # Check cache first
            cached_data = await self.cache.get(DataSource.QUANDL, dataset, DataType.ECONOMIC, start_date, end_date)
            if cached_data:
                return cached_data
            
            # Acquire rate limit permission
            await self.rate_limiters[DataSource.QUANDL].acquire()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: quandl.get(dataset, start_date=start_date.date(), end_date=end_date.date())
            )
            
            if data.empty:
                return []
            
            data_points = []
            for timestamp, value in data.iloc[:, 0].items():  # Use first column
                if pd.notna(value):
                    data_points.append(DataPoint(
                        timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                        value=float(value),
                        source=DataSource.QUANDL,
                        data_type=DataType.ECONOMIC,
                        symbol=dataset,
                        metric='value'
                    ))
            
            # Cache the data
            await self.cache.set(DataSource.QUANDL, dataset, DataType.ECONOMIC, start_date, end_date, data_points)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Quandl fetch error for {dataset}: {e}")
            return []
    
    def normalize_data(self, data_dict: Dict[DataSource, List[DataPoint]]) -> List[DataPoint]:
        """Normalize and merge data from multiple sources"""
        all_data = []
        
        for source, data_points in data_dict.items():
            # Validate each data point
            valid_points = [point for point in data_points if DataValidator.validate_data_point(point)]
            all_data.extend(valid_points)
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x.timestamp)
        
        # Remove duplicates (same timestamp and symbol)
        seen = set()
        unique_data = []
        for point in all_data:
            key = (point.timestamp.date(), point.symbol)
            if key not in seen:
                seen.add(key)
                unique_data.append(point)
        
        return unique_data
    
    def assess_data_quality(self, data: List[DataPoint]) -> DataQualityMetrics:
        """Assess overall data quality"""
        return DataValidator.assess_quality(data)
    
    async def get_consolidated_stock_data(self, symbol: str, start_date: datetime, 
                                        end_date: datetime) -> tuple[List[DataPoint], DataQualityMetrics]:
        """Get consolidated stock data from all available sources with quality assessment"""
        # Fetch from all sources
        source_data = await self.get_stock_data(symbol, start_date, end_date)
        
        # Normalize and merge
        consolidated_data = self.normalize_data(source_data)
        
        # Assess quality
        quality_metrics = self.assess_data_quality(consolidated_data)
        
        return consolidated_data, quality_metrics


# Global instance
data_integration_service = DataIntegrationService()