"""
Market data service for real-time market data broadcasting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
from ..config import get_settings
from .websocket_service import connection_manager
from .cache_service import cache_service, CacheKey

logger = logging.getLogger(__name__)
settings = get_settings()


class MarketDataService:
    """Service for managing real-time market data"""
    
    def __init__(self):
        self.cache_expiry = 300  # 5 minutes cache
        
    async def initialize(self):
        """Initialize market data service"""
        try:
            # Cache service is initialized globally
            logger.info("Market data service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize market data service: {e}")
    
    async def get_real_time_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a ticker"""
        try:
            # Check cache first
            cache_key = CacheKey.MARKET_QUOTE.format(ticker=ticker.upper())
            cached_data = await cache_service.get(cache_key)
            
            if cached_data:
                return cached_data
            
            # Fetch from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None
            
            quote_data = {
                "ticker": ticker.upper(),
                "current_price": current_price,
                "previous_close": info.get('previousClose'),
                "day_change": None,
                "day_change_percent": None,
                "volume": info.get('volume'),
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "day_high": info.get('dayHigh'),
                "day_low": info.get('dayLow'),
                "fifty_two_week_high": info.get('fiftyTwoWeekHigh'),
                "fifty_two_week_low": info.get('fiftyTwoWeekLow'),
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate day change
            if quote_data["previous_close"]:
                quote_data["day_change"] = current_price - quote_data["previous_close"]
                quote_data["day_change_percent"] = (quote_data["day_change"] / quote_data["previous_close"]) * 100
            
            # Cache the data
            await cache_service.set(cache_key, quote_data, cache_type="market")
            
            return quote_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {ticker}: {e}")
            return None
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview with major indices"""
        try:
            indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow, NASDAQ, Russell 2000
            overview_data = {}
            
            for index in indices:
                quote = await self.get_real_time_quote(index)
                if quote:
                    overview_data[index] = quote
            
            return {
                "indices": overview_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {}
    
    async def broadcast_market_update(self, ticker: str):
        """Broadcast market update for a specific ticker"""
        try:
            quote_data = await self.get_real_time_quote(ticker)
            if quote_data:
                await connection_manager.broadcast_market_update(ticker.upper(), quote_data)
                logger.debug(f"Broadcasted market update for {ticker}")
        except Exception as e:
            logger.error(f"Error broadcasting market update for {ticker}: {e}")
    
    async def start_market_data_stream(self, tickers: List[str], interval_seconds: int = 60):
        """Start streaming market data for specified tickers"""
        try:
            while True:
                for ticker in tickers:
                    await self.broadcast_market_update(ticker)
                    await asyncio.sleep(1)  # Small delay between tickers
                
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in market data stream: {e}")
    
    async def get_historical_data(self, ticker: str, period: str = "1d", interval: str = "1m") -> Optional[Dict[str, Any]]:
        """Get historical data for charting"""
        try:
            cache_key = CacheKey.MARKET_HISTORICAL.format(ticker=ticker.upper(), period=period, interval=interval)
            cached_data = await cache_service.get(cache_key)
            
            if cached_data:
                return cached_data
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                return None
            
            # Convert to JSON-serializable format
            historical_data = {
                "ticker": ticker.upper(),
                "period": period,
                "interval": interval,
                "data": [
                    {
                        "timestamp": index.isoformat(),
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"]
                    }
                    for index, row in hist.iterrows()
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            # Cache for shorter time for intraday data
            cache_type = "market" if interval.endswith('m') else "market_historical"
            await cache_service.set(cache_key, historical_data, cache_type=cache_type)
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cache service cleanup is handled globally
        pass


# Global market data service instance
market_data_service = MarketDataService()


async def get_market_data_service() -> MarketDataService:
    """Get the global market data service instance"""
    return market_data_service