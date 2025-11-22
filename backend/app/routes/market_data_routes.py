"""
Market data API routes
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from ..services.market_data_service import get_market_data_service, MarketDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market-data"])


@router.get("/quote/{ticker}")
async def get_quote(
    ticker: str,
    market_service: MarketDataService = Depends(get_market_data_service)
):
    """Get real-time quote for a ticker"""
    try:
        quote_data = await market_service.get_real_time_quote(ticker)
        
        if not quote_data:
            raise HTTPException(status_code=404, detail=f"Quote not found for ticker: {ticker}")
        
        return quote_data
        
    except Exception as e:
        logger.error(f"Error getting quote for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quote data")


@router.get("/overview")
async def get_market_overview(
    market_service: MarketDataService = Depends(get_market_data_service)
):
    """Get market overview with major indices"""
    try:
        overview_data = await market_service.get_market_overview()
        return overview_data
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market overview")


@router.get("/historical/{ticker}")
async def get_historical_data(
    ticker: str,
    period: str = Query(default="1d", description="Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query(default="1m", description="Interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    market_service: MarketDataService = Depends(get_market_data_service)
):
    """Get historical data for charting"""
    try:
        historical_data = await market_service.get_historical_data(ticker, period, interval)
        
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"Historical data not found for ticker: {ticker}")
        
        return historical_data
        
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")


@router.post("/broadcast/{ticker}")
async def broadcast_ticker_update(
    ticker: str,
    market_service: MarketDataService = Depends(get_market_data_service)
):
    """Manually trigger broadcast of market update for a ticker"""
    try:
        await market_service.broadcast_market_update(ticker)
        return {"message": f"Market update broadcasted for {ticker}"}
        
    except Exception as e:
        logger.error(f"Error broadcasting update for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast market update")


@router.get("/quotes")
async def get_multiple_quotes(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    market_service: MarketDataService = Depends(get_market_data_service)
):
    """Get quotes for multiple tickers"""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        quotes = {}
        
        for ticker in ticker_list:
            quote_data = await market_service.get_real_time_quote(ticker)
            if quote_data:
                quotes[ticker] = quote_data
        
        return {
            "quotes": quotes,
            "requested_tickers": ticker_list,
            "found_tickers": list(quotes.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting multiple quotes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quotes")