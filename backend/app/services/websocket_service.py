"""
WebSocket service for real-time data streaming and connection management
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import yfinance as yf
from ..config import get_settings
from .cache_service import cache_service, CacheKey

logger = logging.getLogger(__name__)
settings = get_settings()


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()


class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, Set[str]] = {}
        self.market_data_cache: Dict[str, Dict] = {}
        self._background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize WebSocket service and background tasks"""
        try:
            logger.info("WebSocket service initialized")
            
            # Start background market data streaming
            task = asyncio.create_task(self._stream_market_data())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket service: {e}")
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_subscriptions[user_id] = set()
        
        logger.info(f"WebSocket connection established for user: {user_id}")
        
        # Send initial connection confirmation
        await self._send_to_user(user_id, {
            "type": "connection_established",
            "data": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "message": "WebSocket connection established"
            }
        })
    
    async def disconnect(self, user_id: str):
        """Handle WebSocket disconnection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]
        
        logger.info(f"WebSocket connection closed for user: {user_id}")
    
    async def subscribe_to_ticker(self, user_id: str, ticker: str):
        """Subscribe user to ticker updates"""
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = set()
        
        self.user_subscriptions[user_id].add(ticker.upper())
        
        # Send current market data if available
        cache_key = CacheKey.MARKET_QUOTE.format(ticker=ticker.upper())
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            await self._send_to_user(user_id, {
                "type": "market_data",
                "data": {
                    "ticker": ticker.upper(),
                    **cached_data
                }
            })
        
        logger.info(f"User {user_id} subscribed to ticker: {ticker}")
    
    async def unsubscribe_from_ticker(self, user_id: str, ticker: str):
        """Unsubscribe user from ticker updates"""
        if user_id in self.user_subscriptions:
            self.user_subscriptions[user_id].discard(ticker.upper())
        
        logger.info(f"User {user_id} unsubscribed from ticker: {ticker}")
    
    async def broadcast_market_update(self, ticker: str, data: Dict[str, Any]):
        """Broadcast market data update to subscribed users"""
        message = {
            "type": "market_data",
            "data": {
                "ticker": ticker,
                **data
            }
        }
        
        # Update cache
        self.market_data_cache[ticker] = data
        cache_key = CacheKey.MARKET_QUOTE.format(ticker=ticker)
        await cache_service.set(cache_key, data, cache_type="market")
        
        # Send to subscribed users
        for user_id, subscriptions in self.user_subscriptions.items():
            if ticker in subscriptions:
                await self._send_to_user(user_id, message)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to all connected users"""
        message = {
            "type": "alert",
            "data": alert_data
        }
        
        for user_id in self.active_connections:
            await self._send_to_user(user_id, message)
    
    async def send_analysis_update(self, user_id: str, analysis_type: str, data: Dict[str, Any]):
        """Send analysis update to specific user"""
        message = {
            "type": "analysis_update",
            "data": {
                "analysis_type": analysis_type,
                **data
            }
        }
        
        await self._send_to_user(user_id, message)
    
    async def _send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to specific user"""
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                # Remove disconnected user
                await self.disconnect(user_id)
    
    async def _stream_market_data(self):
        """Background task to stream market data"""
        while True:
            try:
                # Get all unique tickers from subscriptions
                all_tickers = set()
                for subscriptions in self.user_subscriptions.values():
                    all_tickers.update(subscriptions)
                
                if all_tickers:
                    # Fetch market data for all subscribed tickers
                    for ticker in all_tickers:
                        try:
                            market_data = await self._fetch_market_data(ticker)
                            if market_data:
                                await self.broadcast_market_update(ticker, market_data)
                        except Exception as e:
                            logger.error(f"Error fetching market data for {ticker}: {e}")
                
                # Wait before next update (60 seconds as per requirements)
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in market data streaming: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_market_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch current market data for ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current price and basic metrics
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None
            
            market_data = {
                "current_price": current_price,
                "previous_close": info.get('previousClose'),
                "day_change": None,
                "day_change_percent": None,
                "volume": info.get('volume'),
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate day change
            if market_data["previous_close"]:
                market_data["day_change"] = current_price - market_data["previous_close"]
                market_data["day_change_percent"] = (market_data["day_change"] / market_data["previous_close"]) * 100
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None
    
    async def handle_client_message(self, user_id: str, message: str):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                ticker = data.get("ticker")
                if ticker:
                    await self.subscribe_to_ticker(user_id, ticker)
            
            elif message_type == "unsubscribe":
                ticker = data.get("ticker")
                if ticker:
                    await self.unsubscribe_from_ticker(user_id, ticker)
            
            elif message_type == "ping":
                await self._send_to_user(user_id, {
                    "type": "pong",
                    "data": {"timestamp": datetime.now().isoformat()}
                })
            
            else:
                logger.warning(f"Unknown message type from user {user_id}: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from user {user_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling client message from user {user_id}: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Close all WebSocket connections
        for user_id in list(self.active_connections.keys()):
            await self.disconnect(user_id)


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance"""
    return connection_manager