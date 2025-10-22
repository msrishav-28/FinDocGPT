"""
WebSocket routes for real-time data streaming
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from ..services.websocket_service import get_connection_manager, ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(..., description="User ID for connection identification"),
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    WebSocket endpoint for real-time data streaming
    
    Supports the following message types:
    - subscribe: Subscribe to ticker updates
    - unsubscribe: Unsubscribe from ticker updates  
    - ping: Heartbeat message
    """
    await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Wait for client messages
            message = await websocket.receive_text()
            await connection_manager.handle_client_message(user_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user_id}")
        await connection_manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await connection_manager.disconnect(user_id)


@router.get("/connections")
async def get_active_connections(
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Get information about active WebSocket connections"""
    return {
        "active_connections": len(connection_manager.active_connections),
        "total_subscriptions": sum(len(subs) for subs in connection_manager.user_subscriptions.values()),
        "unique_tickers": len(set().union(*connection_manager.user_subscriptions.values())) if connection_manager.user_subscriptions else 0
    }


@router.post("/broadcast/alert")
async def broadcast_alert(
    alert_data: dict,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Broadcast alert to all connected users"""
    await connection_manager.broadcast_alert(alert_data)
    return {"message": "Alert broadcasted successfully"}


@router.post("/send/analysis/{user_id}")
async def send_analysis_update(
    user_id: str,
    analysis_type: str,
    data: dict,
    connection_manager: ConnectionManager = Depends(get_connection_manager)
):
    """Send analysis update to specific user"""
    await connection_manager.send_analysis_update(user_id, analysis_type, data)
    return {"message": f"Analysis update sent to user {user_id}"}