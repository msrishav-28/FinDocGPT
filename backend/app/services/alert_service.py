"""
Alert service for real-time alert generation and notification management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel
import aioredis
from ..config import get_settings
from .websocket_service import connection_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    PRICE_MOVEMENT = "price_movement"
    VOLUME_SPIKE = "volume_spike"
    SENTIMENT_CHANGE = "sentiment_change"
    ANOMALY_DETECTED = "anomaly_detected"
    FORECAST_UPDATE = "forecast_update"
    RECOMMENDATION_CHANGE = "recommendation_change"
    SYSTEM_ALERT = "system_alert"


class AlertRule(BaseModel):
    """Alert rule configuration"""
    id: str
    name: str
    type: AlertType
    severity: AlertSeverity
    conditions: Dict[str, Any]
    enabled: bool = True
    user_id: Optional[str] = None
    created_at: datetime = datetime.now()
    last_triggered: Optional[datetime] = None


class Alert(BaseModel):
    """Alert instance"""
    id: str
    rule_id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: datetime = datetime.now()
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None


class AlertService:
    """Service for managing alerts and notifications"""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = {}
        self._monitoring_tasks: set = set()
        
    async def initialize(self):
        """Initialize Redis connection and load alert rules"""
        try:
            self.redis_client = aioredis.from_url(
                settings.database.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load existing alert rules from Redis
            await self._load_alert_rules()
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            logger.info("Alert service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize alert service: {e}")
    
    async def create_alert_rule(self, rule: AlertRule) -> str:
        """Create a new alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            
            # Save to Redis
            await self._save_alert_rule(rule)
            
            logger.info(f"Created alert rule: {rule.name} ({rule.id})")
            return rule.id
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {e}")
            raise
    
    async def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule"""
        try:
            if rule_id not in self.alert_rules:
                return False
            
            rule = self.alert_rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            await self._save_alert_rule(rule)
            
            logger.info(f"Updated alert rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert rule {rule_id}: {e}")
            return False
    
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule"""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.delete(f"alert_rule:{rule_id}")
                
                logger.info(f"Deleted alert rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting alert rule {rule_id}: {e}")
            return False
    
    async def trigger_alert(self, rule_id: str, data: Dict[str, Any]) -> Optional[str]:
        """Trigger an alert based on a rule"""
        try:
            if rule_id not in self.alert_rules:
                logger.warning(f"Alert rule not found: {rule_id}")
                return None
            
            rule = self.alert_rules[rule_id]
            if not rule.enabled:
                return None
            
            # Check cooldown period to prevent spam
            if rule.last_triggered:
                cooldown_minutes = rule.conditions.get('cooldown_minutes', 5)
                if datetime.now() - rule.last_triggered < timedelta(minutes=cooldown_minutes):
                    return None
            
            # Create alert
            alert_id = f"alert_{datetime.now().timestamp()}"
            alert = Alert(
                id=alert_id,
                rule_id=rule_id,
                type=rule.type,
                severity=rule.severity,
                title=self._generate_alert_title(rule, data),
                message=self._generate_alert_message(rule, data),
                data=data,
                user_id=rule.user_id
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            await self._save_alert(alert)
            
            # Update rule last triggered time
            rule.last_triggered = datetime.now()
            await self._save_alert_rule(rule)
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.info(f"Alert triggered: {alert.title} (ID: {alert_id})")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error triggering alert for rule {rule_id}: {e}")
            return None
    
    async def acknowledge_alert(self, alert_id: str, user_id: Optional[str] = None) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                
                await self._save_alert(alert)
                
                logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def get_active_alerts(self, user_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts for a user or all users"""
        try:
            alerts = list(self.active_alerts.values())
            
            if user_id:
                alerts = [a for a in alerts if a.user_id == user_id or a.user_id is None]
            
            # Sort by severity and creation time
            severity_order = {AlertSeverity.CRITICAL: 0, AlertSeverity.HIGH: 1, 
                            AlertSeverity.MEDIUM: 2, AlertSeverity.LOW: 3}
            
            alerts.sort(key=lambda x: (severity_order.get(x.severity, 4), x.created_at), reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def create_default_alert_rules(self, user_id: str) -> List[str]:
        """Create default alert rules for a new user"""
        default_rules = [
            AlertRule(
                id=f"price_movement_{user_id}",
                name="Large Price Movement",
                type=AlertType.PRICE_MOVEMENT,
                severity=AlertSeverity.HIGH,
                conditions={
                    "price_change_percent": 5.0,
                    "timeframe_minutes": 15,
                    "cooldown_minutes": 30
                },
                user_id=user_id
            ),
            AlertRule(
                id=f"volume_spike_{user_id}",
                name="Volume Spike",
                type=AlertType.VOLUME_SPIKE,
                severity=AlertSeverity.MEDIUM,
                conditions={
                    "volume_multiplier": 3.0,
                    "comparison_period_days": 7,
                    "cooldown_minutes": 60
                },
                user_id=user_id
            ),
            AlertRule(
                id=f"sentiment_change_{user_id}",
                name="Sentiment Change",
                type=AlertType.SENTIMENT_CHANGE,
                severity=AlertSeverity.MEDIUM,
                conditions={
                    "sentiment_change_threshold": 0.3,
                    "cooldown_minutes": 120
                },
                user_id=user_id
            ),
            AlertRule(
                id=f"anomaly_detected_{user_id}",
                name="Anomaly Detected",
                type=AlertType.ANOMALY_DETECTED,
                severity=AlertSeverity.HIGH,
                conditions={
                    "severity_threshold": "medium",
                    "cooldown_minutes": 60
                },
                user_id=user_id
            )
        ]
        
        rule_ids = []
        for rule in default_rules:
            rule_id = await self.create_alert_rule(rule)
            rule_ids.append(rule_id)
        
        return rule_ids
    
    async def check_price_movement_alerts(self, ticker: str, current_price: float, previous_price: float):
        """Check for price movement alerts"""
        try:
            price_change_percent = abs((current_price - previous_price) / previous_price) * 100
            
            for rule in self.alert_rules.values():
                if (rule.type == AlertType.PRICE_MOVEMENT and 
                    rule.enabled and 
                    price_change_percent >= rule.conditions.get('price_change_percent', 5.0)):
                    
                    await self.trigger_alert(rule.id, {
                        'ticker': ticker,
                        'current_price': current_price,
                        'previous_price': previous_price,
                        'price_change_percent': price_change_percent,
                        'direction': 'up' if current_price > previous_price else 'down'
                    })
                    
        except Exception as e:
            logger.error(f"Error checking price movement alerts: {e}")
    
    async def check_volume_spike_alerts(self, ticker: str, current_volume: int, average_volume: int):
        """Check for volume spike alerts"""
        try:
            if average_volume > 0:
                volume_multiplier = current_volume / average_volume
                
                for rule in self.alert_rules.values():
                    if (rule.type == AlertType.VOLUME_SPIKE and 
                        rule.enabled and 
                        volume_multiplier >= rule.conditions.get('volume_multiplier', 3.0)):
                        
                        await self.trigger_alert(rule.id, {
                            'ticker': ticker,
                            'current_volume': current_volume,
                            'average_volume': average_volume,
                            'volume_multiplier': volume_multiplier
                        })
                        
        except Exception as e:
            logger.error(f"Error checking volume spike alerts: {e}")
    
    def _generate_alert_title(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Generate alert title based on rule and data"""
        if rule.type == AlertType.PRICE_MOVEMENT:
            ticker = data.get('ticker', 'Unknown')
            direction = data.get('direction', 'changed')
            percent = data.get('price_change_percent', 0)
            return f"{ticker} price {direction} {percent:.1f}%"
        
        elif rule.type == AlertType.VOLUME_SPIKE:
            ticker = data.get('ticker', 'Unknown')
            multiplier = data.get('volume_multiplier', 0)
            return f"{ticker} volume spike {multiplier:.1f}x average"
        
        elif rule.type == AlertType.SENTIMENT_CHANGE:
            ticker = data.get('ticker', 'Unknown')
            return f"{ticker} sentiment changed significantly"
        
        elif rule.type == AlertType.ANOMALY_DETECTED:
            ticker = data.get('ticker', 'Unknown')
            return f"Anomaly detected in {ticker}"
        
        else:
            return rule.name
    
    def _generate_alert_message(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Generate alert message based on rule and data"""
        if rule.type == AlertType.PRICE_MOVEMENT:
            ticker = data.get('ticker', 'Unknown')
            current_price = data.get('current_price', 0)
            previous_price = data.get('previous_price', 0)
            percent = data.get('price_change_percent', 0)
            direction = data.get('direction', 'changed')
            return f"{ticker} price {direction} from ${previous_price:.2f} to ${current_price:.2f} ({percent:.1f}%)"
        
        elif rule.type == AlertType.VOLUME_SPIKE:
            ticker = data.get('ticker', 'Unknown')
            current_volume = data.get('current_volume', 0)
            multiplier = data.get('volume_multiplier', 0)
            return f"{ticker} trading volume is {multiplier:.1f}x higher than average ({current_volume:,} shares)"
        
        else:
            return f"Alert triggered for {rule.name}"
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            # Send WebSocket notification
            await connection_manager.broadcast_alert({
                'id': alert.id,
                'type': alert.type,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.created_at.isoformat(),
                'data': alert.data
            })
            
            # Here you could add other notification channels:
            # - Email notifications
            # - SMS notifications
            # - Push notifications
            # - Slack/Discord webhooks
            
        except Exception as e:
            logger.error(f"Error sending notifications for alert {alert.id}: {e}")
    
    async def _save_alert_rule(self, rule: AlertRule):
        """Save alert rule to Redis"""
        if self.redis_client:
            try:
                import json
                await self.redis_client.set(
                    f"alert_rule:{rule.id}",
                    json.dumps(rule.dict(), default=str),
                    ex=86400 * 30  # 30 days expiry
                )
            except Exception as e:
                logger.error(f"Error saving alert rule to Redis: {e}")
    
    async def _save_alert(self, alert: Alert):
        """Save alert to Redis"""
        if self.redis_client:
            try:
                import json
                await self.redis_client.set(
                    f"alert:{alert.id}",
                    json.dumps(alert.dict(), default=str),
                    ex=86400 * 7  # 7 days expiry
                )
            except Exception as e:
                logger.error(f"Error saving alert to Redis: {e}")
    
    async def _load_alert_rules(self):
        """Load alert rules from Redis"""
        if self.redis_client:
            try:
                import json
                keys = await self.redis_client.keys("alert_rule:*")
                for key in keys:
                    rule_data = await self.redis_client.get(key)
                    if rule_data:
                        rule_dict = json.loads(rule_data)
                        rule = AlertRule(**rule_dict)
                        self.alert_rules[rule.id] = rule
                
                logger.info(f"Loaded {len(self.alert_rules)} alert rules")
                
            except Exception as e:
                logger.error(f"Error loading alert rules from Redis: {e}")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        try:
            # Start market data monitoring task
            task = asyncio.create_task(self._monitor_market_data())
            self._monitoring_tasks.add(task)
            task.add_done_callback(self._monitoring_tasks.discard)
            
        except Exception as e:
            logger.error(f"Error starting monitoring tasks: {e}")
    
    async def _monitor_market_data(self):
        """Background task to monitor market data for alerts"""
        while True:
            try:
                # This would integrate with your market data service
                # to continuously monitor for alert conditions
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in market data monitoring: {e}")
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()


# Global alert service instance
alert_service = AlertService()


async def get_alert_service() -> AlertService:
    """Get the global alert service instance"""
    return alert_service