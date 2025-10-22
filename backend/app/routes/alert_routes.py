"""
Alert management API routes
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from ..services.alert_service import get_alert_service, AlertService, AlertRule, Alert, AlertType, AlertSeverity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


class CreateAlertRuleRequest(BaseModel):
    name: str
    type: AlertType
    severity: AlertSeverity
    conditions: dict
    user_id: Optional[str] = None


class UpdateAlertRuleRequest(BaseModel):
    name: Optional[str] = None
    severity: Optional[AlertSeverity] = None
    conditions: Optional[dict] = None
    enabled: Optional[bool] = None


class TriggerAlertRequest(BaseModel):
    rule_id: str
    data: dict


@router.post("/rules", response_model=dict)
async def create_alert_rule(
    request: CreateAlertRuleRequest,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Create a new alert rule"""
    try:
        import uuid
        rule_id = str(uuid.uuid4())
        
        rule = AlertRule(
            id=rule_id,
            name=request.name,
            type=request.type,
            severity=request.severity,
            conditions=request.conditions,
            user_id=request.user_id
        )
        
        created_rule_id = await alert_service.create_alert_rule(rule)
        
        return {
            "rule_id": created_rule_id,
            "message": "Alert rule created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail="Failed to create alert rule")


@router.get("/rules", response_model=List[dict])
async def get_alert_rules(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get all alert rules"""
    try:
        rules = list(alert_service.alert_rules.values())
        
        if user_id:
            rules = [rule for rule in rules if rule.user_id == user_id or rule.user_id is None]
        
        return [rule.dict() for rule in rules]
        
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert rules")


@router.get("/rules/{rule_id}", response_model=dict)
async def get_alert_rule(
    rule_id: str,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get a specific alert rule"""
    try:
        if rule_id not in alert_service.alert_rules:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        rule = alert_service.alert_rules[rule_id]
        return rule.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert rule")


@router.put("/rules/{rule_id}", response_model=dict)
async def update_alert_rule(
    rule_id: str,
    request: UpdateAlertRuleRequest,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Update an alert rule"""
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        success = await alert_service.update_alert_rule(rule_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return {"message": "Alert rule updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating alert rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update alert rule")


@router.delete("/rules/{rule_id}", response_model=dict)
async def delete_alert_rule(
    rule_id: str,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Delete an alert rule"""
    try:
        success = await alert_service.delete_alert_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return {"message": "Alert rule deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting alert rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete alert rule")


@router.post("/trigger", response_model=dict)
async def trigger_alert(
    request: TriggerAlertRequest,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Manually trigger an alert"""
    try:
        alert_id = await alert_service.trigger_alert(request.rule_id, request.data)
        
        if not alert_id:
            raise HTTPException(status_code=400, detail="Failed to trigger alert")
        
        return {
            "alert_id": alert_id,
            "message": "Alert triggered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger alert")


@router.get("/active", response_model=List[dict])
async def get_active_alerts(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get active alerts"""
    try:
        alerts = await alert_service.get_active_alerts(user_id)
        return [alert.dict() for alert in alerts]
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active alerts")


@router.post("/acknowledge/{alert_id}", response_model=dict)
async def acknowledge_alert(
    alert_id: str,
    user_id: Optional[str] = Query(None, description="User acknowledging the alert"),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Acknowledge an alert"""
    try:
        success = await alert_service.acknowledge_alert(alert_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/setup-defaults/{user_id}", response_model=dict)
async def setup_default_alert_rules(
    user_id: str,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Set up default alert rules for a user"""
    try:
        rule_ids = await alert_service.create_default_alert_rules(user_id)
        
        return {
            "rule_ids": rule_ids,
            "message": f"Created {len(rule_ids)} default alert rules"
        }
        
    except Exception as e:
        logger.error(f"Error setting up default alert rules for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to set up default alert rules")


@router.get("/types", response_model=List[dict])
async def get_alert_types():
    """Get available alert types and severities"""
    return {
        "alert_types": [
            {"value": alert_type.value, "label": alert_type.value.replace("_", " ").title()}
            for alert_type in AlertType
        ],
        "severities": [
            {"value": severity.value, "label": severity.value.title()}
            for severity in AlertSeverity
        ]
    }


@router.post("/test-price-movement", response_model=dict)
async def test_price_movement_alert(
    ticker: str,
    current_price: float,
    previous_price: float,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Test price movement alert detection"""
    try:
        await alert_service.check_price_movement_alerts(ticker, current_price, previous_price)
        
        return {
            "message": f"Price movement check completed for {ticker}",
            "price_change_percent": abs((current_price - previous_price) / previous_price) * 100
        }
        
    except Exception as e:
        logger.error(f"Error testing price movement alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to test price movement alert")


@router.post("/test-volume-spike", response_model=dict)
async def test_volume_spike_alert(
    ticker: str,
    current_volume: int,
    average_volume: int,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Test volume spike alert detection"""
    try:
        await alert_service.check_volume_spike_alerts(ticker, current_volume, average_volume)
        
        volume_multiplier = current_volume / average_volume if average_volume > 0 else 0
        
        return {
            "message": f"Volume spike check completed for {ticker}",
            "volume_multiplier": volume_multiplier
        }
        
    except Exception as e:
        logger.error(f"Error testing volume spike alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to test volume spike alert")