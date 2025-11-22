"""
Audit trail API routes
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..models.audit import AuditEventType, AuditSeverity
from ..services.audit_service import audit_service
from ..dependencies.auth import get_current_user, require_permission
from ..models.auth import UserPermission
from ..models.base import PaginationParams

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditLogQuery(BaseModel):
    """Query parameters for audit log search"""
    user_id: Optional[UUID] = None
    event_type: Optional[AuditEventType] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    severity: Optional[AuditSeverity] = None
    success: Optional[bool] = None
    search_term: Optional[str] = Field(None, description="Search in event name or description")


class AuditLogResponse(BaseModel):
    """Audit log response model"""
    id: UUID
    event_type: str
    severity: str
    event_name: str
    description: str
    user_id: Optional[UUID]
    username: Optional[str]
    user_role: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    endpoint: Optional[str]
    http_method: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    resource_name: Optional[str]
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    duration_ms: Optional[int]
    response_size: Optional[int]
    compliance_tags: List[str]
    created_at: datetime


class AuditSummary(BaseModel):
    """Audit summary statistics"""
    total_events: int
    success_rate: float
    avg_response_time_ms: float
    top_event_types: List[Dict[str, Any]]
    top_users: List[Dict[str, Any]]
    error_summary: List[Dict[str, Any]]
    security_incidents: int


class IntegrityCheckResult(BaseModel):
    """Result of audit log integrity check"""
    audit_id: UUID
    is_valid: bool
    message: str


@router.get("/logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    user_id: Optional[UUID] = Query(None),
    event_type: Optional[AuditEventType] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    severity: Optional[AuditSeverity] = Query(None),
    success: Optional[bool] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Get audit logs with filtering and pagination"""
    
    try:
        logs = await audit_service.get_audit_logs(
            user_id=user_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            start_date=start_date,
            end_date=end_date,
            severity=severity,
            success=success,
            limit=limit,
            offset=offset
        )
        
        return [AuditLogResponse(**log) for log in logs]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit logs: {str(e)}")


@router.get("/logs/{audit_id}", response_model=AuditLogResponse)
async def get_audit_log(
    audit_id: UUID,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Get a specific audit log entry"""
    
    try:
        logs = await audit_service.get_audit_logs(limit=1, offset=0)
        # Filter by ID (this is a simplified approach - in production, add direct ID query)
        matching_logs = [log for log in logs if log.get('id') == audit_id]
        
        if not matching_logs:
            raise HTTPException(status_code=404, detail="Audit log not found")
        
        return AuditLogResponse(**matching_logs[0])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit log: {str(e)}")


@router.get("/summary", response_model=AuditSummary)
async def get_audit_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Get audit summary statistics"""
    
    # Default to last 30 days if no date range specified
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()
    
    try:
        # Get all logs in the date range
        logs = await audit_service.get_audit_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Large limit to get all logs for summary
        )
        
        if not logs:
            return AuditSummary(
                total_events=0,
                success_rate=0.0,
                avg_response_time_ms=0.0,
                top_event_types=[],
                top_users=[],
                error_summary=[],
                security_incidents=0
            )
        
        # Calculate statistics
        total_events = len(logs)
        successful_events = sum(1 for log in logs if log.get('success', True))
        success_rate = (successful_events / total_events) * 100 if total_events > 0 else 0
        
        # Calculate average response time
        response_times = [log.get('duration_ms', 0) for log in logs if log.get('duration_ms')]
        avg_response_time_ms = sum(response_times) / len(response_times) if response_times else 0
        
        # Top event types
        event_type_counts = {}
        for log in logs:
            event_type = log.get('event_type', 'unknown')
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        top_event_types = [
            {"event_type": event_type, "count": count}
            for event_type, count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Top users
        user_counts = {}
        for log in logs:
            username = log.get('username') or 'anonymous'
            user_counts[username] = user_counts.get(username, 0) + 1
        
        top_users = [
            {"username": username, "count": count}
            for username, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Error summary
        error_counts = {}
        for log in logs:
            if not log.get('success', True):
                error_code = log.get('error_code', 'unknown_error')
                error_counts[error_code] = error_counts.get(error_code, 0) + 1
        
        error_summary = [
            {"error_code": error_code, "count": count}
            for error_code, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Security incidents (high/critical severity events)
        security_incidents = sum(
            1 for log in logs 
            if log.get('severity') in ['high', 'critical'] or 
            log.get('event_type') in ['unauthorized_access', 'permission_denied', 'suspicious_activity']
        )
        
        return AuditSummary(
            total_events=total_events,
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time_ms,
            top_event_types=top_event_types,
            top_users=top_users,
            error_summary=error_summary,
            security_incidents=security_incidents
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate audit summary: {str(e)}")


@router.post("/verify/{audit_id}", response_model=IntegrityCheckResult)
async def verify_audit_integrity(
    audit_id: UUID,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Verify the integrity of an audit log entry"""
    
    try:
        is_valid = await audit_service.verify_audit_integrity(audit_id)
        
        return IntegrityCheckResult(
            audit_id=audit_id,
            is_valid=is_valid,
            message="Audit log integrity verified" if is_valid else "Audit log integrity check failed"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify audit integrity: {str(e)}")


@router.post("/retention/apply")
async def apply_retention_policies(
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Manually trigger application of retention policies"""
    
    try:
        await audit_service.apply_retention_policies()
        
        # Log this administrative action
        await audit_service.log_event(
            event_type=AuditEventType.SYSTEM_CONFIG_CHANGED,
            event_name="Retention Policies Applied",
            description="Manual application of data retention policies",
            user_id=current_user.id,
            username=current_user.username,
            user_role=current_user.role.value,
            severity=AuditSeverity.MEDIUM
        )
        
        return {"message": "Retention policies applied successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply retention policies: {str(e)}")


@router.get("/export")
async def export_audit_logs(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    format: str = Query("json", regex="^(json|csv)$"),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Export audit logs for compliance reporting"""
    
    # Default to last 30 days if no date range specified
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()
    
    try:
        logs = await audit_service.get_audit_logs(
            start_date=start_date,
            end_date=end_date,
            limit=50000  # Large limit for export
        )
        
        # Log the export action
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            event_name="Audit Logs Exported",
            description=f"Exported {len(logs)} audit logs in {format} format",
            user_id=current_user.id,
            username=current_user.username,
            user_role=current_user.role.value,
            event_data={
                "export_format": format,
                "record_count": len(logs),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            },
            severity=AuditSeverity.HIGH,
            compliance_tags=["data_export", "compliance_reporting"]
        )
        
        if format == "json":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=logs,
                headers={"Content-Disposition": f"attachment; filename=audit_logs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"}
            )
        
        elif format == "csv":
            import csv
            import io
            from fastapi.responses import StreamingResponse
            
            output = io.StringIO()
            if logs:
                writer = csv.DictWriter(output, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            
            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=audit_logs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"}
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export audit logs: {str(e)}")