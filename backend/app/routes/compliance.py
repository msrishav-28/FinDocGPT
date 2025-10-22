"""
Compliance reporting and data governance API routes
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.compliance_service import compliance_service, ComplianceFramework, DataClassification
from ..dependencies.auth import get_current_user, require_permission
from ..models.auth import UserPermission
from ..models.base import PaginationParams

router = APIRouter(prefix="/compliance", tags=["compliance"])


class ComplianceReportRequest(BaseModel):
    """Request model for generating compliance reports"""
    report_type: str = Field(..., description="Type of compliance report")
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    period_start: datetime = Field(..., description="Start of reporting period")
    period_end: datetime = Field(..., description="End of reporting period")
    include_recommendations: bool = Field(default=True, description="Include recommendations")


class ComplianceReportResponse(BaseModel):
    """Response model for compliance reports"""
    id: UUID
    report_name: str
    framework: str
    compliance_score: Optional[float]
    total_findings: int
    critical_findings: int
    created_at: datetime
    status: str = "completed"


class RetentionPolicyRequest(BaseModel):
    """Request model for creating retention policies"""
    policy_name: str = Field(..., description="Name of the retention policy")
    policy_description: str = Field(..., description="Description of the policy")
    data_types: List[str] = Field(..., description="Data types covered by policy")
    retention_period_days: int = Field(..., gt=0, description="Retention period in days")
    user_roles: Optional[List[str]] = Field(None, description="User roles this applies to")
    archive_after_days: Optional[int] = Field(None, gt=0, description="Archive after days")
    auto_delete: bool = Field(default=False, description="Automatically delete after retention period")
    effective_date: Optional[datetime] = Field(None, description="When policy becomes effective")


class DataLineageResponse(BaseModel):
    """Response model for data lineage"""
    id: UUID
    data_id: str
    data_type: str
    data_name: str
    source_system: str
    classification: str
    retention_policy: str
    created_at: datetime
    consent_status: Optional[str]
    legal_basis: Optional[str]


class ComplianceDashboardResponse(BaseModel):
    """Response model for compliance dashboard"""
    compliance_reports: List[Dict[str, Any]]
    retention_policies: List[Dict[str, Any]]
    security_incidents: List[Dict[str, Any]]
    data_governance_metrics: Dict[str, Any]
    dashboard_generated_at: str
    reporting_period: Dict[str, Any]


@router.post("/reports", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Generate a compliance report for the specified framework and period"""
    
    try:
        report_id = await compliance_service.generate_compliance_report(
            report_type=request.report_type,
            framework=request.framework,
            period_start=request.period_start,
            period_end=request.period_end,
            generated_by=current_user.id,
            include_recommendations=request.include_recommendations
        )
        
        # Get the generated report details
        from ..database.connection import get_database_connection
        async with get_database_connection() as conn:
            report = await conn.fetchrow(
                "SELECT * FROM compliance_reports WHERE id = $1", report_id
            )
        
        if not report:
            raise HTTPException(status_code=404, detail="Generated report not found")
        
        # Count findings by severity
        findings = report['findings'] if report['findings'] else []
        critical_findings = sum(1 for f in findings if f.get('severity') == 'critical')
        
        return ComplianceReportResponse(
            id=report['id'],
            report_name=report['report_name'],
            framework=request.framework.value,
            compliance_score=report['compliance_score'],
            total_findings=len(findings),
            critical_findings=critical_findings,
            created_at=report['created_at']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate compliance report: {str(e)}")


@router.get("/reports", response_model=List[ComplianceReportResponse])
async def get_compliance_reports(
    framework: Optional[ComplianceFramework] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Get list of compliance reports"""
    
    try:
        from ..database.connection import get_database_connection
        
        conditions = []
        params = []
        param_count = 0
        
        if framework:
            param_count += 1
            conditions.append(f"${param_count} = ANY(regulations)")
            params.append(framework.value)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        query = f"""
            SELECT id, report_name, regulations, compliance_score, findings, created_at
            FROM compliance_reports
            {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param} OFFSET {offset_param}
        """
        
        async with get_database_connection() as conn:
            reports = await conn.fetch(query, *params)
        
        result = []
        for report in reports:
            findings = report['findings'] if report['findings'] else []
            critical_findings = sum(1 for f in findings if f.get('severity') == 'critical')
            
            result.append(ComplianceReportResponse(
                id=report['id'],
                report_name=report['report_name'],
                framework=report['regulations'][0] if report['regulations'] else 'unknown',
                compliance_score=report['compliance_score'],
                total_findings=len(findings),
                critical_findings=critical_findings,
                created_at=report['created_at']
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve compliance reports: {str(e)}")


@router.get("/reports/{report_id}")
async def get_compliance_report(
    report_id: UUID,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Get detailed compliance report"""
    
    try:
        from ..database.connection import get_database_connection
        
        async with get_database_connection() as conn:
            report = await conn.fetchrow(
                "SELECT * FROM compliance_reports WHERE id = $1", report_id
            )
        
        if not report:
            raise HTTPException(status_code=404, detail="Compliance report not found")
        
        return dict(report)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve compliance report: {str(e)}")


@router.post("/retention-policies", response_model=Dict[str, str])
async def create_retention_policy(
    request: RetentionPolicyRequest,
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.SYSTEM_ADMIN))
):
    """Create a new data retention policy"""
    
    try:
        policy_id = await compliance_service.create_retention_policy(
            policy_name=request.policy_name,
            policy_description=request.policy_description,
            data_types=request.data_types,
            retention_period_days=request.retention_period_days,
            approved_by=current_user.id,
            user_roles=request.user_roles,
            archive_after_days=request.archive_after_days,
            auto_delete=request.auto_delete,
            effective_date=request.effective_date
        )
        
        return {
            "message": "Retention policy created successfully",
            "policy_id": str(policy_id)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create retention policy: {str(e)}")


@router.get("/retention-policies")
async def get_retention_policies(
    active_only: bool = Query(True),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Get list of retention policies"""
    
    try:
        from ..database.connection import get_database_connection
        
        where_clause = "WHERE is_active = true" if active_only else ""
        
        async with get_database_connection() as conn:
            policies = await conn.fetch(f"""
                SELECT * FROM retention_policies
                {where_clause}
                ORDER BY created_at DESC
            """)
        
        return [dict(policy) for policy in policies]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve retention policies: {str(e)}")


@router.get("/data-lineage", response_model=List[DataLineageResponse])
async def get_data_lineage(
    data_type: Optional[str] = Query(None),
    source_system: Optional[str] = Query(None),
    classification: Optional[DataClassification] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Get data lineage records"""
    
    try:
        lineage_records = await compliance_service.get_data_lineage(
            data_type=data_type,
            source_system=source_system,
            classification=classification,
            limit=limit,
            offset=offset
        )
        
        return [DataLineageResponse(**record) for record in lineage_records]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data lineage: {str(e)}")


@router.post("/export")
async def export_compliance_data(
    export_type: str = Query(..., description="Type of export (regulatory_submission, audit_report)"),
    framework: ComplianceFramework = Query(..., description="Compliance framework"),
    period_start: datetime = Query(..., description="Start of export period"),
    period_end: datetime = Query(..., description="End of export period"),
    include_personal_data: bool = Query(False, description="Include personal data in export"),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Export compliance data for regulatory submissions"""
    
    try:
        export_data = await compliance_service.export_compliance_data(
            export_type=export_type,
            framework=framework,
            period_start=period_start,
            period_end=period_end,
            user_id=current_user.id,
            include_personal_data=include_personal_data
        )
        
        from fastapi.responses import JSONResponse
        filename = f"compliance_export_{framework.value}_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}.json"
        
        return JSONResponse(
            content=export_data,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export compliance data: {str(e)}")


@router.get("/dashboard", response_model=ComplianceDashboardResponse)
async def get_compliance_dashboard(
    framework: Optional[ComplianceFramework] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Get compliance dashboard data"""
    
    try:
        dashboard_data = await compliance_service.get_compliance_dashboard_data(
            framework=framework,
            days_back=days_back
        )
        
        return ComplianceDashboardResponse(**dashboard_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve compliance dashboard: {str(e)}")


@router.get("/frameworks")
async def get_compliance_frameworks(
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Get list of supported compliance frameworks"""
    
    frameworks = []
    for framework in ComplianceFramework:
        framework_info = compliance_service._compliance_frameworks.get(framework, {})
        frameworks.append({
            "code": framework.value,
            "name": framework_info.get("name", framework.value.upper()),
            "description": framework_info.get("description", ""),
            "requirements": framework_info.get("requirements", [])
        })
    
    return {"frameworks": frameworks}


@router.post("/frameworks/{framework}/assess")
async def assess_compliance(
    framework: ComplianceFramework,
    period_start: Optional[datetime] = Query(None),
    period_end: Optional[datetime] = Query(None),
    current_user=Depends(get_current_user),
    _=Depends(require_permission(UserPermission.COMPLIANCE_OFFICER))
):
    """Perform a quick compliance assessment for a framework"""
    
    # Default to last 30 days if no period specified
    if not period_start:
        period_start = datetime.utcnow() - timedelta(days=30)
    if not period_end:
        period_end = datetime.utcnow()
    
    try:
        # Gather metrics for assessment
        metrics = await compliance_service._gather_compliance_metrics(
            framework, period_start, period_end
        )
        
        # Analyze findings
        findings = await compliance_service._analyze_compliance_findings(framework, metrics)
        
        # Calculate compliance score
        compliance_score = await compliance_service._calculate_compliance_score(
            framework, metrics, findings
        )
        
        # Generate recommendations
        recommendations = await compliance_service._generate_compliance_recommendations(
            framework, findings
        )
        
        return {
            "framework": framework.value,
            "assessment_period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "compliance_score": compliance_score,
            "metrics": metrics,
            "findings": findings,
            "recommendations": recommendations,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assess compliance: {str(e)}")