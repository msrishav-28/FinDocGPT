"""
Compliance reporting and data governance service
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from ..models.audit import (
    ComplianceReport, RetentionPolicy, DataLineage, AuditLog, 
    AuditEventType, AuditSeverity
)
from ..database.connection import get_database_connection
from ..config import get_settings
from .audit_service import audit_service

settings = get_settings()


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    SOX = "sox"  # Sarbanes-Oxley Act
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    SEC = "sec"  # Securities and Exchange Commission
    FINRA = "finra"  # Financial Industry Regulatory Authority
    MiFID_II = "mifid_ii"  # Markets in Financial Instruments Directive II
    BASEL_III = "basel_iii"  # Basel III banking regulations


class DataClassification(str, Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ComplianceService:
    """Service for compliance reporting and data governance"""
    
    def __init__(self):
        self._compliance_frameworks = {
            ComplianceFramework.SOX: {
                "name": "Sarbanes-Oxley Act",
                "description": "Financial reporting and corporate governance",
                "requirements": [
                    "Accurate financial reporting",
                    "Internal controls assessment",
                    "Executive certification",
                    "Audit trail maintenance"
                ]
            },
            ComplianceFramework.GDPR: {
                "name": "General Data Protection Regulation",
                "description": "Data protection and privacy",
                "requirements": [
                    "Data subject consent",
                    "Right to be forgotten",
                    "Data portability",
                    "Privacy by design"
                ]
            },
            ComplianceFramework.SEC: {
                "name": "Securities and Exchange Commission",
                "description": "Securities regulation and investor protection",
                "requirements": [
                    "Investment advisor registration",
                    "Fiduciary duty compliance",
                    "Record keeping requirements",
                    "Client disclosure"
                ]
            },
            ComplianceFramework.FINRA: {
                "name": "Financial Industry Regulatory Authority",
                "description": "Broker-dealer regulation",
                "requirements": [
                    "Know your customer (KYC)",
                    "Anti-money laundering (AML)",
                    "Best execution",
                    "Supervision and surveillance"
                ]
            }
        }
    
    async def initialize(self):
        """Initialize the compliance service"""
        await self._ensure_compliance_tables()
        await self._create_default_retention_policies()
    
    async def _ensure_compliance_tables(self):
        """Ensure compliance-related tables exist"""
        # Tables are already created in audit_service.py
        pass
    
    async def _create_default_retention_policies(self):
        """Create default retention policies if they don't exist"""
        default_policies = [
            {
                "policy_name": "Financial_Data_Retention",
                "policy_description": "Retention policy for financial data and reports",
                "data_types": ["financial_report", "market_data", "forecast", "recommendation"],
                "retention_period_days": 2555,  # 7 years
                "archive_after_days": 1095,  # 3 years
                "auto_delete": False,
                "legal_hold_override": True
            },
            {
                "policy_name": "User_Activity_Logs",
                "policy_description": "Retention policy for user activity and audit logs",
                "data_types": ["audit_log", "user_action", "system_event"],
                "retention_period_days": 2190,  # 6 years
                "archive_after_days": 730,  # 2 years
                "auto_delete": False,
                "legal_hold_override": True
            },
            {
                "policy_name": "Model_Decision_Logs",
                "policy_description": "Retention policy for AI/ML model decisions and explanations",
                "data_types": ["model_decision", "prediction", "recommendation_reasoning"],
                "retention_period_days": 1825,  # 5 years
                "archive_after_days": 365,  # 1 year
                "auto_delete": False,
                "legal_hold_override": True
            },
            {
                "policy_name": "Personal_Data_GDPR",
                "policy_description": "GDPR compliance retention policy for personal data",
                "data_types": ["personal_data", "user_profile", "consent_record"],
                "retention_period_days": 1095,  # 3 years
                "archive_after_days": 365,  # 1 year
                "auto_delete": True,
                "legal_hold_override": True
            }
        ]
        
        async with get_database_connection() as conn:
            for policy_data in default_policies:
                # Check if policy already exists
                existing = await conn.fetchrow(
                    "SELECT id FROM retention_policies WHERE policy_name = $1",
                    policy_data["policy_name"]
                )
                
                if not existing:
                    # Create system user ID for default policies
                    system_user_id = uuid4()
                    
                    await conn.execute("""
                        INSERT INTO retention_policies (
                            id, policy_name, policy_description, data_types,
                            retention_period_days, archive_after_days, auto_delete,
                            legal_hold_override, is_active, effective_date,
                            approved_by, approval_date
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                        )
                    """,
                        uuid4(), policy_data["policy_name"], policy_data["policy_description"],
                        policy_data["data_types"], policy_data["retention_period_days"],
                        policy_data["archive_after_days"], policy_data["auto_delete"],
                        policy_data["legal_hold_override"], True, datetime.utcnow(),
                        system_user_id, datetime.utcnow()
                    )
    
    async def generate_compliance_report(
        self,
        report_type: str,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        generated_by: UUID,
        include_recommendations: bool = True
    ) -> UUID:
        """Generate a comprehensive compliance report"""
        
        report_name = f"{framework.value.upper()} Compliance Report - {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}"
        
        # Gather compliance metrics
        metrics = await self._gather_compliance_metrics(framework, period_start, period_end)
        
        # Generate findings and recommendations
        findings = await self._analyze_compliance_findings(framework, metrics)
        recommendations = await self._generate_compliance_recommendations(framework, findings) if include_recommendations else []
        
        # Calculate compliance score
        compliance_score = await self._calculate_compliance_score(framework, metrics, findings)
        
        # Create compliance report
        report = ComplianceReport(
            report_type=report_type,
            report_name=report_name,
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            summary=await self._generate_executive_summary(framework, metrics, findings),
            findings=findings,
            recommendations=recommendations,
            total_events=metrics.get("total_events", 0),
            security_incidents=metrics.get("security_incidents", 0),
            data_breaches=metrics.get("data_breaches", 0),
            policy_violations=metrics.get("policy_violations", 0),
            regulations=[framework.value],
            compliance_score=compliance_score,
            generated_by=generated_by
        )
        
        # Store report in database
        async with get_database_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO compliance_reports (
                    id, report_type, report_name, reporting_period_start, reporting_period_end,
                    summary, findings, recommendations, total_events, security_incidents,
                    data_breaches, policy_violations, regulations, compliance_score,
                    generated_by, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                ) RETURNING id
            """,
                report.id, report.report_type, report.report_name,
                report.reporting_period_start, report.reporting_period_end,
                report.summary, json.dumps(report.findings), report.recommendations,
                report.total_events, report.security_incidents, report.data_breaches,
                report.policy_violations, report.regulations, report.compliance_score,
                report.generated_by, report.created_at
            )
        
        # Log report generation
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            event_name="Compliance Report Generated",
            description=f"Generated {framework.value.upper()} compliance report for period {period_start} to {period_end}",
            user_id=generated_by,
            resource_type="compliance_report",
            resource_id=str(report.id),
            resource_name=report_name,
            event_data={
                "framework": framework.value,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "compliance_score": compliance_score
            },
            compliance_tags=[framework.value, "compliance_report"]
        )
        
        return result['id']
    
    async def _gather_compliance_metrics(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Gather metrics relevant to the compliance framework"""
        
        async with get_database_connection() as conn:
            # Basic audit metrics
            total_events = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_logs 
                WHERE created_at BETWEEN $1 AND $2
            """, period_start, period_end)
            
            security_incidents = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_logs 
                WHERE created_at BETWEEN $1 AND $2
                AND event_type IN ('unauthorized_access', 'permission_denied', 'suspicious_activity')
            """, period_start, period_end)
            
            data_breaches = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_logs 
                WHERE created_at BETWEEN $1 AND $2
                AND severity = 'critical'
                AND event_type IN ('data_export', 'unauthorized_access')
            """, period_start, period_end)
            
            policy_violations = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_logs 
                WHERE created_at BETWEEN $1 AND $2
                AND success = false
                AND error_code IN ('POLICY_VIOLATION', 'COMPLIANCE_ERROR')
            """, period_start, period_end)
            
            # Framework-specific metrics
            framework_metrics = {}
            
            if framework == ComplianceFramework.SOX:
                # SOX-specific metrics
                financial_reports_processed = await conn.fetchval("""
                    SELECT COUNT(*) FROM audit_logs 
                    WHERE created_at BETWEEN $1 AND $2
                    AND resource_type = 'financial_report'
                    AND event_type = 'document_analyzed'
                """, period_start, period_end)
                
                model_decisions_logged = await conn.fetchval("""
                    SELECT COUNT(*) FROM model_decision_logs 
                    WHERE created_at BETWEEN $1 AND $2
                """, period_start, period_end)
                
                framework_metrics.update({
                    "financial_reports_processed": financial_reports_processed,
                    "model_decisions_logged": model_decisions_logged
                })
            
            elif framework == ComplianceFramework.GDPR:
                # GDPR-specific metrics
                personal_data_processed = await conn.fetchval("""
                    SELECT COUNT(*) FROM data_lineage 
                    WHERE created_at BETWEEN $1 AND $2
                    AND classification IN ('confidential', 'restricted')
                """, period_start, period_end)
                
                consent_records = await conn.fetchval("""
                    SELECT COUNT(*) FROM data_lineage 
                    WHERE created_at BETWEEN $1 AND $2
                    AND consent_status IS NOT NULL
                """, period_start, period_end)
                
                framework_metrics.update({
                    "personal_data_processed": personal_data_processed,
                    "consent_records": consent_records
                })
            
            elif framework == ComplianceFramework.SEC:
                # SEC-specific metrics
                investment_recommendations = await conn.fetchval("""
                    SELECT COUNT(*) FROM model_decision_logs 
                    WHERE created_at BETWEEN $1 AND $2
                    AND model_type = 'investment_advisory'
                """, period_start, period_end)
                
                client_disclosures = await conn.fetchval("""
                    SELECT COUNT(*) FROM audit_logs 
                    WHERE created_at BETWEEN $1 AND $2
                    AND event_type = 'recommendation_created'
                """, period_start, period_end)
                
                framework_metrics.update({
                    "investment_recommendations": investment_recommendations,
                    "client_disclosures": client_disclosures
                })
        
        return {
            "total_events": total_events,
            "security_incidents": security_incidents,
            "data_breaches": data_breaches,
            "policy_violations": policy_violations,
            **framework_metrics
        }
    
    async def _analyze_compliance_findings(
        self,
        framework: ComplianceFramework,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze compliance metrics and generate findings"""
        
        findings = []
        
        # Common findings
        if metrics.get("security_incidents", 0) > 0:
            findings.append({
                "category": "Security",
                "severity": "high" if metrics["security_incidents"] > 10 else "medium",
                "finding": f"Detected {metrics['security_incidents']} security incidents during reporting period",
                "impact": "Potential unauthorized access to sensitive financial data",
                "evidence": f"Audit logs show {metrics['security_incidents']} security-related events",
                "recommendation": "Review security controls and implement additional monitoring"
            })
        
        if metrics.get("data_breaches", 0) > 0:
            findings.append({
                "category": "Data Protection",
                "severity": "critical",
                "finding": f"Identified {metrics['data_breaches']} potential data breaches",
                "impact": "Possible exposure of confidential financial information",
                "evidence": f"Critical severity events in audit logs: {metrics['data_breaches']}",
                "recommendation": "Immediate investigation and breach notification procedures"
            })
        
        if metrics.get("policy_violations", 0) > 0:
            findings.append({
                "category": "Policy Compliance",
                "severity": "medium",
                "finding": f"Found {metrics['policy_violations']} policy violations",
                "impact": "Non-compliance with internal governance policies",
                "evidence": f"Failed operations with policy violation errors: {metrics['policy_violations']}",
                "recommendation": "Review and strengthen policy enforcement mechanisms"
            })
        
        # Framework-specific findings
        if framework == ComplianceFramework.SOX:
            if metrics.get("model_decisions_logged", 0) < metrics.get("investment_recommendations", 0):
                findings.append({
                    "category": "Internal Controls",
                    "severity": "high",
                    "finding": "Incomplete audit trail for investment recommendations",
                    "impact": "Inability to demonstrate proper internal controls over financial reporting",
                    "evidence": "Model decision logs do not cover all investment recommendations",
                    "recommendation": "Ensure all AI/ML decisions are properly logged and auditable"
                })
        
        elif framework == ComplianceFramework.GDPR:
            if metrics.get("consent_records", 0) < metrics.get("personal_data_processed", 0):
                findings.append({
                    "category": "Data Protection",
                    "severity": "high",
                    "finding": "Personal data processed without documented consent",
                    "impact": "Potential GDPR violation and regulatory penalties",
                    "evidence": f"Personal data records: {metrics.get('personal_data_processed', 0)}, Consent records: {metrics.get('consent_records', 0)}",
                    "recommendation": "Implement consent management system and update data processing procedures"
                })
        
        elif framework == ComplianceFramework.SEC:
            if metrics.get("client_disclosures", 0) < metrics.get("investment_recommendations", 0):
                findings.append({
                    "category": "Client Disclosure",
                    "severity": "high",
                    "finding": "Investment recommendations made without proper client disclosure",
                    "impact": "Violation of SEC fiduciary duty requirements",
                    "evidence": f"Recommendations: {metrics.get('investment_recommendations', 0)}, Disclosures: {metrics.get('client_disclosures', 0)}",
                    "recommendation": "Ensure all investment advice includes proper risk disclosure and conflicts of interest"
                })
        
        return findings
    
    async def _generate_compliance_recommendations(
        self,
        framework: ComplianceFramework,
        findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable compliance recommendations"""
        
        recommendations = []
        
        # Extract recommendations from findings
        for finding in findings:
            if finding.get("recommendation"):
                recommendations.append(finding["recommendation"])
        
        # Add framework-specific recommendations
        framework_info = self._compliance_frameworks.get(framework, {})
        requirements = framework_info.get("requirements", [])
        
        for requirement in requirements:
            if framework == ComplianceFramework.SOX and "Internal controls" in requirement:
                recommendations.append("Implement automated controls testing for AI/ML model decisions")
                recommendations.append("Establish quarterly management assessment of internal controls")
            
            elif framework == ComplianceFramework.GDPR and "Data subject" in requirement:
                recommendations.append("Implement data subject rights management system")
                recommendations.append("Establish data protection impact assessment procedures")
            
            elif framework == ComplianceFramework.SEC and "Record keeping" in requirement:
                recommendations.append("Enhance record retention policies for investment advisory activities")
                recommendations.append("Implement client communication archival system")
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    async def _calculate_compliance_score(
        self,
        framework: ComplianceFramework,
        metrics: Dict[str, Any],
        findings: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall compliance score (0-100)"""
        
        base_score = 100.0
        
        # Deduct points for findings based on severity
        for finding in findings:
            severity = finding.get("severity", "low")
            if severity == "critical":
                base_score -= 25.0
            elif severity == "high":
                base_score -= 15.0
            elif severity == "medium":
                base_score -= 10.0
            elif severity == "low":
                base_score -= 5.0
        
        # Additional deductions for specific metrics
        if metrics.get("data_breaches", 0) > 0:
            base_score -= 20.0  # Significant penalty for data breaches
        
        if metrics.get("security_incidents", 0) > 10:
            base_score -= 10.0  # Penalty for high number of security incidents
        
        # Framework-specific scoring adjustments
        if framework == ComplianceFramework.SOX:
            # SOX requires strong internal controls
            if metrics.get("model_decisions_logged", 0) == 0:
                base_score -= 15.0
        
        elif framework == ComplianceFramework.GDPR:
            # GDPR requires proper consent management
            personal_data = metrics.get("personal_data_processed", 0)
            consent_records = metrics.get("consent_records", 0)
            if personal_data > 0 and consent_records == 0:
                base_score -= 20.0
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, base_score))
    
    async def _generate_executive_summary(
        self,
        framework: ComplianceFramework,
        metrics: Dict[str, Any],
        findings: List[Dict[str, Any]]
    ) -> str:
        """Generate executive summary for compliance report"""
        
        framework_info = self._compliance_frameworks.get(framework, {})
        framework_name = framework_info.get("name", framework.value.upper())
        
        critical_findings = [f for f in findings if f.get("severity") == "critical"]
        high_findings = [f for f in findings if f.get("severity") == "high"]
        
        summary = f"""
Executive Summary - {framework_name} Compliance Assessment

This report provides a comprehensive assessment of compliance with {framework_name} requirements during the specified reporting period.

Key Metrics:
- Total audit events processed: {metrics.get('total_events', 0):,}
- Security incidents detected: {metrics.get('security_incidents', 0)}
- Data breaches identified: {metrics.get('data_breaches', 0)}
- Policy violations recorded: {metrics.get('policy_violations', 0)}

Compliance Status:
- Total findings identified: {len(findings)}
- Critical issues requiring immediate attention: {len(critical_findings)}
- High-priority issues for remediation: {len(high_findings)}

Overall Assessment:
The organization demonstrates {"strong" if len(critical_findings) == 0 else "moderate" if len(critical_findings) < 3 else "weak"} compliance with {framework_name} requirements. 
{"No critical issues were identified during this assessment period." if len(critical_findings) == 0 else f"{len(critical_findings)} critical issues require immediate management attention and remediation."}

{"The compliance program appears to be operating effectively with minor areas for improvement." if len(findings) <= 3 else "Several compliance gaps have been identified that require systematic remediation efforts."}
        """.strip()
        
        return summary
    
    async def create_retention_policy(
        self,
        policy_name: str,
        policy_description: str,
        data_types: List[str],
        retention_period_days: int,
        approved_by: UUID,
        user_roles: Optional[List[str]] = None,
        archive_after_days: Optional[int] = None,
        auto_delete: bool = False,
        effective_date: Optional[datetime] = None
    ) -> UUID:
        """Create a new data retention policy"""
        
        policy = RetentionPolicy(
            policy_name=policy_name,
            policy_description=policy_description,
            data_types=data_types,
            user_roles=user_roles or [],
            retention_period_days=retention_period_days,
            archive_after_days=archive_after_days,
            auto_delete=auto_delete,
            effective_date=effective_date or datetime.utcnow(),
            approved_by=approved_by,
            approval_date=datetime.utcnow()
        )
        
        async with get_database_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO retention_policies (
                    id, policy_name, policy_description, data_types, user_roles,
                    retention_period_days, archive_after_days, auto_delete,
                    legal_hold_override, is_active, effective_date,
                    approved_by, approval_date, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                ) RETURNING id
            """,
                policy.id, policy.policy_name, policy.policy_description,
                policy.data_types, policy.user_roles, policy.retention_period_days,
                policy.archive_after_days, policy.auto_delete, policy.legal_hold_override,
                policy.is_active, policy.effective_date, policy.approved_by,
                policy.approval_date, policy.created_at
            )
        
        # Log policy creation
        await audit_service.log_event(
            event_type=AuditEventType.SYSTEM_CONFIG_CHANGED,
            event_name="Retention Policy Created",
            description=f"Created new retention policy: {policy_name}",
            user_id=approved_by,
            resource_type="retention_policy",
            resource_id=str(policy.id),
            resource_name=policy_name,
            event_data={
                "data_types": data_types,
                "retention_period_days": retention_period_days,
                "auto_delete": auto_delete
            },
            compliance_tags=["data_governance", "retention_policy"]
        )
        
        return result['id']
    
    async def get_data_lineage(
        self,
        data_id: Optional[str] = None,
        data_type: Optional[str] = None,
        source_system: Optional[str] = None,
        classification: Optional[DataClassification] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query data lineage records"""
        
        conditions = []
        params = []
        param_count = 0
        
        if data_id:
            param_count += 1
            conditions.append(f"data_id = ${param_count}")
            params.append(data_id)
        
        if data_type:
            param_count += 1
            conditions.append(f"data_type = ${param_count}")
            params.append(data_type)
        
        if source_system:
            param_count += 1
            conditions.append(f"source_system = ${param_count}")
            params.append(source_system)
        
        if classification:
            param_count += 1
            conditions.append(f"classification = ${param_count}")
            params.append(classification.value)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        query = f"""
            SELECT * FROM data_lineage
            {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param} OFFSET {offset_param}
        """
        
        async with get_database_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def export_compliance_data(
        self,
        export_type: str,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        user_id: UUID,
        include_personal_data: bool = False
    ) -> Dict[str, Any]:
        """Export compliance data for regulatory submissions"""
        
        export_data = {
            "export_metadata": {
                "export_type": export_type,
                "framework": framework.value,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "exported_by": str(user_id),
                "export_timestamp": datetime.utcnow().isoformat(),
                "include_personal_data": include_personal_data
            }
        }
        
        async with get_database_connection() as conn:
            # Export audit logs
            audit_logs = await conn.fetch("""
                SELECT event_type, event_name, description, user_id, created_at,
                       resource_type, resource_id, success, compliance_tags
                FROM audit_logs
                WHERE created_at BETWEEN $1 AND $2
                AND ($3 = true OR 'personal_data' != ANY(compliance_tags))
                ORDER BY created_at
            """, period_start, period_end, include_personal_data)
            
            export_data["audit_logs"] = [dict(row) for row in audit_logs]
            
            # Export model decisions
            model_decisions = await conn.fetch("""
                SELECT model_name, model_type, prediction, confidence_score,
                       explanation, regulatory_flags, created_at
                FROM model_decision_logs
                WHERE created_at BETWEEN $1 AND $2
                ORDER BY created_at
            """, period_start, period_end)
            
            export_data["model_decisions"] = [dict(row) for row in model_decisions]
            
            # Export data lineage (if including personal data)
            if include_personal_data:
                data_lineage = await conn.fetch("""
                    SELECT data_id, data_type, source_system, classification,
                           consent_status, legal_basis, created_at
                    FROM data_lineage
                    WHERE created_at BETWEEN $1 AND $2
                    ORDER BY created_at
                """, period_start, period_end)
                
                export_data["data_lineage"] = [dict(row) for row in data_lineage]
        
        # Log data export
        await audit_service.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            event_name="Compliance Data Export",
            description=f"Exported compliance data for {framework.value.upper()} framework",
            user_id=user_id,
            event_data={
                "export_type": export_type,
                "framework": framework.value,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "records_exported": len(export_data.get("audit_logs", [])) + len(export_data.get("model_decisions", []))
            },
            compliance_tags=[framework.value, "data_export", "regulatory_submission"],
            severity=AuditSeverity.HIGH
        )
        
        return export_data
    
    async def get_compliance_dashboard_data(
        self,
        framework: Optional[ComplianceFramework] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get compliance dashboard data for monitoring"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        async with get_database_connection() as conn:
            # Recent compliance reports
            reports = await conn.fetch("""
                SELECT id, report_name, compliance_score, created_at, regulations
                FROM compliance_reports
                WHERE created_at >= $1
                ORDER BY created_at DESC
                LIMIT 10
            """, start_date)
            
            # Active retention policies
            policies = await conn.fetch("""
                SELECT policy_name, data_types, retention_period_days, is_active
                FROM retention_policies
                WHERE is_active = true
                ORDER BY created_at DESC
            """)
            
            # Recent security incidents
            incidents = await conn.fetch("""
                SELECT event_name, severity, created_at, event_data
                FROM audit_logs
                WHERE created_at >= $1
                AND event_type IN ('unauthorized_access', 'permission_denied', 'suspicious_activity')
                ORDER BY created_at DESC
                LIMIT 20
            """, start_date)
            
            # Data governance metrics
            data_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_data_records,
                    COUNT(CASE WHEN classification = 'confidential' THEN 1 END) as confidential_records,
                    COUNT(CASE WHEN classification = 'restricted' THEN 1 END) as restricted_records,
                    COUNT(CASE WHEN consent_status IS NOT NULL THEN 1 END) as consent_tracked_records
                FROM data_lineage
                WHERE created_at >= $1
            """, start_date)
        
        return {
            "compliance_reports": [dict(row) for row in reports],
            "retention_policies": [dict(row) for row in policies],
            "security_incidents": [dict(row) for row in incidents],
            "data_governance_metrics": dict(data_metrics) if data_metrics else {},
            "dashboard_generated_at": datetime.utcnow().isoformat(),
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days_back
            }
        }


# Global compliance service instance
compliance_service = ComplianceService()