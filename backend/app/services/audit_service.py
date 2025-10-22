"""
Audit service for comprehensive logging and compliance tracking
"""

import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

from ..models.audit import (
    AuditLog, AuditEventType, AuditSeverity, ModelDecisionLog, 
    DataLineage, ComplianceReport, RetentionPolicy
)
from ..database.connection import get_database_connection
from ..config import get_settings

settings = get_settings()


class AuditService:
    """Service for audit logging and compliance tracking"""
    
    def __init__(self):
        self._db_pool = None
        self._integrity_key = settings.api.secret_key
    
    async def initialize(self):
        """Initialize the audit service"""
        # Create audit tables if they don't exist
        await self._ensure_audit_tables()
    
    async def _ensure_audit_tables(self):
        """Ensure audit tables exist in the database"""
        async with get_database_connection() as conn:
            # Create audit_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
                    event_name VARCHAR(200) NOT NULL,
                    description TEXT NOT NULL,
                    user_id UUID,
                    username VARCHAR(100),
                    user_role VARCHAR(50),
                    session_id VARCHAR(100),
                    ip_address INET,
                    user_agent TEXT,
                    request_id VARCHAR(100),
                    endpoint VARCHAR(200),
                    http_method VARCHAR(10),
                    resource_type VARCHAR(100),
                    resource_id VARCHAR(100),
                    resource_name VARCHAR(200),
                    event_data JSONB DEFAULT '{}',
                    before_state JSONB,
                    after_state JSONB,
                    success BOOLEAN DEFAULT true,
                    error_code VARCHAR(50),
                    error_message TEXT,
                    duration_ms INTEGER,
                    response_size INTEGER,
                    compliance_tags TEXT[] DEFAULT '{}',
                    retention_period INTEGER,
                    checksum VARCHAR(64),
                    signature VARCHAR(256),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                );
            """)
            
            # Create model_decision_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_decision_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    user_id UUID,
                    request_id VARCHAR(100),
                    input_data JSONB NOT NULL,
                    prediction JSONB NOT NULL,
                    confidence_score FLOAT,
                    probability_distribution JSONB,
                    feature_importance JSONB,
                    explanation TEXT,
                    decision_factors TEXT[] DEFAULT '{}',
                    processing_time_ms INTEGER,
                    memory_usage_mb FLOAT,
                    ground_truth JSONB,
                    accuracy_score FLOAT,
                    regulatory_flags TEXT[] DEFAULT '{}',
                    bias_metrics JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                );
            """)
            
            # Create data_lineage table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    data_id VARCHAR(100) NOT NULL UNIQUE,
                    data_type VARCHAR(50) NOT NULL,
                    data_name VARCHAR(200) NOT NULL,
                    source_system VARCHAR(100) NOT NULL,
                    source_id VARCHAR(100),
                    ingestion_method VARCHAR(50) NOT NULL,
                    transformations JSONB DEFAULT '[]',
                    quality_checks JSONB DEFAULT '[]',
                    accessed_by UUID[] DEFAULT '{}',
                    used_in_models TEXT[] DEFAULT '{}',
                    derived_data TEXT[] DEFAULT '{}',
                    classification VARCHAR(50) NOT NULL,
                    retention_policy VARCHAR(100) NOT NULL,
                    deletion_date TIMESTAMP WITH TIME ZONE,
                    consent_status VARCHAR(50),
                    legal_basis VARCHAR(100),
                    cross_border_transfers TEXT[] DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                );
            """)
            
            # Create compliance_reports table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    report_type VARCHAR(50) NOT NULL,
                    report_name VARCHAR(200) NOT NULL,
                    reporting_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    reporting_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    summary TEXT NOT NULL,
                    findings JSONB DEFAULT '[]',
                    recommendations TEXT[] DEFAULT '{}',
                    total_events INTEGER DEFAULT 0,
                    security_incidents INTEGER DEFAULT 0,
                    data_breaches INTEGER DEFAULT 0,
                    policy_violations INTEGER DEFAULT 0,
                    regulations TEXT[] DEFAULT '{}',
                    compliance_score FLOAT,
                    generated_by UUID NOT NULL,
                    approved_by UUID,
                    approval_date TIMESTAMP WITH TIME ZONE,
                    recipients TEXT[] DEFAULT '{}',
                    distribution_date TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                );
            """)
            
            # Create retention_policies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_policies (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    policy_name VARCHAR(100) NOT NULL UNIQUE,
                    policy_description TEXT NOT NULL,
                    data_types TEXT[] NOT NULL,
                    user_roles TEXT[] DEFAULT '{}',
                    retention_period_days INTEGER NOT NULL,
                    archive_after_days INTEGER,
                    auto_delete BOOLEAN DEFAULT false,
                    deletion_method VARCHAR(50) DEFAULT 'soft_delete',
                    legal_hold_override BOOLEAN DEFAULT true,
                    is_active BOOLEAN DEFAULT true,
                    effective_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    expiration_date TIMESTAMP WITH TIME ZONE,
                    approved_by UUID NOT NULL,
                    approval_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE
                );
            """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);")
            
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_logs_model_name ON model_decision_logs(model_name);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_logs_created_at ON model_decision_logs(created_at);")
            
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_data_lineage_data_id ON data_lineage(data_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_data_lineage_data_type ON data_lineage(data_type);")
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate integrity checksum for audit data"""
        # Create a deterministic string representation
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(f"{data_str}{self._integrity_key}".encode()).hexdigest()
    
    async def log_event(
        self,
        event_type: AuditEventType,
        event_name: str,
        description: str,
        user_id: Optional[UUID] = None,
        username: Optional[str] = None,
        user_role: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        http_method: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        response_size: Optional[int] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        compliance_tags: Optional[List[str]] = None,
        retention_period: Optional[int] = None
    ) -> UUID:
        """Log an audit event"""
        
        audit_log = AuditLog(
            event_type=event_type,
            severity=severity,
            event_name=event_name,
            description=description,
            user_id=user_id,
            username=username,
            user_role=user_role,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            endpoint=endpoint,
            http_method=http_method,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            event_data=event_data or {},
            before_state=before_state,
            after_state=after_state,
            success=success,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
            response_size=response_size,
            compliance_tags=compliance_tags or [],
            retention_period=retention_period
        )
        
        # Calculate integrity checksum
        checksum_data = {
            "event_type": event_type.value,
            "event_name": event_name,
            "description": description,
            "user_id": str(user_id) if user_id else None,
            "created_at": audit_log.created_at.isoformat(),
            "event_data": event_data or {}
        }
        audit_log.checksum = self._calculate_checksum(checksum_data)
        
        # Store in database
        async with get_database_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO audit_logs (
                    id, event_type, severity, event_name, description,
                    user_id, username, user_role, session_id, ip_address,
                    user_agent, request_id, endpoint, http_method,
                    resource_type, resource_id, resource_name,
                    event_data, before_state, after_state,
                    success, error_code, error_message,
                    duration_ms, response_size, compliance_tags,
                    retention_period, checksum, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26, $27, $28, $29
                ) RETURNING id
            """,
                audit_log.id, audit_log.event_type.value, audit_log.severity.value,
                audit_log.event_name, audit_log.description,
                audit_log.user_id, audit_log.username, audit_log.user_role,
                audit_log.session_id, audit_log.ip_address, audit_log.user_agent,
                audit_log.request_id, audit_log.endpoint, audit_log.http_method,
                audit_log.resource_type, audit_log.resource_id, audit_log.resource_name,
                json.dumps(audit_log.event_data), 
                json.dumps(audit_log.before_state) if audit_log.before_state else None,
                json.dumps(audit_log.after_state) if audit_log.after_state else None,
                audit_log.success, audit_log.error_code, audit_log.error_message,
                audit_log.duration_ms, audit_log.response_size, audit_log.compliance_tags,
                audit_log.retention_period, audit_log.checksum, audit_log.created_at
            )
        
        return result['id']
    
    async def log_model_decision(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        user_id: Optional[UUID] = None,
        request_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
        probability_distribution: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        explanation: Optional[str] = None,
        decision_factors: Optional[List[str]] = None,
        processing_time_ms: Optional[int] = None,
        memory_usage_mb: Optional[float] = None,
        regulatory_flags: Optional[List[str]] = None,
        bias_metrics: Optional[Dict[str, float]] = None
    ) -> UUID:
        """Log a model decision for explainability and audit"""
        
        decision_log = ModelDecisionLog(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            user_id=user_id,
            request_id=request_id,
            input_data=input_data,
            prediction=prediction,
            confidence_score=confidence_score,
            probability_distribution=probability_distribution,
            feature_importance=feature_importance,
            explanation=explanation,
            decision_factors=decision_factors or [],
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            regulatory_flags=regulatory_flags or [],
            bias_metrics=bias_metrics
        )
        
        async with get_database_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO model_decision_logs (
                    id, model_name, model_version, model_type, user_id, request_id,
                    input_data, prediction, confidence_score, probability_distribution,
                    feature_importance, explanation, decision_factors,
                    processing_time_ms, memory_usage_mb, regulatory_flags,
                    bias_metrics, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18
                ) RETURNING id
            """,
                decision_log.id, decision_log.model_name, decision_log.model_version,
                decision_log.model_type, decision_log.user_id, decision_log.request_id,
                json.dumps(decision_log.input_data), json.dumps(decision_log.prediction),
                decision_log.confidence_score, 
                json.dumps(decision_log.probability_distribution) if decision_log.probability_distribution else None,
                json.dumps(decision_log.feature_importance) if decision_log.feature_importance else None,
                decision_log.explanation, decision_log.decision_factors,
                decision_log.processing_time_ms, decision_log.memory_usage_mb,
                decision_log.regulatory_flags,
                json.dumps(decision_log.bias_metrics) if decision_log.bias_metrics else None,
                decision_log.created_at
            )
        
        return result['id']
    
    async def track_data_lineage(
        self,
        data_id: str,
        data_type: str,
        data_name: str,
        source_system: str,
        ingestion_method: str,
        classification: str,
        retention_policy: str,
        source_id: Optional[str] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
        quality_checks: Optional[List[Dict[str, Any]]] = None,
        consent_status: Optional[str] = None,
        legal_basis: Optional[str] = None
    ) -> UUID:
        """Track data lineage for governance and compliance"""
        
        lineage = DataLineage(
            data_id=data_id,
            data_type=data_type,
            data_name=data_name,
            source_system=source_system,
            source_id=source_id,
            ingestion_method=ingestion_method,
            transformations=transformations or [],
            quality_checks=quality_checks or [],
            classification=classification,
            retention_policy=retention_policy,
            consent_status=consent_status,
            legal_basis=legal_basis
        )
        
        async with get_database_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO data_lineage (
                    id, data_id, data_type, data_name, source_system, source_id,
                    ingestion_method, transformations, quality_checks,
                    classification, retention_policy, consent_status, legal_basis,
                    created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                ) RETURNING id
                ON CONFLICT (data_id) DO UPDATE SET
                    transformations = EXCLUDED.transformations,
                    quality_checks = EXCLUDED.quality_checks,
                    updated_at = NOW()
            """,
                lineage.id, lineage.data_id, lineage.data_type, lineage.data_name,
                lineage.source_system, lineage.source_id, lineage.ingestion_method,
                json.dumps(lineage.transformations), json.dumps(lineage.quality_checks),
                lineage.classification, lineage.retention_policy,
                lineage.consent_status, lineage.legal_basis, lineage.created_at
            )
        
        return result['id']
    
    async def get_audit_logs(
        self,
        user_id: Optional[UUID] = None,
        event_type: Optional[AuditEventType] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        success: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        
        conditions = []
        params = []
        param_count = 0
        
        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
        
        if event_type:
            param_count += 1
            conditions.append(f"event_type = ${param_count}")
            params.append(event_type.value)
        
        if resource_type:
            param_count += 1
            conditions.append(f"resource_type = ${param_count}")
            params.append(resource_type)
        
        if resource_id:
            param_count += 1
            conditions.append(f"resource_id = ${param_count}")
            params.append(resource_id)
        
        if start_date:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(start_date)
        
        if end_date:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(end_date)
        
        if severity:
            param_count += 1
            conditions.append(f"severity = ${param_count}")
            params.append(severity.value)
        
        if success is not None:
            param_count += 1
            conditions.append(f"success = ${param_count}")
            params.append(success)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        query = f"""
            SELECT * FROM audit_logs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param} OFFSET {offset_param}
        """
        
        async with get_database_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def verify_audit_integrity(self, audit_id: UUID) -> bool:
        """Verify the integrity of an audit log entry"""
        async with get_database_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM audit_logs WHERE id = $1", audit_id
            )
            
            if not row:
                return False
            
            # Recalculate checksum
            checksum_data = {
                "event_type": row['event_type'],
                "event_name": row['event_name'],
                "description": row['description'],
                "user_id": str(row['user_id']) if row['user_id'] else None,
                "created_at": row['created_at'].isoformat(),
                "event_data": row['event_data'] or {}
            }
            
            expected_checksum = self._calculate_checksum(checksum_data)
            return expected_checksum == row['checksum']
    
    async def apply_retention_policies(self):
        """Apply data retention policies and clean up old data"""
        async with get_database_connection() as conn:
            # Get active retention policies
            policies = await conn.fetch("""
                SELECT * FROM retention_policies 
                WHERE is_active = true AND effective_date <= NOW()
                AND (expiration_date IS NULL OR expiration_date > NOW())
            """)
            
            for policy in policies:
                cutoff_date = datetime.utcnow() - timedelta(days=policy['retention_period_days'])
                
                if policy['auto_delete']:
                    # Hard delete old audit logs
                    deleted_count = await conn.fetchval("""
                        DELETE FROM audit_logs 
                        WHERE created_at < $1 
                        AND NOT EXISTS (
                            SELECT 1 FROM audit_logs al2 
                            WHERE al2.id = audit_logs.id 
                            AND 'legal_hold' = ANY(al2.compliance_tags)
                        )
                    """, cutoff_date)
                    
                    if deleted_count > 0:
                        await self.log_event(
                            event_type=AuditEventType.SYSTEM_CONFIG_CHANGED,
                            event_name="Retention Policy Applied",
                            description=f"Deleted {deleted_count} audit logs per retention policy '{policy['policy_name']}'",
                            event_data={"policy_id": str(policy['id']), "deleted_count": deleted_count}
                        )


# Global audit service instance
audit_service = AuditService()


@asynccontextmanager
async def audit_context(
    event_type: AuditEventType,
    event_name: str,
    description: str,
    user_id: Optional[UUID] = None,
    username: Optional[str] = None,
    user_role: Optional[str] = None,
    session_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    event_data: Optional[Dict[str, Any]] = None,
    severity: AuditSeverity = AuditSeverity.MEDIUM
):
    """Context manager for automatic audit logging with timing"""
    start_time = datetime.utcnow()
    success = True
    error_code = None
    error_message = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_code = type(e).__name__
        error_message = str(e)
        raise
    finally:
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        await audit_service.log_event(
            event_type=event_type,
            event_name=event_name,
            description=description,
            user_id=user_id,
            username=username,
            user_role=user_role,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            endpoint=endpoint,
            http_method=http_method,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            event_data=event_data,
            success=success,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
            severity=severity
        )