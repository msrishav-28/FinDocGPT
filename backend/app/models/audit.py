"""
Audit trail and compliance data models
"""

from pydantic import Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin


class AuditEventType(str, Enum):
    """Types of audit events"""
    # User actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    PASSWORD_CHANGED = "password_changed"
    
    # Document actions
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_VIEWED = "document_viewed"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_ANALYZED = "document_analyzed"
    QUESTION_ASKED = "question_asked"
    
    # Analysis actions
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECAST_GENERATED = "forecast_generated"
    RECOMMENDATION_CREATED = "recommendation_created"
    
    # System actions
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    DATA_EXPORT = "data_export"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    
    # Security events
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"


class AuditSeverity(str, Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLog(UUIDMixin, TimestampMixin):
    """Comprehensive audit log entry"""
    
    # Event identification
    event_type: AuditEventType = Field(..., description="Type of audit event")
    severity: AuditSeverity = Field(default=AuditSeverity.MEDIUM)
    event_name: str = Field(..., description="Human-readable event name")
    description: str = Field(..., description="Detailed event description")
    
    # User context
    user_id: Optional[UUID] = Field(None, description="User who performed the action")
    username: Optional[str] = Field(None, description="Username for quick reference")
    user_role: Optional[str] = Field(None, description="User role at time of action")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    # Request context
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    endpoint: Optional[str] = Field(None, description="API endpoint accessed")
    http_method: Optional[str] = Field(None, description="HTTP method used")
    
    # Resource context
    resource_type: Optional[str] = Field(None, description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="ID of resource affected")
    resource_name: Optional[str] = Field(None, description="Name of resource affected")
    
    # Event data
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")
    before_state: Optional[Dict[str, Any]] = Field(None, description="State before change")
    after_state: Optional[Dict[str, Any]] = Field(None, description="State after change")
    
    # Outcome
    success: bool = Field(default=True, description="Whether the action succeeded")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Performance metrics
    duration_ms: Optional[int] = Field(None, ge=0, description="Action duration in milliseconds")
    response_size: Optional[int] = Field(None, ge=0, description="Response size in bytes")
    
    # Compliance fields
    compliance_tags: List[str] = Field(default_factory=list, description="Compliance-related tags")
    retention_period: Optional[int] = Field(None, description="Retention period in days")
    
    # Integrity protection
    checksum: Optional[str] = Field(None, description="Integrity checksum")
    signature: Optional[str] = Field(None, description="Digital signature")
    
    @validator('event_data', 'before_state', 'after_state')
    def sanitize_sensitive_data(cls, v):
        """Remove sensitive data from audit logs"""
        if not v:
            return v
        
        # List of sensitive field names to redact
        sensitive_fields = {
            'password', 'hashed_password', 'secret_key', 'api_key', 
            'token', 'access_token', 'refresh_token', 'private_key',
            'ssn', 'social_security_number', 'credit_card', 'bank_account'
        }
        
        def redact_dict(data):
            if isinstance(data, dict):
                return {
                    k: "[REDACTED]" if k.lower() in sensitive_fields 
                    else redact_dict(v) if isinstance(v, (dict, list)) 
                    else v
                    for k, v in data.items()
                }
            elif isinstance(data, list):
                return [redact_dict(item) for item in data]
            return data
        
        return redact_dict(v)


class ModelDecisionLog(UUIDMixin, TimestampMixin):
    """Log for AI/ML model decisions and explanations"""
    
    # Model identification
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of model (sentiment, forecast, etc.)")
    
    # Decision context
    user_id: Optional[UUID] = Field(None, description="User who triggered the decision")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    input_data: Dict[str, Any] = Field(..., description="Input data for the model")
    
    # Decision output
    prediction: Dict[str, Any] = Field(..., description="Model prediction/output")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    probability_distribution: Optional[Dict[str, float]] = Field(None, description="Probability distribution")
    
    # Explainability
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    decision_factors: List[str] = Field(default_factory=list, description="Key decision factors")
    
    # Model performance
    processing_time_ms: Optional[int] = Field(None, ge=0, description="Processing time")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage")
    
    # Validation
    ground_truth: Optional[Dict[str, Any]] = Field(None, description="Actual outcome if known")
    accuracy_score: Optional[float] = Field(None, ge=0, le=1, description="Accuracy if validated")
    
    # Compliance
    regulatory_flags: List[str] = Field(default_factory=list, description="Regulatory compliance flags")
    bias_metrics: Optional[Dict[str, float]] = Field(None, description="Bias detection metrics")


class DataLineage(UUIDMixin, TimestampMixin):
    """Track data lineage for compliance and governance"""
    
    # Data identification
    data_id: str = Field(..., description="Unique identifier for the data")
    data_type: str = Field(..., description="Type of data (document, market_data, etc.)")
    data_name: str = Field(..., description="Human-readable data name")
    
    # Source information
    source_system: str = Field(..., description="Source system or API")
    source_id: Optional[str] = Field(None, description="ID in source system")
    ingestion_method: str = Field(..., description="How data was ingested")
    
    # Processing history
    transformations: List[Dict[str, Any]] = Field(default_factory=list, description="Data transformations applied")
    quality_checks: List[Dict[str, Any]] = Field(default_factory=list, description="Quality checks performed")
    
    # Usage tracking
    accessed_by: List[UUID] = Field(default_factory=list, description="Users who accessed this data")
    used_in_models: List[str] = Field(default_factory=list, description="Models that used this data")
    derived_data: List[str] = Field(default_factory=list, description="Data derived from this data")
    
    # Governance
    classification: str = Field(..., description="Data classification (public, internal, confidential)")
    retention_policy: str = Field(..., description="Data retention policy")
    deletion_date: Optional[datetime] = Field(None, description="Scheduled deletion date")
    
    # Compliance
    consent_status: Optional[str] = Field(None, description="User consent status")
    legal_basis: Optional[str] = Field(None, description="Legal basis for processing")
    cross_border_transfers: List[str] = Field(default_factory=list, description="Countries data was transferred to")


class ComplianceReport(UUIDMixin, TimestampMixin):
    """Compliance reporting and audit summaries"""
    
    # Report identification
    report_type: str = Field(..., description="Type of compliance report")
    report_name: str = Field(..., description="Human-readable report name")
    reporting_period_start: datetime = Field(..., description="Start of reporting period")
    reporting_period_end: datetime = Field(..., description="End of reporting period")
    
    # Report content
    summary: str = Field(..., description="Executive summary")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Compliance findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    # Metrics
    total_events: int = Field(default=0, ge=0, description="Total audit events in period")
    security_incidents: int = Field(default=0, ge=0, description="Security incidents")
    data_breaches: int = Field(default=0, ge=0, description="Data breaches")
    policy_violations: int = Field(default=0, ge=0, description="Policy violations")
    
    # Regulatory compliance
    regulations: List[str] = Field(default_factory=list, description="Applicable regulations")
    compliance_score: Optional[float] = Field(None, ge=0, le=100, description="Overall compliance score")
    
    # Report metadata
    generated_by: UUID = Field(..., description="User who generated the report")
    approved_by: Optional[UUID] = Field(None, description="User who approved the report")
    approval_date: Optional[datetime] = Field(None, description="Report approval date")
    
    # Distribution
    recipients: List[str] = Field(default_factory=list, description="Report recipients")
    distribution_date: Optional[datetime] = Field(None, description="Report distribution date")


class RetentionPolicy(UUIDMixin, TimestampMixin):
    """Data retention and deletion policies"""
    
    # Policy identification
    policy_name: str = Field(..., description="Name of the retention policy")
    policy_description: str = Field(..., description="Description of the policy")
    
    # Scope
    data_types: List[str] = Field(..., description="Data types covered by this policy")
    user_roles: List[str] = Field(default_factory=list, description="User roles this applies to")
    
    # Retention rules
    retention_period_days: int = Field(..., gt=0, description="Retention period in days")
    archive_after_days: Optional[int] = Field(None, gt=0, description="Archive after days")
    
    # Deletion rules
    auto_delete: bool = Field(default=False, description="Automatically delete after retention period")
    deletion_method: str = Field(default="soft_delete", description="Deletion method")
    
    # Legal holds
    legal_hold_override: bool = Field(default=True, description="Legal holds override this policy")
    
    # Policy status
    is_active: bool = Field(default=True, description="Whether policy is active")
    effective_date: datetime = Field(..., description="When policy becomes effective")
    expiration_date: Optional[datetime] = Field(None, description="When policy expires")
    
    # Approval
    approved_by: UUID = Field(..., description="User who approved the policy")
    approval_date: datetime = Field(..., description="Policy approval date")