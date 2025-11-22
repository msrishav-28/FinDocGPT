"""
Document-related data models
"""

from pydantic import Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from .base import BaseModel, TimestampMixin, UUIDMixin


class DocumentType(str, Enum):
    """Types of financial documents"""
    EARNINGS_REPORT = "earnings_report"
    SEC_FILING = "sec_filing"
    PRESS_RELEASE = "press_release"
    ANALYST_REPORT = "analyst_report"
    EARNINGS_CALL = "earnings_call"
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"


class DocumentMetadata(BaseModel):
    """Metadata for financial documents"""
    company: str = Field(..., description="Company ticker or identifier")
    document_type: DocumentType
    filing_date: datetime
    period: Optional[str] = Field(None, description="Reporting period (e.g., Q1 2024, FY 2023)")
    source: str = Field(..., description="Source of the document")
    language: str = Field(default="en", description="Document language")
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    
    @validator('company')
    def validate_company(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Company identifier cannot be empty")
        return v.upper().strip()


class DocumentContent(BaseModel):
    """Document content and processing information"""
    raw_content: str = Field(..., description="Raw document text")
    processed_content: Optional[str] = Field(None, description="Cleaned and processed text")
    entities: Optional[Dict[str, List[str]]] = Field(None, description="Extracted entities")
    key_metrics: Optional[Dict[str, Any]] = Field(None, description="Extracted financial metrics")
    summary: Optional[str] = Field(None, description="Document summary")


class Document(UUIDMixin, TimestampMixin):
    """Complete document model"""
    metadata: DocumentMetadata
    content: DocumentContent
    vector_embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    processing_status: str = Field(default="pending", description="Processing status")
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QueryContext(BaseModel):
    """Context for document queries"""
    company: Optional[str] = None
    document_types: Optional[List[DocumentType]] = None
    date_range: Optional[tuple] = None
    include_related: bool = Field(default=True, description="Include related documents")
    max_documents: int = Field(default=10, ge=1, le=50)


class QAResponse(BaseModel):
    """Response from document Q&A system"""
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list, description="Source document references")
    related_questions: List[str] = Field(default_factory=list)
    context_used: Optional[str] = Field(None, description="Context used for answering")
    processing_time: Optional[float] = None


class DocumentInsights(BaseModel):
    """Insights extracted from a document"""
    document_id: UUID
    key_topics: List[str] = Field(default_factory=list)
    financial_metrics: Dict[str, Any] = Field(default_factory=dict)
    sentiment_summary: Optional[str] = None
    risk_factors: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    management_commentary: Optional[str] = None


class SearchFilters(BaseModel):
    """Filters for document search"""
    companies: Optional[List[str]] = None
    document_types: Optional[List[DocumentType]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_content: bool = Field(default=False)


class DocumentMatch(BaseModel):
    """Document search result"""
    document_id: UUID
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: DocumentMetadata
    snippet: Optional[str] = Field(None, description="Relevant text snippet")
    highlights: List[str] = Field(default_factory=list)


class DocumentRelationship(BaseModel):
    """Relationship between documents"""
    source_document_id: UUID
    target_document_id: UUID
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(..., ge=0.0, le=1.0, description="Relationship strength")
    description: Optional[str] = None