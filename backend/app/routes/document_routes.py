"""
Document processing API routes
"""

import os
import logging
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..models.document import (
    DocumentMetadata, DocumentType, QueryContext, SearchFilters,
    DocumentMatch, DocumentInsights, QAResponse
)
from ..services.document_processor import DocumentProcessor
from ..database.connection import get_database
from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


async def get_document_processor():
    """Dependency to get document processor instance"""
    db_manager = await get_database()
    vector_store = VectorStore(db_manager)
    return DocumentProcessor(db_manager, vector_store)


@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    company: str = Form(...),
    document_type: DocumentType = Form(...),
    filing_date: datetime = Form(...),
    period: Optional[str] = Form(None),
    source: str = Form(...),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Upload and process a financial document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Create metadata
        metadata = DocumentMetadata(
            company=company,
            document_type=document_type,
            filing_date=filing_date,
            period=period,
            source=source
        )
        
        # Validate metadata
        validation_errors = await processor.validate_document_metadata(metadata)
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail=f"Metadata validation failed: {', '.join(validation_errors)}"
            )
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8', errors='ignore')
        
        # Process document
        document_id = await processor.ingest_document(
            file_path="",  # Not used when content_override is provided
            metadata=metadata,
            content_override=content_str
        )
        
        return {
            "document_id": str(document_id),
            "message": "Document uploaded and processed successfully",
            "company": company,
            "document_type": document_type.value,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.post("/upload-file", response_model=dict)
async def upload_document_file(
    file_path: str = Form(...),
    company: str = Form(...),
    document_type: DocumentType = Form(...),
    filing_date: datetime = Form(...),
    period: Optional[str] = Form(None),
    source: str = Form(...),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Process a document from file path"""
    try:
        # Create metadata
        metadata = DocumentMetadata(
            company=company,
            document_type=document_type,
            filing_date=filing_date,
            period=period,
            source=source
        )
        
        # Validate metadata
        validation_errors = await processor.validate_document_metadata(metadata)
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail=f"Metadata validation failed: {', '.join(validation_errors)}"
            )
        
        # Process document
        document_id = await processor.ingest_document(file_path, metadata)
        
        return {
            "document_id": str(document_id),
            "message": "Document processed successfully",
            "company": company,
            "document_type": document_type.value,
            "file_path": file_path
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/search", response_model=List[DocumentMatch])
async def search_documents(
    query: str = Form(...),
    companies: Optional[List[str]] = Form(None),
    document_types: Optional[List[DocumentType]] = Form(None),
    date_from: Optional[datetime] = Form(None),
    date_to: Optional[datetime] = Form(None),
    min_confidence: float = Form(0.3),
    use_vector_search: bool = Form(True),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Search documents using vector similarity or text search"""
    try:
        # Create search filters
        filters = SearchFilters(
            companies=companies,
            document_types=document_types,
            date_from=date_from,
            date_to=date_to,
            min_confidence=min_confidence,
            include_content=False
        )
        
        # Perform search
        results = await processor.search_documents(query, filters, use_vector_search)
        
        return results
        
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")


@router.post("/qa", response_model=QAResponse)
async def ask_question(
    question: str = Form(...),
    company: Optional[str] = Form(None),
    document_types: Optional[List[DocumentType]] = Form(None),
    include_related: bool = Form(True),
    max_documents: int = Form(10),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Ask questions about financial documents"""
    try:
        # Create query context
        context = QueryContext(
            company=company,
            document_types=document_types,
            include_related=include_related,
            max_documents=max_documents
        )
        
        # Process question
        response = await processor.ask_question(question, context)
        
        return response
        
    except Exception as e:
        logger.error(f"Q&A processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Q&A processing failed: {str(e)}")


@router.get("/insights/{document_id}", response_model=DocumentInsights)
async def get_document_insights(
    document_id: UUID,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Get comprehensive insights for a document"""
    try:
        insights = await processor.get_document_insights(document_id)
        
        if not insights:
            raise HTTPException(status_code=404, detail="Document not found or not processed")
        
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.get("/similar/{document_id}", response_model=List[DocumentMatch])
async def get_similar_documents(
    document_id: UUID,
    limit: int = Query(5, ge=1, le=20),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Find documents similar to a given document"""
    try:
        similar_docs = await processor.get_similar_documents(
            document_id, limit, similarity_threshold
        )
        
        return similar_docs
        
    except Exception as e:
        logger.error(f"Similar documents search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similar documents search failed: {str(e)}")


@router.post("/reindex", response_model=dict)
async def reindex_embeddings(
    document_ids: Optional[List[UUID]] = Form(None),
    batch_size: int = Form(10),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Reindex document embeddings"""
    try:
        result = await processor.reindex_document_embeddings(document_ids, batch_size)
        
        return {
            "message": "Reindexing completed",
            "results": result
        }
        
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")


@router.get("/metadata/{document_id}", response_model=DocumentMetadata)
async def get_document_metadata(
    document_id: UUID,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Get document metadata"""
    try:
        metadata = await processor.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@router.post("/chunk", response_model=dict)
async def chunk_document_content(
    content: str = Form(...),
    chunk_size: int = Form(1000),
    overlap: int = Form(200),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Chunk document content for processing"""
    try:
        chunks = await processor.chunk_document(content, chunk_size, overlap)
        
        return {
            "message": "Document chunked successfully",
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Document chunking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document chunking failed: {str(e)}")


@router.get("/health", response_model=dict)
async def document_service_health(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Check document service health"""
    try:
        # Check if embedding model is available
        embedding_available = processor.embedding_model is not None
        
        # Get vector store stats
        vector_stats = await processor.vector_store.get_embedding_stats()
        
        return {
            "status": "healthy",
            "embedding_model_available": embedding_available,
            "vector_store_stats": vector_stats,
            "service": "document_processor"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "document_processor"
        }