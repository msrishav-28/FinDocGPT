"""
Document processing tasks for asynchronous execution
"""

import logging
from typing import Dict, Any, List
from celery import current_task
from ..celery_app import celery_app
from ..services.document_processor import DocumentProcessor
from ..services.qa_service import QAService
from ..models.document_models import DocumentMetadata

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.document_tasks.process_document")
def process_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document asynchronously
    
    Args:
        document_data: Dictionary containing document content and metadata
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Update task state
        self.update_state(state="PROCESSING", meta={"status": "Starting document processing"})
        
        document_processor = DocumentProcessor()
        
        # Extract document content and metadata
        content = document_data.get("content")
        metadata = DocumentMetadata(**document_data.get("metadata", {}))
        
        # Process document
        self.update_state(state="PROCESSING", meta={"status": "Parsing document content"})
        processed_doc = document_processor.parse_document(content, metadata)
        
        # Generate embeddings
        self.update_state(state="PROCESSING", meta={"status": "Generating embeddings"})
        embeddings = document_processor.generate_embeddings(processed_doc.content)
        
        # Store in vector database
        self.update_state(state="PROCESSING", meta={"status": "Storing in vector database"})
        doc_id = document_processor.store_document(processed_doc, embeddings)
        
        # Extract key insights
        self.update_state(state="PROCESSING", meta={"status": "Extracting insights"})
        insights = document_processor.extract_insights(processed_doc)
        
        return {
            "document_id": doc_id,
            "status": "completed",
            "insights": insights,
            "metadata": metadata.dict()
        }
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Document processing failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.document_tasks.batch_process_documents")
def batch_process_documents(self, documents_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple documents in batch
    
    Args:
        documents_data: List of document data dictionaries
        
    Returns:
        Dictionary with batch processing results
    """
    try:
        total_docs = len(documents_data)
        processed_docs = []
        failed_docs = []
        
        for i, doc_data in enumerate(documents_data):
            try:
                self.update_state(
                    state="PROCESSING",
                    meta={
                        "status": f"Processing document {i+1}/{total_docs}",
                        "progress": (i / total_docs) * 100
                    }
                )
                
                # Process individual document
                result = process_document.apply_async(args=[doc_data]).get()
                processed_docs.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {str(e)}")
                failed_docs.append({"index": i, "error": str(e)})
        
        return {
            "total_documents": total_docs,
            "processed_successfully": len(processed_docs),
            "failed_documents": len(failed_docs),
            "processed_docs": processed_docs,
            "failed_docs": failed_docs,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Batch document processing failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Batch processing failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.document_tasks.update_document_index")
def update_document_index(self, company: str) -> Dict[str, Any]:
    """
    Update document index for a specific company
    
    Args:
        company: Company ticker symbol
        
    Returns:
        Dictionary with index update results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Updating document index"})
        
        document_processor = DocumentProcessor()
        
        # Rebuild index for company documents
        updated_count = document_processor.rebuild_company_index(company)
        
        return {
            "company": company,
            "updated_documents": updated_count,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Document index update failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Index update failed"}
        )
        raise


@celery_app.task(bind=True, name="app.tasks.document_tasks.generate_document_summary")
def generate_document_summary(self, document_id: str) -> Dict[str, Any]:
    """
    Generate summary for a document
    
    Args:
        document_id: Document identifier
        
    Returns:
        Dictionary with summary results
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Generating document summary"})
        
        qa_service = QAService()
        
        # Generate comprehensive summary
        summary = qa_service.generate_document_summary(document_id)
        
        return {
            "document_id": document_id,
            "summary": summary,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Document summary generation failed: {str(e)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Summary generation failed"}
        )
        raise