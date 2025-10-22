"""
Enhanced Document Processing Service

This service provides advanced document ingestion, parsing, and processing capabilities
for multiple file formats including PDF, HTML, TXT, and JSON.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from pathlib import Path
import re

# File processing imports (with fallbacks)
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import pandas as pd
except ImportError:
    pd = None

# ML and NLP imports (with fallbacks)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import torch
except ImportError:
    torch = None

# Internal imports
from ..models.document import (
    Document, DocumentMetadata, DocumentContent, DocumentType,
    QAResponse, QueryContext, DocumentInsights, SearchFilters,
    DocumentMatch, DocumentRelationship
)
from ..database.connection import DatabaseManager
from ..database.vector_store import VectorStore
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentParsingError(Exception):
    """Exception raised when document parsing fails"""
    pass


class DocumentProcessor:
    """Enhanced document processing service with multi-format support"""
    
    def __init__(self, db_manager: DatabaseManager, vector_store: VectorStore):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.embedding_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for document processing"""
        try:
            if SentenceTransformer is None:
                logger.warning("SentenceTransformer not available - embedding features disabled")
                self.embedding_model = None
                return
            
            # Initialize sentence transformer for embeddings
            model_name = settings.ml.default_embedding_model
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    async def ingest_document(
        self, 
        file_path: str, 
        metadata: DocumentMetadata,
        content_override: Optional[str] = None
    ) -> UUID:
        """
        Ingest and process a document from file path or content
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata
            content_override: Optional content to use instead of reading from file
            
        Returns:
            Document ID
        """
        try:
            document_id = uuid4()
            
            # Parse document content
            if content_override:
                raw_content = content_override
                file_extension = self._detect_content_type(content_override)
            else:
                raw_content, file_extension = await self._parse_file(file_path)
            
            # Process and clean content
            processed_content = await self._process_content(raw_content, file_extension)
            
            # Extract entities and metrics
            entities = await self._extract_entities(processed_content)
            key_metrics = await self._extract_financial_metrics(processed_content)
            
            # Generate summary
            summary = await self._generate_summary(processed_content)
            
            # Create document content
            content = DocumentContent(
                raw_content=raw_content,
                processed_content=processed_content,
                entities=entities,
                key_metrics=key_metrics,
                summary=summary
            )
            
            # Create document
            document = Document(
                id=document_id,
                metadata=metadata,
                content=content,
                processing_status="processing"
            )
            
            # Store in database
            await self._store_document(document)
            
            # Generate and store embeddings
            if self.embedding_model:
                await self._generate_embeddings(document_id, processed_content)
            
            # Update processing status
            await self._update_processing_status(document_id, "completed")
            
            # Build document relationships
            await self._build_document_relationships(document_id, metadata.company)
            
            logger.info(f"Successfully ingested document {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            if 'document_id' in locals():
                await self._update_processing_status(document_id, "failed", str(e))
            raise DocumentParsingError(f"Document ingestion failed: {e}")
    
    async def _parse_file(self, file_path: str) -> Tuple[str, str]:
        """Parse file based on extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if extension == '.pdf':
                return await self._parse_pdf(file_path), 'pdf'
            elif extension in ['.html', '.htm']:
                return await self._parse_html(file_path), 'html'
            elif extension == '.json':
                return await self._parse_json(file_path), 'json'
            elif extension in ['.txt', '.md']:
                return await self._parse_text(file_path), 'txt'
            else:
                # Try to parse as text
                return await self._parse_text(file_path), 'txt'
                
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse {extension} file: {e}")
    
    async def _parse_pdf(self, file_path: str) -> str:
        """Parse PDF file"""
        if PyPDF2 is None:
            raise DocumentParsingError("PyPDF2 not installed - cannot parse PDF files")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except Exception as e:
            raise DocumentParsingError(f"PDF parsing failed: {e}")
    
    async def _parse_html(self, file_path: str) -> str:
        """Parse HTML file"""
        if BeautifulSoup is None:
            raise DocumentParsingError("BeautifulSoup not installed - cannot parse HTML files")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            raise DocumentParsingError(f"HTML parsing failed: {e}")
    
    async def _parse_json(self, file_path: str) -> str:
        """Parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Convert JSON to readable text
            if isinstance(data, dict):
                text_parts = []
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"{key}: {', '.join(map(str, value))}")
                    elif isinstance(value, dict):
                        text_parts.append(f"{key}: {json.dumps(value, indent=2)}")
                
                return '\n'.join(text_parts)
            else:
                return json.dumps(data, indent=2)
                
        except Exception as e:
            raise DocumentParsingError(f"JSON parsing failed: {e}")
    
    async def _parse_text(self, file_path: str) -> str:
        """Parse text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                raise DocumentParsingError(f"Text parsing failed: {e}")
        except Exception as e:
            raise DocumentParsingError(f"Text parsing failed: {e}")
    
    def _detect_content_type(self, content: str) -> str:
        """Detect content type from content string"""
        content_lower = content.lower().strip()
        
        if content_lower.startswith('<!doctype html') or content_lower.startswith('<html'):
            return 'html'
        elif content_lower.startswith('{') or content_lower.startswith('['):
            try:
                json.loads(content)
                return 'json'
            except:
                return 'txt'
        else:
            return 'txt'
    
    async def _process_content(self, raw_content: str, file_type: str) -> str:
        """Process and clean document content"""
        try:
            # Basic text cleaning
            content = raw_content.strip()
            
            # Remove excessive whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Remove special characters that might interfere with processing
            content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\$\%\@\#]', ' ', content)
            
            # Normalize financial terms
            content = self._normalize_financial_terms(content)
            
            # Truncate if too long
            max_length = settings.ml.max_document_length
            if len(content) > max_length:
                content = content[:max_length] + "..."
                logger.warning(f"Document truncated to {max_length} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            return raw_content
    
    def _normalize_financial_terms(self, content: str) -> str:
        """Normalize financial terms and formats"""
        # Normalize currency formats
        content = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'$\1', content)
        
        # Normalize percentage formats
        content = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1%', content)
        
        # Normalize date formats
        content = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', r'\1/\2/\3', content)
        
        return content    

    async def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from document content"""
        try:
            entities = {
                'companies': [],
                'financial_metrics': [],
                'dates': [],
                'currencies': [],
                'percentages': []
            }
            
            # Extract company names (simple pattern matching)
            company_patterns = [
                r'\b[A-Z][a-z]+ (?:Corp|Corporation|Inc|LLC|Ltd|Company|Co)\b',
                r'\b[A-Z]{2,5}\b'  # Stock tickers
            ]
            
            for pattern in company_patterns:
                matches = re.findall(pattern, content)
                entities['companies'].extend(matches)
            
            # Extract financial metrics
            financial_patterns = [
                r'\b(?:revenue|profit|earnings|income|loss|margin|ratio|growth)\b',
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?'
            ]
            
            for pattern in financial_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                entities['financial_metrics'].extend(matches)
            
            # Extract dates
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:Q[1-4]|FY)\s*\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                entities['dates'].extend(matches)
            
            # Extract currencies
            currency_matches = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', content)
            entities['currencies'] = currency_matches
            
            # Extract percentages
            percentage_matches = re.findall(r'\d+(?:\.\d+)?%', content)
            entities['percentages'] = percentage_matches
            
            # Remove duplicates and clean
            for key in entities:
                entities[key] = list(set(entities[key]))
                entities[key] = [item.strip() for item in entities[key] if item.strip()]
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}
    
    async def _extract_financial_metrics(self, content: str) -> Dict[str, Any]:
        """Extract financial metrics from content"""
        try:
            metrics = {}
            
            # Revenue patterns
            revenue_patterns = [
                r'revenue.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?',
                r'sales.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?'
            ]
            
            for pattern in revenue_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metrics['revenue'] = matches[0]
                    break
            
            # Profit patterns
            profit_patterns = [
                r'(?:net\s+)?profit.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?',
                r'earnings.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?'
            ]
            
            for pattern in profit_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metrics['profit'] = matches[0]
                    break
            
            # Margin patterns
            margin_patterns = [
                r'(?:profit\s+)?margin.*?(\d+(?:\.\d+)?)%',
                r'margin.*?(\d+(?:\.\d+)?)%'
            ]
            
            for pattern in margin_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metrics['margin'] = f"{matches[0]}%"
                    break
            
            # Growth patterns
            growth_patterns = [
                r'growth.*?(\d+(?:\.\d+)?)%',
                r'increase.*?(\d+(?:\.\d+)?)%'
            ]
            
            for pattern in growth_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metrics['growth'] = f"{matches[0]}%"
                    break
            
            return metrics
            
        except Exception as e:
            logger.error(f"Financial metrics extraction failed: {e}")
            return {}
    
    async def _generate_summary(self, content: str) -> str:
        """Generate document summary"""
        try:
            # Simple extractive summarization - take first few sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Filter sentences with financial content
            financial_keywords = [
                'revenue', 'profit', 'earnings', 'growth', 'margin', 'performance',
                'quarter', 'year', 'financial', 'results', 'outlook'
            ]
            
            relevant_sentences = []
            for sentence in sentences[:20]:  # Look at first 20 sentences
                if any(keyword in sentence.lower() for keyword in financial_keywords):
                    relevant_sentences.append(sentence.strip())
                
                if len(relevant_sentences) >= 3:
                    break
            
            if relevant_sentences:
                return ' '.join(relevant_sentences)
            else:
                # Fallback to first few sentences
                return ' '.join(sentences[:3])
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return content[:500] + "..." if len(content) > 500 else content
    
    async def _store_document(self, document: Document):
        """Store document in database"""
        try:
            query = """
            INSERT INTO documents (
                id, company, document_type, filing_date, period, source,
                content, processed_content, entities, key_metrics, summary,
                processing_status, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """
            
            await self.db_manager.execute(
                query,
                str(document.id),
                document.metadata.company,
                document.metadata.document_type.value,
                document.metadata.filing_date,
                document.metadata.period,
                document.metadata.source,
                document.content.raw_content,
                document.content.processed_content,
                json.dumps(document.content.entities),
                json.dumps(document.content.key_metrics),
                document.content.summary,
                document.processing_status,
                document.created_at
            )
            
            logger.info(f"Document {document.id} stored in database")
            
        except Exception as e:
            logger.error(f"Failed to store document {document.id}: {e}")
            raise
    
    async def _generate_embeddings(self, document_id: UUID, content: str):
        """Generate and store document embeddings"""
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available, skipping embedding generation")
                return
            
            # Generate embedding
            embedding = self.embedding_model.encode(content, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            # Store in vector store
            success = await self.vector_store.store_embedding(
                document_id, 
                embedding,
                {'content_length': len(content), 'generated_at': datetime.utcnow().isoformat()}
            )
            
            if success:
                logger.info(f"Generated and stored embedding for document {document_id}")
            else:
                logger.warning(f"Failed to store embedding for document {document_id}")
                
        except Exception as e:
            logger.error(f"Embedding generation failed for document {document_id}: {e}")
    
    async def _update_processing_status(
        self, 
        document_id: UUID, 
        status: str, 
        error_message: Optional[str] = None
    ):
        """Update document processing status"""
        try:
            query = """
            UPDATE documents 
            SET processing_status = $1, error_message = $2, updated_at = CURRENT_TIMESTAMP
            WHERE id = $3
            """
            
            await self.db_manager.execute(query, status, error_message, str(document_id))
            
        except Exception as e:
            logger.error(f"Failed to update processing status for document {document_id}: {e}")
    
    async def _build_document_relationships(self, document_id: UUID, company: str):
        """Build relationships between documents"""
        try:
            # Find other documents for the same company
            query = """
            SELECT id, document_type, filing_date, period
            FROM documents 
            WHERE company = $1 AND id != $2 AND processing_status = 'completed'
            ORDER BY filing_date DESC
            LIMIT 10
            """
            
            related_docs = await self.db_manager.fetch(query, company, str(document_id))
            
            # Create relationships
            for doc in related_docs:
                relationship_type = self._determine_relationship_type(doc)
                strength = self._calculate_relationship_strength(doc)
                
                # Store relationship
                await self._store_document_relationship(
                    document_id,
                    UUID(doc['id']),
                    relationship_type,
                    strength
                )
            
            logger.info(f"Built {len(related_docs)} relationships for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to build document relationships: {e}")
    
    def _determine_relationship_type(self, related_doc: Dict) -> str:
        """Determine the type of relationship between documents"""
        doc_type = related_doc['document_type']
        
        if doc_type in ['earnings_report', 'quarterly_report']:
            return 'quarterly_sequence'
        elif doc_type == 'annual_report':
            return 'annual_sequence'
        elif doc_type == 'press_release':
            return 'related_announcement'
        else:
            return 'company_related'
    
    def _calculate_relationship_strength(self, related_doc: Dict) -> float:
        """Calculate relationship strength based on document properties"""
        # Simple scoring based on document type and recency
        base_score = 0.5
        
        doc_type = related_doc['document_type']
        if doc_type in ['earnings_report', 'quarterly_report']:
            base_score += 0.3
        elif doc_type == 'annual_report':
            base_score += 0.2
        
        # Adjust for recency (more recent = stronger relationship)
        filing_date = related_doc['filing_date']
        if filing_date:
            days_ago = (datetime.utcnow() - filing_date).days
            if days_ago < 90:  # Within 3 months
                base_score += 0.2
            elif days_ago < 365:  # Within 1 year
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    async def _store_document_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        relationship_type: str,
        strength: float
    ):
        """Store document relationship"""
        try:
            query = """
            INSERT INTO document_relationships (
                source_document_id, target_document_id, relationship_type, strength, created_at
            ) VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            ON CONFLICT (source_document_id, target_document_id) 
            DO UPDATE SET relationship_type = $3, strength = $4, updated_at = CURRENT_TIMESTAMP
            """
            
            await self.db_manager.execute(
                query, str(source_id), str(target_id), relationship_type, strength
            )
            
        except Exception as e:
            logger.error(f"Failed to store document relationship: {e}")
    
    async def get_document_metadata(self, document_id: UUID) -> Optional[DocumentMetadata]:
        """Get document metadata by ID"""
        try:
            query = """
            SELECT company, document_type, filing_date, period, source
            FROM documents WHERE id = $1
            """
            
            row = await self.db_manager.fetchrow(query, str(document_id))
            if not row:
                return None
            
            return DocumentMetadata(
                company=row['company'],
                document_type=DocumentType(row['document_type']),
                filing_date=row['filing_date'],
                period=row['period'],
                source=row['source']
            )
            
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            return None
    
    async def validate_document_metadata(self, metadata: DocumentMetadata) -> List[str]:
        """Validate document metadata and return list of validation errors"""
        errors = []
        
        # Validate company
        if not metadata.company or len(metadata.company.strip()) == 0:
            errors.append("Company identifier is required")
        
        # Validate document type
        if not metadata.document_type:
            errors.append("Document type is required")
        
        # Validate filing date
        if not metadata.filing_date:
            errors.append("Filing date is required")
        elif metadata.filing_date > datetime.utcnow():
            errors.append("Filing date cannot be in the future")
        
        # Validate source
        if not metadata.source or len(metadata.source.strip()) == 0:
            errors.append("Document source is required")
        
        return errors    
# Vector-based document indexing and search methods
    
    async def search_documents(
        self, 
        query: str, 
        filters: SearchFilters,
        use_vector_search: bool = True
    ) -> List[DocumentMatch]:
        """
        Search documents using vector similarity or text-based search
        
        Args:
            query: Search query
            filters: Search filters
            use_vector_search: Whether to use vector similarity search
            
        Returns:
            List of document matches
        """
        try:
            if use_vector_search and self.embedding_model:
                return await self._vector_search(query, filters)
            else:
                return await self._text_search(query, filters)
                
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    async def _vector_search(self, query: str, filters: SearchFilters) -> List[DocumentMatch]:
        """Perform vector similarity search"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # Build search filters for vector store
            vector_filters = {}
            if filters.companies:
                vector_filters['company'] = filters.companies[0]  # Vector store expects single company
            if filters.document_types:
                vector_filters['document_type'] = filters.document_types[0].value
            if filters.date_from:
                vector_filters['date_from'] = filters.date_from
            if filters.date_to:
                vector_filters['date_to'] = filters.date_to
            
            # Perform similarity search
            results = await self.vector_store.similarity_search(
                query_embedding,
                limit=20,  # Get more results for filtering
                similarity_threshold=filters.min_confidence,
                filters=vector_filters
            )
            
            # Convert to DocumentMatch objects
            matches = []
            for result in results:
                try:
                    # Get document metadata
                    doc_id = UUID(result['document_id'])
                    metadata = await self.get_document_metadata(doc_id)
                    
                    if metadata:
                        match = DocumentMatch(
                            document_id=doc_id,
                            score=result['similarity'],
                            metadata=metadata,
                            snippet=result.get('snippet', ''),
                            highlights=self._extract_highlights(query, result.get('snippet', ''))
                        )
                        matches.append(match)
                        
                except Exception as e:
                    logger.warning(f"Failed to process search result: {e}")
                    continue
            
            # Apply additional filters
            matches = self._apply_search_filters(matches, filters)
            
            # Sort by score
            matches.sort(key=lambda x: x.score, reverse=True)
            
            return matches[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _text_search(self, query: str, filters: SearchFilters) -> List[DocumentMatch]:
        """Perform text-based search as fallback"""
        try:
            # Build SQL query
            where_conditions = ["processing_status = 'completed'"]
            params = []
            param_count = 0
            
            # Add text search condition
            param_count += 1
            where_conditions.append(f"(processed_content ILIKE ${param_count} OR summary ILIKE ${param_count})")
            params.append(f"%{query}%")
            
            # Apply filters
            if filters.companies:
                param_count += 1
                where_conditions.append(f"company = ANY(${param_count})")
                params.append(filters.companies)
            
            if filters.document_types:
                param_count += 1
                doc_types = [dt.value for dt in filters.document_types]
                where_conditions.append(f"document_type = ANY(${param_count})")
                params.append(doc_types)
            
            if filters.date_from:
                param_count += 1
                where_conditions.append(f"filing_date >= ${param_count}")
                params.append(filters.date_from)
            
            if filters.date_to:
                param_count += 1
                where_conditions.append(f"filing_date <= ${param_count}")
                params.append(filters.date_to)
            
            where_clause = " AND ".join(where_conditions)
            
            query_sql = f"""
            SELECT 
                id, company, document_type, filing_date, period, source,
                SUBSTRING(processed_content, 1, 500) as snippet,
                ts_rank(to_tsvector('english', processed_content), plainto_tsquery('english', $1)) as score
            FROM documents 
            WHERE {where_clause}
            ORDER BY score DESC, filing_date DESC
            LIMIT 10
            """
            
            rows = await self.db_manager.fetch(query_sql, query, *params[1:])
            
            # Convert to DocumentMatch objects
            matches = []
            for row in rows:
                try:
                    metadata = DocumentMetadata(
                        company=row['company'],
                        document_type=DocumentType(row['document_type']),
                        filing_date=row['filing_date'],
                        period=row['period'],
                        source=row['source']
                    )
                    
                    match = DocumentMatch(
                        document_id=UUID(row['id']),
                        score=float(row['score']) if row['score'] else 0.0,
                        metadata=metadata,
                        snippet=row['snippet'],
                        highlights=self._extract_highlights(query, row['snippet'])
                    )
                    matches.append(match)
                    
                except Exception as e:
                    logger.warning(f"Failed to process text search result: {e}")
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def _apply_search_filters(self, matches: List[DocumentMatch], filters: SearchFilters) -> List[DocumentMatch]:
        """Apply additional filters to search results"""
        filtered_matches = []
        
        for match in matches:
            # Apply company filter
            if filters.companies and match.metadata.company not in filters.companies:
                continue
            
            # Apply document type filter
            if filters.document_types and match.metadata.document_type not in filters.document_types:
                continue
            
            # Apply date filters
            if filters.date_from and match.metadata.filing_date < filters.date_from:
                continue
            
            if filters.date_to and match.metadata.filing_date > filters.date_to:
                continue
            
            # Apply confidence filter
            if match.score < filters.min_confidence:
                continue
            
            filtered_matches.append(match)
        
        return filtered_matches
    
    def _extract_highlights(self, query: str, text: str) -> List[str]:
        """Extract highlighted text snippets"""
        if not text:
            return []
        
        highlights = []
        query_words = query.lower().split()
        
        # Find sentences containing query words
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                highlights.append(sentence.strip())
                
                if len(highlights) >= 3:  # Limit to 3 highlights
                    break
        
        return highlights
    
    async def get_similar_documents(
        self, 
        document_id: UUID, 
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[DocumentMatch]:
        """Find documents similar to a given document"""
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available for similarity search")
                return []
            
            # Use vector store to find similar documents
            similar_docs = await self.vector_store.find_similar_documents(
                document_id, limit, similarity_threshold
            )
            
            # Convert to DocumentMatch objects
            matches = []
            for doc in similar_docs:
                try:
                    doc_id = UUID(doc['document_id'])
                    metadata = await self.get_document_metadata(doc_id)
                    
                    if metadata:
                        match = DocumentMatch(
                            document_id=doc_id,
                            score=doc['similarity'],
                            metadata=metadata,
                            snippet=doc.get('snippet', ''),
                            highlights=[]
                        )
                        matches.append(match)
                        
                except Exception as e:
                    logger.warning(f"Failed to process similar document: {e}")
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"Similar documents search failed: {e}")
            return []
    
    async def chunk_document(
        self, 
        content: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk large documents for better processing and embedding
        
        Args:
            content: Document content to chunk
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of document chunks with metadata
        """
        try:
            if len(content) <= chunk_size:
                return [{
                    'chunk_id': 0,
                    'content': content,
                    'start_pos': 0,
                    'end_pos': len(content),
                    'size': len(content)
                }]
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(content):
                # Calculate end position
                end = min(start + chunk_size, len(content))
                
                # Try to break at sentence boundary
                if end < len(content):
                    # Look for sentence endings within the last 200 characters
                    search_start = max(end - 200, start)
                    sentence_end = -1
                    
                    for i in range(end - 1, search_start - 1, -1):
                        if content[i] in '.!?':
                            sentence_end = i + 1
                            break
                    
                    if sentence_end > start:
                        end = sentence_end
                
                # Extract chunk
                chunk_content = content[start:end].strip()
                
                if chunk_content:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'content': chunk_content,
                        'start_pos': start,
                        'end_pos': end,
                        'size': len(chunk_content)
                    })
                    chunk_id += 1
                
                # Move to next chunk with overlap
                start = max(end - overlap, start + 1)
                
                # Prevent infinite loop
                if start >= len(content):
                    break
            
            logger.info(f"Document chunked into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            return []
    
    async def reindex_document_embeddings(
        self, 
        document_ids: Optional[List[UUID]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Reindex document embeddings for specified documents or all documents
        
        Args:
            document_ids: Specific document IDs to reindex (None for all)
            batch_size: Number of documents to process in each batch
            
        Returns:
            Reindexing results
        """
        try:
            if not self.embedding_model:
                return {'error': 'Embedding model not available'}
            
            # Build query
            if document_ids:
                placeholders = ','.join(f'${i+1}' for i in range(len(document_ids)))
                query = f"""
                SELECT id, processed_content 
                FROM documents 
                WHERE id IN ({placeholders}) AND processing_status = 'completed'
                ORDER BY created_at DESC
                """
                params = [str(doc_id) for doc_id in document_ids]
            else:
                query = """
                SELECT id, processed_content 
                FROM documents 
                WHERE processing_status = 'completed'
                ORDER BY created_at DESC
                """
                params = []
            
            rows = await self.db_manager.fetch(query, *params)
            
            total_docs = len(rows)
            processed_docs = 0
            failed_docs = 0
            
            # Process in batches
            for i in range(0, total_docs, batch_size):
                batch = rows[i:i + batch_size]
                
                for row in batch:
                    try:
                        doc_id = UUID(row['id'])
                        content = row['processed_content']
                        
                        if content:
                            await self._generate_embeddings(doc_id, content)
                            processed_docs += 1
                        else:
                            logger.warning(f"No content for document {doc_id}")
                            failed_docs += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to reindex document {row['id']}: {e}")
                        failed_docs += 1
                
                # Small delay between batches to avoid overwhelming the system
                if i + batch_size < total_docs:
                    await asyncio.sleep(0.1)
            
            return {
                'total_documents': total_docs,
                'processed_documents': processed_docs,
                'failed_documents': failed_docs,
                'success_rate': processed_docs / total_docs if total_docs > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Reindexing failed: {e}")
            return {'error': str(e)}
    
    async def get_document_insights(self, document_id: UUID) -> Optional[DocumentInsights]:
        """Get comprehensive insights for a document"""
        try:
            # Get document data
            query = """
            SELECT entities, key_metrics, summary, processed_content
            FROM documents WHERE id = $1
            """
            
            row = await self.db_manager.fetchrow(query, str(document_id))
            if not row:
                return None
            
            # Parse stored data
            entities = json.loads(row['entities']) if row['entities'] else {}
            key_metrics = json.loads(row['key_metrics']) if row['key_metrics'] else {}
            
            # Extract additional insights
            content = row['processed_content'] or ''
            
            # Extract key topics
            key_topics = self._extract_key_topics(content)
            
            # Extract risk factors
            risk_factors = self._extract_risk_factors(content)
            
            # Extract opportunities
            opportunities = self._extract_opportunities(content)
            
            # Extract management commentary
            management_commentary = self._extract_management_commentary(content)
            
            return DocumentInsights(
                document_id=document_id,
                key_topics=key_topics,
                financial_metrics=key_metrics,
                sentiment_summary=row['summary'],
                risk_factors=risk_factors,
                opportunities=opportunities,
                management_commentary=management_commentary
            )
            
        except Exception as e:
            logger.error(f"Failed to get document insights: {e}")
            return None
    
    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from document content"""
        topics = []
        
        # Financial topics
        financial_topics = [
            'revenue', 'profit', 'earnings', 'growth', 'margin', 'cash flow',
            'debt', 'equity', 'assets', 'liabilities', 'dividend', 'acquisition'
        ]
        
        content_lower = content.lower()
        for topic in financial_topics:
            if topic in content_lower:
                topics.append(topic.title())
        
        return topics[:10]  # Limit to top 10 topics
    
    def _extract_risk_factors(self, content: str) -> List[str]:
        """Extract risk factors from content"""
        risk_patterns = [
            r'risk[s]?\s+(?:include|are|of)([^.]+)',
            r'challenge[s]?\s+(?:include|are)([^.]+)',
            r'concern[s]?\s+(?:about|regarding)([^.]+)',
            r'uncertainty\s+(?:about|regarding)([^.]+)'
        ]
        
        risks = []
        for pattern in risk_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                risk = match.strip()
                if len(risk) > 10 and len(risk) < 200:
                    risks.append(risk)
        
        return risks[:5]  # Limit to top 5 risks
    
    def _extract_opportunities(self, content: str) -> List[str]:
        """Extract opportunities from content"""
        opportunity_patterns = [
            r'opportunit(?:y|ies)\s+(?:include|are|to)([^.]+)',
            r'potential\s+(?:for|to)([^.]+)',
            r'expect\s+(?:to|growth|improvement)([^.]+)',
            r'outlook\s+(?:is|remains)([^.]+)'
        ]
        
        opportunities = []
        for pattern in opportunity_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                opportunity = match.strip()
                if len(opportunity) > 10 and len(opportunity) < 200:
                    opportunities.append(opportunity)
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    def _extract_management_commentary(self, content: str) -> Optional[str]:
        """Extract management commentary from content"""
        commentary_patterns = [
            r'(?:CEO|CFO|management|executive)\s+(?:said|stated|commented|noted)([^.]+\.)',
            r'according\s+to\s+(?:management|executives)([^.]+\.)',
            r'management\s+(?:believes|expects|anticipates)([^.]+\.)'
        ]
        
        for pattern in commentary_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Return the first substantial commentary
                commentary = matches[0].strip()
                if len(commentary) > 20:
                    return commentary
        
        return None  
  # Advanced Q&A engine with context management
    
    async def ask_question(
        self, 
        question: str, 
        context: QueryContext
    ) -> QAResponse:
        """
        Answer questions about documents with advanced context management
        
        Args:
            question: The question to answer
            context: Query context with filters and options
            
        Returns:
            QA response with answer, confidence, and sources
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Find relevant documents
            relevant_docs = await self._find_relevant_documents(question, context)
            
            if not relevant_docs:
                return QAResponse(
                    answer="I couldn't find relevant documents to answer your question.",
                    confidence=0.0,
                    sources=[],
                    related_questions=[],
                    processing_time=asyncio.get_event_loop().time() - start_time
                )
            
            # Build context from relevant documents
            context_text = await self._build_qa_context(relevant_docs, question)
            
            # Generate answer using financial language model
            answer, confidence = await self._generate_answer(question, context_text)
            
            # Extract sources
            sources = [f"{doc['company']} - {doc['document_type']} ({doc['period']})" 
                      for doc in relevant_docs]
            
            # Generate related questions
            related_questions = await self._generate_related_questions(question, context_text)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return QAResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                related_questions=related_questions,
                context_used=context_text[:500] + "..." if len(context_text) > 500 else context_text,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Q&A processing failed: {e}")
            return QAResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                confidence=0.0,
                sources=[],
                related_questions=[]
            )
    
    async def _find_relevant_documents(
        self, 
        question: str, 
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Find documents relevant to the question"""
        try:
            # Build search filters from context
            filters = SearchFilters(
                companies=context.company.split(',') if context.company else None,
                document_types=context.document_types,
                date_from=context.date_range[0] if context.date_range else None,
                date_to=context.date_range[1] if context.date_range else None,
                min_confidence=0.3,  # Lower threshold for Q&A
                include_content=True
            )
            
            # Search for relevant documents
            matches = await self.search_documents(question, filters, use_vector_search=True)
            
            # Convert to dict format for easier processing
            relevant_docs = []
            for match in matches[:context.max_documents]:
                doc_data = await self._get_document_content(match.document_id)
                if doc_data:
                    doc_data.update({
                        'relevance_score': match.score,
                        'company': match.metadata.company,
                        'document_type': match.metadata.document_type.value,
                        'period': match.metadata.period,
                        'filing_date': match.metadata.filing_date
                    })
                    relevant_docs.append(doc_data)
            
            # If include_related is True, add related documents
            if context.include_related and relevant_docs:
                for doc in relevant_docs[:3]:  # Only for top 3 documents
                    related_matches = await self.get_similar_documents(
                        UUID(doc['id']), limit=2, similarity_threshold=0.8
                    )
                    
                    for related_match in related_matches:
                        related_doc_data = await self._get_document_content(related_match.document_id)
                        if related_doc_data and related_doc_data not in relevant_docs:
                            related_doc_data.update({
                                'relevance_score': related_match.score * 0.8,  # Reduce score for related docs
                                'company': related_match.metadata.company,
                                'document_type': related_match.metadata.document_type.value,
                                'period': related_match.metadata.period,
                                'filing_date': related_match.metadata.filing_date
                            })
                            relevant_docs.append(related_doc_data)
            
            # Sort by relevance score
            relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return relevant_docs[:context.max_documents]
            
        except Exception as e:
            logger.error(f"Failed to find relevant documents: {e}")
            return []
    
    async def _get_document_content(self, document_id: UUID) -> Optional[Dict[str, Any]]:
        """Get document content and metadata"""
        try:
            query = """
            SELECT id, company, document_type, period, filing_date, 
                   processed_content, summary, key_metrics
            FROM documents 
            WHERE id = $1 AND processing_status = 'completed'
            """
            
            row = await self.db_manager.fetchrow(query, str(document_id))
            if not row:
                return None
            
            return {
                'id': str(row['id']),
                'company': row['company'],
                'document_type': row['document_type'],
                'period': row['period'],
                'filing_date': row['filing_date'],
                'content': row['processed_content'],
                'summary': row['summary'],
                'key_metrics': json.loads(row['key_metrics']) if row['key_metrics'] else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get document content: {e}")
            return None
    
    async def _build_qa_context(
        self, 
        relevant_docs: List[Dict[str, Any]], 
        question: str
    ) -> str:
        """Build context text from relevant documents for Q&A"""
        try:
            context_parts = []
            
            for doc in relevant_docs:
                # Add document header
                header = f"Document: {doc['company']} {doc['document_type']} ({doc['period']})"
                context_parts.append(header)
                context_parts.append("-" * len(header))
                
                # Add summary if available
                if doc.get('summary'):
                    context_parts.append(f"Summary: {doc['summary']}")
                    context_parts.append("")
                
                # Add key metrics if relevant to question
                if doc.get('key_metrics') and self._is_metrics_relevant(question):
                    metrics_text = self._format_metrics(doc['key_metrics'])
                    if metrics_text:
                        context_parts.append(f"Key Metrics: {metrics_text}")
                        context_parts.append("")
                
                # Add relevant content chunks
                content = doc.get('content', '')
                if content:
                    # Extract most relevant chunks
                    relevant_chunks = self._extract_relevant_chunks(content, question)
                    for chunk in relevant_chunks:
                        context_parts.append(chunk)
                        context_parts.append("")
                
                context_parts.append("=" * 50)
                context_parts.append("")
            
            context_text = "\n".join(context_parts)
            
            # Truncate if too long (keep within model limits)
            max_context_length = 4000  # Leave room for question and answer
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "\n[Context truncated...]"
            
            return context_text
            
        except Exception as e:
            logger.error(f"Failed to build Q&A context: {e}")
            return ""
    
    def _is_metrics_relevant(self, question: str) -> bool:
        """Check if financial metrics are relevant to the question"""
        metrics_keywords = [
            'revenue', 'profit', 'earnings', 'margin', 'growth', 'financial',
            'performance', 'results', 'numbers', 'metrics', 'money', 'cost',
            'expense', 'income', 'cash', 'debt', 'ratio'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in metrics_keywords)
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format financial metrics for context"""
        if not metrics:
            return ""
        
        formatted_parts = []
        for key, value in metrics.items():
            if value:
                formatted_parts.append(f"{key.title()}: {value}")
        
        return ", ".join(formatted_parts)
    
    def _extract_relevant_chunks(self, content: str, question: str, max_chunks: int = 3) -> List[str]:
        """Extract content chunks most relevant to the question"""
        try:
            # Split content into sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Score sentences based on question relevance
            question_words = set(question.lower().split())
            scored_sentences = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                sentence_words = set(sentence.lower().split())
                
                # Calculate relevance score
                common_words = question_words.intersection(sentence_words)
                score = len(common_words) / len(question_words) if question_words else 0
                
                # Boost score for financial terms
                financial_terms = ['revenue', 'profit', 'earnings', 'growth', 'margin']
                if any(term in sentence.lower() for term in financial_terms):
                    score += 0.2
                
                scored_sentences.append((score, i, sentence))
            
            # Sort by score and select top chunks
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            
            # Group consecutive high-scoring sentences into chunks
            chunks = []
            used_indices = set()
            
            for score, idx, sentence in scored_sentences[:max_chunks * 3]:
                if score < 0.1 or idx in used_indices:  # Skip low-relevance sentences
                    continue
                
                # Build chunk with context (previous and next sentences)
                chunk_sentences = []
                
                # Add previous sentence for context
                if idx > 0 and idx - 1 not in used_indices:
                    chunk_sentences.append(sentences[idx - 1])
                    used_indices.add(idx - 1)
                
                # Add main sentence
                chunk_sentences.append(sentence)
                used_indices.add(idx)
                
                # Add next sentence for context
                if idx < len(sentences) - 1 and idx + 1 not in used_indices:
                    chunk_sentences.append(sentences[idx + 1])
                    used_indices.add(idx + 1)
                
                chunk = " ".join(chunk_sentences)
                chunks.append(chunk)
                
                if len(chunks) >= max_chunks:
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to extract relevant chunks: {e}")
            return [content[:1000]]  # Fallback to first 1000 characters
    
    async def _generate_answer(self, question: str, context: str) -> Tuple[str, float]:
        """Generate answer using financial language model"""
        try:
            # Try to use the existing QA pipeline first
            from .qa_service import _ensure_qa
            qa_pipeline = _ensure_qa()
            
            if qa_pipeline:
                # Use the existing pipeline
                result = qa_pipeline({"question": question, "context": context})
                answer = result.get("answer", "")
                confidence = float(result.get("score", 0.0))
                
                # Enhance answer with financial context
                enhanced_answer = self._enhance_financial_answer(answer, context)
                
                return enhanced_answer, confidence
            else:
                # Fallback to rule-based answer generation
                return self._generate_rule_based_answer(question, context)
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._generate_rule_based_answer(question, context)
    
    def _enhance_financial_answer(self, answer: str, context: str) -> str:
        """Enhance answer with financial context and formatting"""
        try:
            # Add financial context if the answer seems incomplete
            if len(answer) < 50:
                # Look for relevant financial information in context
                financial_info = self._extract_financial_context(context)
                if financial_info:
                    answer = f"{answer} {financial_info}"
            
            # Format financial numbers
            answer = self._format_financial_numbers(answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer enhancement failed: {e}")
            return answer
    
    def _extract_financial_context(self, context: str) -> str:
        """Extract relevant financial context"""
        # Look for sentences with financial metrics
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        financial_sentences = []
        for sentence in sentences:
            if re.search(r'\$\d+|revenue|profit|earnings|\d+%', sentence, re.IGNORECASE):
                financial_sentences.append(sentence.strip())
                
                if len(financial_sentences) >= 2:
                    break
        
        return " ".join(financial_sentences)
    
    def _format_financial_numbers(self, text: str) -> str:
        """Format financial numbers in text"""
        # Format large numbers
        text = re.sub(r'\$(\d+)000000000', r'$\1 billion', text)
        text = re.sub(r'\$(\d+)000000', r'$\1 million', text)
        
        return text
    
    def _generate_rule_based_answer(self, question: str, context: str) -> Tuple[str, float]:
        """Generate rule-based answer as fallback"""
        try:
            question_lower = question.lower()
            
            # Revenue questions
            if 'revenue' in question_lower or 'sales' in question_lower:
                revenue_match = re.search(r'revenue.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?', context, re.IGNORECASE)
                if revenue_match:
                    return f"The revenue was ${revenue_match.group(1)}.", 0.8
            
            # Profit questions
            if 'profit' in question_lower or 'earnings' in question_lower:
                profit_match = re.search(r'(?:profit|earnings).*?\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?', context, re.IGNORECASE)
                if profit_match:
                    return f"The profit/earnings was ${profit_match.group(1)}.", 0.8
            
            # Growth questions
            if 'growth' in question_lower:
                growth_match = re.search(r'growth.*?(\d+(?:\.\d+)?)%', context, re.IGNORECASE)
                if growth_match:
                    return f"The growth rate was {growth_match.group(1)}%.", 0.7
            
            # Margin questions
            if 'margin' in question_lower:
                margin_match = re.search(r'margin.*?(\d+(?:\.\d+)?)%', context, re.IGNORECASE)
                if margin_match:
                    return f"The margin was {margin_match.group(1)}%.", 0.7
            
            # Generic fallback
            sentences = re.split(r'(?<=[.!?])\s+', context)
            question_words = set(question_lower.split())
            
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences[:10]:  # Check first 10 sentences
                sentence_words = set(sentence.lower().split())
                common_words = question_words.intersection(sentence_words)
                score = len(common_words) / len(question_words) if question_words else 0
                
                if score > best_score and len(sentence) > 20:
                    best_score = score
                    best_sentence = sentence
            
            if best_sentence:
                return best_sentence.strip(), best_score
            else:
                return "I couldn't find specific information to answer your question in the available documents.", 0.1
                
        except Exception as e:
            logger.error(f"Rule-based answer generation failed: {e}")
            return "I encountered an error while processing your question.", 0.0
    
    async def _generate_related_questions(self, question: str, context: str) -> List[str]:
        """Generate related questions based on context"""
        try:
            related_questions = []
            
            # Extract key topics from context
            topics = self._extract_key_topics(context)
            
            # Generate questions based on topics
            question_templates = {
                'revenue': [
                    "What was the revenue growth rate?",
                    "How does this revenue compare to previous periods?",
                    "What drove the revenue changes?"
                ],
                'profit': [
                    "What was the profit margin?",
                    "How did profitability change?",
                    "What factors affected profitability?"
                ],
                'growth': [
                    "What are the growth drivers?",
                    "Is this growth sustainable?",
                    "What are the growth projections?"
                ],
                'margin': [
                    "How do margins compare to competitors?",
                    "What is impacting margins?",
                    "Are margins improving or declining?"
                ]
            }
            
            question_lower = question.lower()
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower in question_templates and topic_lower not in question_lower:
                    related_questions.extend(question_templates[topic_lower][:2])
            
            # Add generic financial questions if none found
            if not related_questions:
                related_questions = [
                    "What were the key financial highlights?",
                    "How did the company perform this period?",
                    "What are the main business drivers?"
                ]
            
            return related_questions[:3]  # Limit to 3 related questions
            
        except Exception as e:
            logger.error(f"Related questions generation failed: {e}")
            return []