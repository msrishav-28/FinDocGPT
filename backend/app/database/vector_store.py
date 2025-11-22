"""
Vector database integration for document embeddings and similarity search
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from uuid import UUID
import json

from .connection import DatabaseManager

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database integration for document embeddings"""
    
    def __init__(self, db_manager: DatabaseManager, embedding_dimension: int = 768):
        self.db_manager = db_manager
        self.embedding_dimension = embedding_dimension
    
    async def initialize(self):
        """Initialize vector store with pgvector extension"""
        try:
            # Enable pgvector extension
            await self.db_manager.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("Vector extension initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector extension: {e}")
            # Continue without vector support for development
    
    async def store_embedding(
        self, 
        document_id: UUID, 
        embedding: List[float], 
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store document embedding"""
        try:
            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
            
            # Convert to vector format
            vector_str = f"[{','.join(map(str, embedding))}]"
            
            query = """
            UPDATE documents 
            SET vector_embedding = $1::vector, 
                embedding_metadata = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $3
            """
            
            result = await self.db_manager.execute(
                query, 
                vector_str, 
                json.dumps(metadata or {}),
                str(document_id)
            )
            
            return "UPDATE 1" in result
            
        except Exception as e:
            logger.error(f"Failed to store embedding for document {document_id}: {e}")
            return False
    
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using cosine similarity"""
        try:
            # Validate embedding dimension
            if len(query_embedding) != self.embedding_dimension:
                raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}")
            
            # Convert to vector format
            query_vector = f"[{','.join(map(str, query_embedding))}]"
            
            # Build query with optional filters
            where_conditions = ["vector_embedding IS NOT NULL"]
            params = [query_vector, similarity_threshold, limit]
            param_count = 3
            
            if filters:
                if 'company' in filters:
                    param_count += 1
                    where_conditions.append(f"company = ${param_count}")
                    params.append(filters['company'])
                
                if 'document_type' in filters:
                    param_count += 1
                    where_conditions.append(f"document_type = ${param_count}")
                    params.append(filters['document_type'])
                
                if 'date_from' in filters:
                    param_count += 1
                    where_conditions.append(f"filing_date >= ${param_count}")
                    params.append(filters['date_from'])
                
                if 'date_to' in filters:
                    param_count += 1
                    where_conditions.append(f"filing_date <= ${param_count}")
                    params.append(filters['date_to'])
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
            SELECT 
                id,
                company,
                document_type,
                filing_date,
                period,
                1 - (vector_embedding <=> $1::vector) as similarity,
                SUBSTRING(content, 1, 500) as snippet
            FROM documents 
            WHERE {where_clause}
            AND 1 - (vector_embedding <=> $1::vector) >= $2
            ORDER BY vector_embedding <=> $1::vector
            LIMIT $3
            """
            
            rows = await self.db_manager.fetch(query, *params)
            
            results = []
            for row in rows:
                results.append({
                    'document_id': str(row['id']),
                    'company': row['company'],
                    'document_type': row['document_type'],
                    'filing_date': row['filing_date'],
                    'period': row['period'],
                    'similarity': float(row['similarity']),
                    'snippet': row['snippet']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def find_similar_documents(
        self, 
        document_id: UUID, 
        limit: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            # Get the embedding for the source document
            query = "SELECT vector_embedding FROM documents WHERE id = $1"
            row = await self.db_manager.fetchrow(query, str(document_id))
            
            if not row or not row['vector_embedding']:
                return []
            
            # Convert vector back to list for similarity search
            # Note: This is a simplified approach - in production you'd use direct vector operations
            source_embedding = list(row['vector_embedding'])
            
            # Perform similarity search excluding the source document
            results = await self.similarity_search(
                source_embedding, 
                limit + 1,  # +1 to account for excluding source
                similarity_threshold
            )
            
            # Filter out the source document
            return [r for r in results if r['document_id'] != str(document_id)][:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar documents for {document_id}: {e}")
            return []
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_documents,
                COUNT(vector_embedding) as documents_with_embeddings,
                COUNT(CASE WHEN vector_embedding IS NULL THEN 1 END) as documents_without_embeddings,
                AVG(CASE WHEN vector_embedding IS NOT NULL THEN 1.0 ELSE 0.0 END) as embedding_coverage
            FROM documents
            """
            
            row = await self.db_manager.fetchrow(query)
            
            return {
                'total_documents': row['total_documents'],
                'documents_with_embeddings': row['documents_with_embeddings'],
                'documents_without_embeddings': row['documents_without_embeddings'],
                'embedding_coverage': float(row['embedding_coverage']) if row['embedding_coverage'] else 0.0,
                'embedding_dimension': self.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {}
    
    async def batch_store_embeddings(
        self, 
        embeddings_data: List[Tuple[UUID, List[float], Dict[str, Any]]]
    ) -> int:
        """Batch store multiple embeddings"""
        try:
            successful_updates = 0
            
            async with self.db_manager.get_transaction() as conn:
                for document_id, embedding, metadata in embeddings_data:
                    try:
                        # Validate embedding dimension
                        if len(embedding) != self.embedding_dimension:
                            logger.warning(f"Skipping embedding for {document_id}: dimension mismatch")
                            continue
                        
                        # Convert to vector format
                        vector_str = f"[{','.join(map(str, embedding))}]"
                        
                        result = await conn.execute(
                            """
                            UPDATE documents 
                            SET vector_embedding = $1::vector, 
                                embedding_metadata = $2,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = $3
                            """,
                            vector_str,
                            json.dumps(metadata or {}),
                            str(document_id)
                        )
                        
                        if "UPDATE 1" in result:
                            successful_updates += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to store embedding for document {document_id}: {e}")
                        continue
            
            logger.info(f"Successfully stored {successful_updates}/{len(embeddings_data)} embeddings")
            return successful_updates
            
        except Exception as e:
            logger.error(f"Batch embedding storage failed: {e}")
            return 0
    
    async def delete_embedding(self, document_id: UUID) -> bool:
        """Delete embedding for a document"""
        try:
            query = """
            UPDATE documents 
            SET vector_embedding = NULL, 
                embedding_metadata = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """
            
            result = await self.db_manager.execute(query, str(document_id))
            return "UPDATE 1" in result
            
        except Exception as e:
            logger.error(f"Failed to delete embedding for document {document_id}: {e}")
            return False
    
    async def reindex_embeddings(self, batch_size: int = 100):
        """Reindex all embeddings (useful for dimension changes)"""
        try:
            # Get documents without embeddings
            query = """
            SELECT id, content 
            FROM documents 
            WHERE vector_embedding IS NULL 
            ORDER BY created_at DESC
            LIMIT $1
            """
            
            rows = await self.db_manager.fetch(query, batch_size)
            
            logger.info(f"Found {len(rows)} documents without embeddings")
            
            # This would typically integrate with your embedding service
            # For now, we'll just log the documents that need reindexing
            document_ids = [row['id'] for row in rows]
            
            return {
                'documents_to_reindex': len(document_ids),
                'document_ids': document_ids
            }
            
        except Exception as e:
            logger.error(f"Reindexing failed: {e}")
            return {'error': str(e)}