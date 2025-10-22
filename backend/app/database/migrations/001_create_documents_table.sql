-- UP
-- Create documents table for financial document storage
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(10) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    filing_date TIMESTAMP NOT NULL,
    period VARCHAR(20),
    source VARCHAR(200) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    page_count INTEGER,
    file_size INTEGER,
    
    -- Content fields
    raw_content TEXT NOT NULL,
    processed_content TEXT,
    entities JSONB,
    key_metrics JSONB,
    summary TEXT,
    
    -- Vector embedding for similarity search
    vector_embedding VECTOR(768),
    embedding_metadata JSONB,
    
    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX idx_documents_company ON documents(company);
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_filing_date ON documents(filing_date);
CREATE INDEX idx_documents_company_type ON documents(company, document_type);
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_created_at ON documents(created_at);

-- Create GIN index for JSONB fields
CREATE INDEX idx_documents_entities ON documents USING GIN(entities);
CREATE INDEX idx_documents_key_metrics ON documents USING GIN(key_metrics);

-- Create vector similarity index (if pgvector is available)
-- This will be created conditionally by the vector store initialization
-- CREATE INDEX idx_documents_vector_embedding ON documents USING ivfflat (vector_embedding vector_cosine_ops);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP INDEX IF EXISTS idx_documents_vector_embedding;
DROP INDEX IF EXISTS idx_documents_key_metrics;
DROP INDEX IF EXISTS idx_documents_entities;
DROP INDEX IF EXISTS idx_documents_created_at;
DROP INDEX IF EXISTS idx_documents_status;
DROP INDEX IF EXISTS idx_documents_company_type;
DROP INDEX IF EXISTS idx_documents_filing_date;
DROP INDEX IF EXISTS idx_documents_type;
DROP INDEX IF EXISTS idx_documents_company;
DROP TABLE IF EXISTS documents;