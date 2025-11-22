-- UP
-- Create document_relationships table for tracking relationships between documents
CREATE TABLE document_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    strength DECIMAL(3,2) NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
    description TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique relationships
    UNIQUE(source_document_id, target_document_id)
);

-- Create indexes for efficient querying
CREATE INDEX idx_document_relationships_source ON document_relationships(source_document_id);
CREATE INDEX idx_document_relationships_target ON document_relationships(target_document_id);
CREATE INDEX idx_document_relationships_type ON document_relationships(relationship_type);
CREATE INDEX idx_document_relationships_strength ON document_relationships(strength);

-- Create trigger to update updated_at timestamp
CREATE TRIGGER update_document_relationships_updated_at 
    BEFORE UPDATE ON document_relationships 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_document_relationships_updated_at ON document_relationships;
DROP INDEX IF EXISTS idx_document_relationships_strength;
DROP INDEX IF EXISTS idx_document_relationships_type;
DROP INDEX IF EXISTS idx_document_relationships_target;
DROP INDEX IF EXISTS idx_document_relationships_source;
DROP TABLE IF EXISTS document_relationships;