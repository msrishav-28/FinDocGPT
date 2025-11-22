-- UP
-- Create sentiment analysis table
CREATE TABLE sentiment_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    text_snippet TEXT,
    
    -- Sentiment scores
    overall_sentiment DECIMAL(3,2) NOT NULL CHECK (overall_sentiment >= -1.0 AND overall_sentiment <= 1.0),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    polarity VARCHAR(20) NOT NULL,
    
    -- Topic-specific sentiments
    topic_sentiments JSONB DEFAULT '{}',
    
    -- Analysis details
    sentiment_explanation TEXT,
    model_used VARCHAR(100) NOT NULL,
    processing_time DECIMAL(8,3),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_sentiment_document_id ON sentiment_analysis(document_id);
CREATE INDEX idx_sentiment_overall_sentiment ON sentiment_analysis(overall_sentiment);
CREATE INDEX idx_sentiment_confidence ON sentiment_analysis(confidence);
CREATE INDEX idx_sentiment_polarity ON sentiment_analysis(polarity);
CREATE INDEX idx_sentiment_model_used ON sentiment_analysis(model_used);
CREATE INDEX idx_sentiment_created_at ON sentiment_analysis(created_at);

-- Create GIN index for topic sentiments
CREATE INDEX idx_sentiment_topic_sentiments ON sentiment_analysis USING GIN(topic_sentiments);

-- Create trigger for updated_at
CREATE TRIGGER update_sentiment_analysis_updated_at 
    BEFORE UPDATE ON sentiment_analysis 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create sentiment trends table for historical tracking
CREATE TABLE sentiment_trends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(10) NOT NULL,
    topic VARCHAR(50),
    date DATE NOT NULL,
    sentiment_score DECIMAL(3,2) NOT NULL CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    document_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sentiment trends
CREATE INDEX idx_sentiment_trends_company ON sentiment_trends(company);
CREATE INDEX idx_sentiment_trends_topic ON sentiment_trends(topic);
CREATE INDEX idx_sentiment_trends_date ON sentiment_trends(date);
CREATE INDEX idx_sentiment_trends_company_date ON sentiment_trends(company, date);
CREATE UNIQUE INDEX idx_sentiment_trends_unique ON sentiment_trends(company, topic, date);

-- Create sentiment alerts table
CREATE TABLE sentiment_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(10) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    current_sentiment DECIMAL(3,2) NOT NULL,
    previous_sentiment DECIMAL(3,2) NOT NULL,
    change_magnitude DECIMAL(3,2) NOT NULL,
    significance_level DECIMAL(3,2) NOT NULL CHECK (significance_level >= 0.0 AND significance_level <= 1.0),
    description TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sentiment alerts
CREATE INDEX idx_sentiment_alerts_company ON sentiment_alerts(company);
CREATE INDEX idx_sentiment_alerts_type ON sentiment_alerts(alert_type);
CREATE INDEX idx_sentiment_alerts_resolved ON sentiment_alerts(is_resolved);
CREATE INDEX idx_sentiment_alerts_created_at ON sentiment_alerts(created_at);

-- Create trigger for sentiment alerts updated_at
CREATE TRIGGER update_sentiment_alerts_updated_at 
    BEFORE UPDATE ON sentiment_alerts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_sentiment_alerts_updated_at ON sentiment_alerts;
DROP INDEX IF EXISTS idx_sentiment_alerts_created_at;
DROP INDEX IF EXISTS idx_sentiment_alerts_resolved;
DROP INDEX IF EXISTS idx_sentiment_alerts_type;
DROP INDEX IF EXISTS idx_sentiment_alerts_company;
DROP TABLE IF EXISTS sentiment_alerts;

DROP INDEX IF EXISTS idx_sentiment_trends_unique;
DROP INDEX IF EXISTS idx_sentiment_trends_company_date;
DROP INDEX IF EXISTS idx_sentiment_trends_date;
DROP INDEX IF EXISTS idx_sentiment_trends_topic;
DROP INDEX IF EXISTS idx_sentiment_trends_company;
DROP TABLE IF EXISTS sentiment_trends;

DROP TRIGGER IF EXISTS update_sentiment_analysis_updated_at ON sentiment_analysis;
DROP INDEX IF EXISTS idx_sentiment_topic_sentiments;
DROP INDEX IF EXISTS idx_sentiment_created_at;
DROP INDEX IF EXISTS idx_sentiment_model_used;
DROP INDEX IF EXISTS idx_sentiment_polarity;
DROP INDEX IF EXISTS idx_sentiment_confidence;
DROP INDEX IF EXISTS idx_sentiment_overall_sentiment;
DROP INDEX IF EXISTS idx_sentiment_document_id;
DROP TABLE IF EXISTS sentiment_analysis;