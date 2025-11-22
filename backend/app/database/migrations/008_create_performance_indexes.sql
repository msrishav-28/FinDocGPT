-- UP
-- Advanced performance indexes for optimal query performance

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_company_date_type 
ON documents(company, filing_date DESC, document_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_document_created 
ON sentiment_analysis(document_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_company_status_created 
ON anomalies(company, status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_ticker_horizon_target 
ON forecasts(ticker, horizon_days, target_date DESC);

-- Partial indexes for active/recent data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_active_processing 
ON documents(processing_status, created_at) 
WHERE processing_status IN ('pending', 'processing');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_unresolved 
ON anomalies(company, severity, created_at DESC) 
WHERE status IN ('detected', 'investigating');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_recent_accuracy 
ON forecasts(model_used, accuracy_score DESC) 
WHERE created_at >= NOW() - INTERVAL '30 days';

-- Expression indexes for computed values
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_abs_deviation 
ON anomalies(company, ABS(current_value - expected_value));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_confidence_width 
ON forecasts(ticker, (confidence_upper - confidence_lower));

-- Full-text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_fts 
ON documents USING gin(to_tsvector('english', processed_content));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_summary_fts 
ON documents USING gin(to_tsvector('english', summary));

-- Hash indexes for exact matches on high-cardinality columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_id_hash 
ON documents USING hash(id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_document_id_hash 
ON sentiment_analysis USING hash(document_id);

-- Additional composite indexes for complex queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_company_date_confidence 
ON sentiment_analysis(document_id) 
INCLUDE (overall_sentiment, confidence, created_at)
WHERE confidence >= 0.8;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_model_performance 
ON forecasts(model_used, horizon_days) 
INCLUDE (accuracy_score, confidence, created_at)
WHERE actual_value IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_severity_metric 
ON anomalies(severity, metric_name, company) 
INCLUDE (deviation_score, created_at)
WHERE status IN ('detected', 'investigating');

-- Indexes for time-series analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_trends_company_topic_date 
ON sentiment_trends(company, topic, date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecast_performance_model_horizon 
ON forecast_performance(forecast_id) 
INCLUDE (mae, rmse, mape, directional_accuracy);

-- Indexes for audit and compliance queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_created_company 
ON documents(created_at DESC, company) 
WHERE processing_status = 'completed';

-- Statistics and monitoring indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_analysis_model_performance 
ON sentiment_analysis(model_used, confidence DESC, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_accuracy_tracking 
ON forecasts(target_date, model_used) 
INCLUDE (predicted_value, actual_value, accuracy_score)
WHERE actual_value IS NOT NULL;

-- DOWN
-- Drop performance indexes
DROP INDEX CONCURRENTLY IF EXISTS idx_forecasts_accuracy_tracking;
DROP INDEX CONCURRENTLY IF EXISTS idx_sentiment_analysis_model_performance;
DROP INDEX CONCURRENTLY IF EXISTS idx_documents_created_company;
DROP INDEX CONCURRENTLY IF EXISTS idx_forecast_performance_model_horizon;
DROP INDEX CONCURRENTLY IF EXISTS idx_sentiment_trends_company_topic_date;
DROP INDEX CONCURRENTLY IF EXISTS idx_anomalies_severity_metric;
DROP INDEX CONCURRENTLY IF EXISTS idx_forecasts_model_performance;
DROP INDEX CONCURRENTLY IF EXISTS idx_sentiment_company_date_confidence;
DROP INDEX CONCURRENTLY IF EXISTS idx_sentiment_document_id_hash;
DROP INDEX CONCURRENTLY IF EXISTS idx_documents_id_hash;
DROP INDEX CONCURRENTLY IF EXISTS idx_documents_summary_fts;
DROP INDEX CONCURRENTLY IF EXISTS idx_documents_content_fts;
DROP INDEX CONCURRENTLY IF EXISTS idx_forecasts_confidence_width;
DROP INDEX CONCURRENTLY IF EXISTS idx_anomalies_abs_deviation;
DROP INDEX CONCURRENTLY IF EXISTS idx_forecasts_recent_accuracy;
DROP INDEX CONCURRENTLY IF EXISTS idx_anomalies_unresolved;
DROP INDEX CONCURRENTLY IF EXISTS idx_documents_active_processing;
DROP INDEX CONCURRENTLY IF EXISTS idx_forecasts_ticker_horizon_target;
DROP INDEX CONCURRENTLY IF EXISTS idx_anomalies_company_status_created;
DROP INDEX CONCURRENTLY IF EXISTS idx_sentiment_document_created;
DROP INDEX CONCURRENTLY IF EXISTS idx_documents_company_date_type;