-- Create anomalies table for storing detected anomalies
CREATE TABLE IF NOT EXISTS anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    current_value DECIMAL(20, 6) NOT NULL,
    expected_value DECIMAL(20, 6) NOT NULL,
    deviation_score DECIMAL(10, 6) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    anomaly_type VARCHAR(50) NOT NULL CHECK (anomaly_type IN (
        'statistical_outlier', 'pattern_deviation', 'trend_break', 
        'seasonal_anomaly', 'correlation_break', 'volatility_spike'
    )),
    status VARCHAR(20) NOT NULL DEFAULT 'detected' CHECK (status IN (
        'detected', 'investigating', 'explained', 'resolved', 'false_positive'
    )),
    explanation TEXT NOT NULL,
    historical_context TEXT,
    potential_causes JSONB DEFAULT '[]',
    detection_method VARCHAR(100) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    baseline_period VARCHAR(50),
    investigated_by VARCHAR(100),
    resolution_notes TEXT,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_anomalies_company ON anomalies(company);
CREATE INDEX IF NOT EXISTS idx_anomalies_metric ON anomalies(metric_name);
CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_anomalies_status ON anomalies(status);
CREATE INDEX IF NOT EXISTS idx_anomalies_created_at ON anomalies(created_at);
CREATE INDEX IF NOT EXISTS idx_anomalies_company_metric ON anomalies(company, metric_name);
CREATE INDEX IF NOT EXISTS idx_anomalies_company_status ON anomalies(company, status);

-- Create financial_metrics table for storing historical financial data
CREATE TABLE IF NOT EXISTS financial_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(20, 6) NOT NULL,
    period_date DATE NOT NULL,
    period_type VARCHAR(20) DEFAULT 'quarterly' CHECK (period_type IN ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')),
    source VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(company, metric_name, period_date, period_type)
);

-- Create indexes for financial_metrics
CREATE INDEX IF NOT EXISTS idx_financial_metrics_company ON financial_metrics(company);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_metric ON financial_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_date ON financial_metrics(period_date);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_company_metric ON financial_metrics(company, metric_name);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_company_date ON financial_metrics(company, period_date);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_anomalies_updated_at 
    BEFORE UPDATE ON anomalies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_financial_metrics_updated_at 
    BEFORE UPDATE ON financial_metrics 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample financial metrics data for testing
INSERT INTO financial_metrics (company, metric_name, value, period_date, period_type) VALUES
('AAPL', 'revenue', 123456.78, '2024-03-31', 'quarterly'),
('AAPL', 'revenue', 119876.54, '2023-12-31', 'quarterly'),
('AAPL', 'revenue', 117321.45, '2023-09-30', 'quarterly'),
('AAPL', 'profit_margin', 0.2543, '2024-03-31', 'quarterly'),
('AAPL', 'profit_margin', 0.2398, '2023-12-31', 'quarterly'),
('AAPL', 'profit_margin', 0.2456, '2023-09-30', 'quarterly'),
('MSFT', 'revenue', 98765.43, '2024-03-31', 'quarterly'),
('MSFT', 'revenue', 95432.10, '2023-12-31', 'quarterly'),
('MSFT', 'revenue', 92187.65, '2023-09-30', 'quarterly'),
('MSFT', 'profit_margin', 0.3210, '2024-03-31', 'quarterly'),
('MSFT', 'profit_margin', 0.3087, '2023-12-31', 'quarterly'),
('MSFT', 'profit_margin', 0.3156, '2023-09-30', 'quarterly')
ON CONFLICT (company, metric_name, period_date, period_type) DO NOTHING;

COMMENT ON TABLE anomalies IS 'Stores detected financial anomalies with risk assessment and resolution tracking';
COMMENT ON TABLE financial_metrics IS 'Stores historical financial metrics data for anomaly detection baseline calculations';