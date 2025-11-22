-- UP
-- Create anomalies table for financial metric anomaly detection
CREATE TABLE anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(10) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    
    -- Anomaly values
    current_value DECIMAL(20,4) NOT NULL,
    expected_value DECIMAL(20,4) NOT NULL,
    deviation_score DECIMAL(10,4) NOT NULL CHECK (deviation_score >= 0),
    
    -- Classification
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    anomaly_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'detected' CHECK (status IN ('detected', 'investigating', 'explained', 'resolved', 'false_positive')),
    
    -- Context and explanation
    explanation TEXT NOT NULL,
    historical_context TEXT,
    potential_causes JSONB DEFAULT '[]',
    
    -- Detection details
    detection_method VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    baseline_period VARCHAR(50),
    
    -- Resolution tracking
    investigated_by VARCHAR(100),
    resolution_notes TEXT,
    resolved_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for anomalies
CREATE INDEX idx_anomalies_company ON anomalies(company);
CREATE INDEX idx_anomalies_metric_name ON anomalies(metric_name);
CREATE INDEX idx_anomalies_severity ON anomalies(severity);
CREATE INDEX idx_anomalies_status ON anomalies(status);
CREATE INDEX idx_anomalies_type ON anomalies(anomaly_type);
CREATE INDEX idx_anomalies_company_metric ON anomalies(company, metric_name);
CREATE INDEX idx_anomalies_created_at ON anomalies(created_at);
CREATE INDEX idx_anomalies_deviation_score ON anomalies(deviation_score);

-- Create GIN index for potential causes
CREATE INDEX idx_anomalies_potential_causes ON anomalies USING GIN(potential_causes);

-- Create trigger for updated_at
CREATE TRIGGER update_anomalies_updated_at 
    BEFORE UPDATE ON anomalies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create anomaly baselines table for tracking baseline models
CREATE TABLE anomaly_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company VARCHAR(10) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    baseline_type VARCHAR(50) NOT NULL,
    baseline_parameters JSONB DEFAULT '{}',
    baseline_period VARCHAR(50) NOT NULL,
    performance_metrics JSONB DEFAULT '{}',
    seasonal_adjustments JSONB,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create unique constraint for company-metric baseline
CREATE UNIQUE INDEX idx_anomaly_baselines_unique ON anomaly_baselines(company, metric_name);
CREATE INDEX idx_anomaly_baselines_company ON anomaly_baselines(company);
CREATE INDEX idx_anomaly_baselines_metric ON anomaly_baselines(metric_name);
CREATE INDEX idx_anomaly_baselines_type ON anomaly_baselines(baseline_type);
CREATE INDEX idx_anomaly_baselines_updated ON anomaly_baselines(last_updated);

-- Create anomaly correlations table
CREATE TABLE anomaly_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    primary_anomaly_id UUID REFERENCES anomalies(id) ON DELETE CASCADE,
    correlated_anomaly_ids UUID[] NOT NULL,
    correlation_strength DECIMAL(3,2) NOT NULL CHECK (correlation_strength >= -1.0 AND correlation_strength <= 1.0),
    correlation_type VARCHAR(50) NOT NULL,
    time_lag DECIMAL(8,2), -- in hours
    statistical_significance DECIMAL(3,2) NOT NULL CHECK (statistical_significance >= 0.0 AND statistical_significance <= 1.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for anomaly correlations
CREATE INDEX idx_anomaly_correlations_primary ON anomaly_correlations(primary_anomaly_id);
CREATE INDEX idx_anomaly_correlations_strength ON anomaly_correlations(correlation_strength);
CREATE INDEX idx_anomaly_correlations_type ON anomaly_correlations(correlation_type);
CREATE INDEX idx_anomaly_correlations_significance ON anomaly_correlations(statistical_significance);

-- Create risk assessments table
CREATE TABLE risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    anomaly_id UUID REFERENCES anomalies(id) ON DELETE CASCADE,
    risk_score DECIMAL(3,2) NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    financial_impact TEXT,
    probability_of_impact DECIMAL(3,2) NOT NULL CHECK (probability_of_impact >= 0.0 AND probability_of_impact <= 1.0),
    time_horizon VARCHAR(50) NOT NULL,
    
    -- Risk factors
    market_risk DECIMAL(3,2) NOT NULL CHECK (market_risk >= 0.0 AND market_risk <= 1.0),
    operational_risk DECIMAL(3,2) NOT NULL CHECK (operational_risk >= 0.0 AND operational_risk <= 1.0),
    regulatory_risk DECIMAL(3,2) NOT NULL CHECK (regulatory_risk >= 0.0 AND regulatory_risk <= 1.0),
    reputational_risk DECIMAL(3,2) NOT NULL CHECK (reputational_risk >= 0.0 AND reputational_risk <= 1.0),
    
    -- Mitigation
    recommended_actions JSONB DEFAULT '[]',
    monitoring_recommendations JSONB DEFAULT '[]',
    escalation_threshold DECIMAL(3,2) CHECK (escalation_threshold >= 0.0 AND escalation_threshold <= 1.0),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for risk assessments
CREATE INDEX idx_risk_assessments_anomaly_id ON risk_assessments(anomaly_id);
CREATE INDEX idx_risk_assessments_risk_score ON risk_assessments(risk_score);
CREATE INDEX idx_risk_assessments_probability ON risk_assessments(probability_of_impact);
CREATE INDEX idx_risk_assessments_created_at ON risk_assessments(created_at);

-- Create trigger for risk assessments updated_at
CREATE TRIGGER update_risk_assessments_updated_at 
    BEFORE UPDATE ON risk_assessments 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_risk_assessments_updated_at ON risk_assessments;
DROP INDEX IF EXISTS idx_risk_assessments_created_at;
DROP INDEX IF EXISTS idx_risk_assessments_probability;
DROP INDEX IF EXISTS idx_risk_assessments_risk_score;
DROP INDEX IF EXISTS idx_risk_assessments_anomaly_id;
DROP TABLE IF EXISTS risk_assessments;

DROP INDEX IF EXISTS idx_anomaly_correlations_significance;
DROP INDEX IF EXISTS idx_anomaly_correlations_type;
DROP INDEX IF EXISTS idx_anomaly_correlations_strength;
DROP INDEX IF EXISTS idx_anomaly_correlations_primary;
DROP TABLE IF EXISTS anomaly_correlations;

DROP INDEX IF EXISTS idx_anomaly_baselines_updated;
DROP INDEX IF EXISTS idx_anomaly_baselines_type;
DROP INDEX IF EXISTS idx_anomaly_baselines_metric;
DROP INDEX IF EXISTS idx_anomaly_baselines_company;
DROP INDEX IF EXISTS idx_anomaly_baselines_unique;
DROP TABLE IF EXISTS anomaly_baselines;

DROP TRIGGER IF EXISTS update_anomalies_updated_at ON anomalies;
DROP INDEX IF EXISTS idx_anomalies_potential_causes;
DROP INDEX IF EXISTS idx_anomalies_deviation_score;
DROP INDEX IF EXISTS idx_anomalies_created_at;
DROP INDEX IF EXISTS idx_anomalies_company_metric;
DROP INDEX IF EXISTS idx_anomalies_type;
DROP INDEX IF EXISTS idx_anomalies_status;
DROP INDEX IF EXISTS idx_anomalies_severity;
DROP INDEX IF EXISTS idx_anomalies_metric_name;
DROP INDEX IF EXISTS idx_anomalies_company;
DROP TABLE IF EXISTS anomalies;