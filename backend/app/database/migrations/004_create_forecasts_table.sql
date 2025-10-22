-- UP
-- Create forecasts table for financial predictions
CREATE TABLE forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10),
    company VARCHAR(10),
    metric_name VARCHAR(100),
    forecast_type VARCHAR(50) NOT NULL,
    
    -- Forecast details
    horizon_days INTEGER NOT NULL CHECK (horizon_days > 0),
    predicted_value DECIMAL(20,4) NOT NULL,
    confidence_lower DECIMAL(20,4) NOT NULL,
    confidence_upper DECIMAL(20,4) NOT NULL,
    
    -- Model information
    model_used VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    ensemble_weights JSONB,
    
    -- Performance and metadata
    confidence DECIMAL(3,2) CHECK (confidence >= 0.0 AND confidence <= 1.0),
    feature_importance JSONB,
    data_sources JSONB DEFAULT '[]',
    training_period VARCHAR(50),
    last_training_date TIMESTAMP,
    
    -- Quality metrics
    forecast_accuracy DECIMAL(3,2) CHECK (forecast_accuracy >= 0.0 AND forecast_accuracy <= 1.0),
    uncertainty_level DECIMAL(3,2) NOT NULL CHECK (uncertainty_level >= 0.0 AND uncertainty_level <= 1.0),
    data_quality_score DECIMAL(3,2) NOT NULL CHECK (data_quality_score >= 0.0 AND data_quality_score <= 1.0),
    
    -- Scenario and assumptions
    base_scenario VARCHAR(50) DEFAULT 'base',
    assumptions JSONB DEFAULT '[]',
    external_factors JSONB,
    
    -- Actual values for performance tracking
    actual_value DECIMAL(20,4),
    accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0.0 AND accuracy_score <= 1.0),
    
    -- Timestamps
    forecast_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT check_ticker_or_company CHECK (ticker IS NOT NULL OR company IS NOT NULL),
    CONSTRAINT check_confidence_bounds CHECK (confidence_lower <= predicted_value AND predicted_value <= confidence_upper)
);

-- Create indexes for forecasts
CREATE INDEX idx_forecasts_ticker ON forecasts(ticker);
CREATE INDEX idx_forecasts_company ON forecasts(company);
CREATE INDEX idx_forecasts_metric_name ON forecasts(metric_name);
CREATE INDEX idx_forecasts_type ON forecasts(forecast_type);
CREATE INDEX idx_forecasts_model ON forecasts(model_used);
CREATE INDEX idx_forecasts_horizon ON forecasts(horizon_days);
CREATE INDEX idx_forecasts_target_date ON forecasts(target_date);
CREATE INDEX idx_forecasts_forecast_date ON forecasts(forecast_date);
CREATE INDEX idx_forecasts_ticker_type ON forecasts(ticker, forecast_type);
CREATE INDEX idx_forecasts_accuracy ON forecasts(accuracy_score);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_forecasts_data_sources ON forecasts USING GIN(data_sources);
CREATE INDEX idx_forecasts_feature_importance ON forecasts USING GIN(feature_importance);
CREATE INDEX idx_forecasts_assumptions ON forecasts USING GIN(assumptions);

-- Create trigger for updated_at
CREATE TRIGGER update_forecasts_updated_at 
    BEFORE UPDATE ON forecasts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create forecast performance table for detailed tracking
CREATE TABLE forecast_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id UUID REFERENCES forecasts(id) ON DELETE CASCADE,
    
    -- Performance metrics
    mae DECIMAL(10,4) NOT NULL CHECK (mae >= 0), -- Mean Absolute Error
    rmse DECIMAL(10,4) NOT NULL CHECK (rmse >= 0), -- Root Mean Square Error
    mape DECIMAL(5,2) NOT NULL CHECK (mape >= 0), -- Mean Absolute Percentage Error
    directional_accuracy DECIMAL(3,2) NOT NULL CHECK (directional_accuracy >= 0.0 AND directional_accuracy <= 1.0),
    
    -- Horizon-specific performance
    horizon_performance JSONB DEFAULT '{}',
    performance_trend VARCHAR(20) CHECK (performance_trend IN ('improving', 'degrading', 'stable')),
    
    -- Evaluation data
    actual_values JSONB NOT NULL,
    predicted_values JSONB NOT NULL,
    timestamps JSONB NOT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for forecast performance
CREATE INDEX idx_forecast_performance_forecast_id ON forecast_performance(forecast_id);
CREATE INDEX idx_forecast_performance_mae ON forecast_performance(mae);
CREATE INDEX idx_forecast_performance_rmse ON forecast_performance(rmse);
CREATE INDEX idx_forecast_performance_mape ON forecast_performance(mape);
CREATE INDEX idx_forecast_performance_directional ON forecast_performance(directional_accuracy);
CREATE INDEX idx_forecast_performance_trend ON forecast_performance(performance_trend);

-- Create trigger for forecast performance updated_at
CREATE TRIGGER update_forecast_performance_updated_at 
    BEFORE UPDATE ON forecast_performance 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create external data sources table
CREATE TABLE external_data_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name VARCHAR(100) NOT NULL UNIQUE,
    api_endpoint VARCHAR(500) NOT NULL,
    api_key_required BOOLEAN DEFAULT TRUE,
    rate_limit INTEGER, -- requests per minute
    data_types JSONB DEFAULT '[]',
    reliability_score DECIMAL(3,2) DEFAULT 1.0 CHECK (reliability_score >= 0.0 AND reliability_score <= 1.0),
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for external data sources
CREATE INDEX idx_external_sources_name ON external_data_sources(source_name);
CREATE INDEX idx_external_sources_active ON external_data_sources(is_active);
CREATE INDEX idx_external_sources_reliability ON external_data_sources(reliability_score);

-- Create trigger for external data sources updated_at
CREATE TRIGGER update_external_data_sources_updated_at 
    BEFORE UPDATE ON external_data_sources 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create forecast alerts table
CREATE TABLE forecast_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id UUID REFERENCES forecasts(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    message TEXT NOT NULL,
    
    -- Alert details
    current_forecast DECIMAL(20,4),
    previous_forecast DECIMAL(20,4),
    change_magnitude DECIMAL(10,4),
    confidence_change DECIMAL(3,2),
    
    -- Resolution
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP,
    resolution_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for forecast alerts
CREATE INDEX idx_forecast_alerts_forecast_id ON forecast_alerts(forecast_id);
CREATE INDEX idx_forecast_alerts_type ON forecast_alerts(alert_type);
CREATE INDEX idx_forecast_alerts_severity ON forecast_alerts(severity);
CREATE INDEX idx_forecast_alerts_acknowledged ON forecast_alerts(is_acknowledged);
CREATE INDEX idx_forecast_alerts_created_at ON forecast_alerts(created_at);

-- Create trigger for forecast alerts updated_at
CREATE TRIGGER update_forecast_alerts_updated_at 
    BEFORE UPDATE ON forecast_alerts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_forecast_alerts_updated_at ON forecast_alerts;
DROP INDEX IF EXISTS idx_forecast_alerts_created_at;
DROP INDEX IF EXISTS idx_forecast_alerts_acknowledged;
DROP INDEX IF EXISTS idx_forecast_alerts_severity;
DROP INDEX IF EXISTS idx_forecast_alerts_type;
DROP INDEX IF EXISTS idx_forecast_alerts_forecast_id;
DROP TABLE IF EXISTS forecast_alerts;

DROP TRIGGER IF EXISTS update_external_data_sources_updated_at ON external_data_sources;
DROP INDEX IF EXISTS idx_external_sources_reliability;
DROP INDEX IF EXISTS idx_external_sources_active;
DROP INDEX IF EXISTS idx_external_sources_name;
DROP TABLE IF EXISTS external_data_sources;

DROP TRIGGER IF EXISTS update_forecast_performance_updated_at ON forecast_performance;
DROP INDEX IF EXISTS idx_forecast_performance_trend;
DROP INDEX IF EXISTS idx_forecast_performance_directional;
DROP INDEX IF EXISTS idx_forecast_performance_mape;
DROP INDEX IF EXISTS idx_forecast_performance_rmse;
DROP INDEX IF EXISTS idx_forecast_performance_mae;
DROP INDEX IF EXISTS idx_forecast_performance_forecast_id;
DROP TABLE IF EXISTS forecast_performance;

DROP TRIGGER IF EXISTS update_forecasts_updated_at ON forecasts;
DROP INDEX IF EXISTS idx_forecasts_assumptions;
DROP INDEX IF EXISTS idx_forecasts_feature_importance;
DROP INDEX IF EXISTS idx_forecasts_data_sources;
DROP INDEX IF EXISTS idx_forecasts_accuracy;
DROP INDEX IF EXISTS idx_forecasts_ticker_type;
DROP INDEX IF EXISTS idx_forecasts_forecast_date;
DROP INDEX IF EXISTS idx_forecasts_target_date;
DROP INDEX IF EXISTS idx_forecasts_horizon;
DROP INDEX IF EXISTS idx_forecasts_model;
DROP INDEX IF EXISTS idx_forecasts_type;
DROP INDEX IF EXISTS idx_forecasts_metric_name;
DROP INDEX IF EXISTS idx_forecasts_company;
DROP INDEX IF EXISTS idx_forecasts_ticker;
DROP TABLE IF EXISTS forecasts;