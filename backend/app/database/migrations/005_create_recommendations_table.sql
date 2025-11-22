-- UP
-- Create investment recommendations table
CREATE TABLE investment_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL,
    signal VARCHAR(20) NOT NULL CHECK (signal IN ('STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Price and targets
    current_price DECIMAL(10,2) NOT NULL CHECK (current_price > 0),
    target_price DECIMAL(10,2) CHECK (target_price > 0),
    stop_loss DECIMAL(10,2) CHECK (stop_loss > 0),
    
    -- Risk and sizing
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('very_low', 'low', 'medium', 'high', 'very_high')),
    position_size DECIMAL(3,2) CHECK (position_size >= 0.0 AND position_size <= 1.0),
    max_position_size DECIMAL(3,2) CHECK (max_position_size >= 0.0 AND max_position_size <= 1.0),
    
    -- Reasoning and context
    reasoning TEXT NOT NULL,
    supporting_factors JSONB DEFAULT '[]',
    risk_factors JSONB DEFAULT '[]',
    time_horizon VARCHAR(20) NOT NULL CHECK (time_horizon IN ('short_term', 'medium_term', 'long_term')),
    
    -- Analysis inputs
    document_insights JSONB,
    sentiment_score DECIMAL(3,2) CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
    anomaly_flags JSONB DEFAULT '[]',
    forecast_data JSONB,
    
    -- Metadata
    analyst_id VARCHAR(100),
    model_version VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'expired', 'superseded', 'withdrawn')),
    expires_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for investment recommendations
CREATE INDEX idx_recommendations_ticker ON investment_recommendations(ticker);
CREATE INDEX idx_recommendations_signal ON investment_recommendations(signal);
CREATE INDEX idx_recommendations_confidence ON investment_recommendations(confidence);
CREATE INDEX idx_recommendations_risk_level ON investment_recommendations(risk_level);
CREATE INDEX idx_recommendations_time_horizon ON investment_recommendations(time_horizon);
CREATE INDEX idx_recommendations_status ON investment_recommendations(status);
CREATE INDEX idx_recommendations_analyst ON investment_recommendations(analyst_id);
CREATE INDEX idx_recommendations_created_at ON investment_recommendations(created_at);
CREATE INDEX idx_recommendations_expires_at ON investment_recommendations(expires_at);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_recommendations_supporting_factors ON investment_recommendations USING GIN(supporting_factors);
CREATE INDEX idx_recommendations_risk_factors ON investment_recommendations USING GIN(risk_factors);
CREATE INDEX idx_recommendations_document_insights ON investment_recommendations USING GIN(document_insights);
CREATE INDEX idx_recommendations_anomaly_flags ON investment_recommendations USING GIN(anomaly_flags);

-- Create trigger for updated_at
CREATE TRIGGER update_investment_recommendations_updated_at 
    BEFORE UPDATE ON investment_recommendations 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create recommendation explanations table
CREATE TABLE recommendation_explanations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id UUID REFERENCES investment_recommendations(id) ON DELETE CASCADE,
    executive_summary TEXT NOT NULL,
    
    -- Factor analysis
    fundamental_analysis TEXT,
    technical_analysis TEXT,
    sentiment_analysis TEXT,
    risk_analysis TEXT NOT NULL,
    
    -- Supporting data
    key_metrics JSONB DEFAULT '{}',
    peer_comparison JSONB,
    historical_performance JSONB,
    
    -- Model insights
    feature_importance JSONB DEFAULT '{}',
    model_confidence_factors JSONB DEFAULT '[]',
    alternative_scenarios JSONB DEFAULT '[]',
    
    -- Disclaimers and limitations
    assumptions JSONB DEFAULT '[]',
    limitations JSONB DEFAULT '[]',
    data_quality_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for recommendation explanations
CREATE INDEX idx_recommendation_explanations_rec_id ON recommendation_explanations(recommendation_id);
CREATE INDEX idx_recommendation_explanations_created_at ON recommendation_explanations(created_at);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_recommendation_explanations_key_metrics ON recommendation_explanations USING GIN(key_metrics);
CREATE INDEX idx_recommendation_explanations_feature_importance ON recommendation_explanations USING GIN(feature_importance);

-- Create trigger for recommendation explanations updated_at
CREATE TRIGGER update_recommendation_explanations_updated_at 
    BEFORE UPDATE ON recommendation_explanations 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create portfolios table
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    owner_id VARCHAR(100) NOT NULL,
    
    -- Holdings
    holdings JSONB DEFAULT '{}', -- ticker -> position size
    cash_position DECIMAL(15,2) DEFAULT 0.0 CHECK (cash_position >= 0),
    total_value DECIMAL(15,2) NOT NULL CHECK (total_value > 0),
    
    -- Portfolio metrics
    beta DECIMAL(5,3),
    sharpe_ratio DECIMAL(5,3),
    volatility DECIMAL(5,3) CHECK (volatility >= 0),
    max_drawdown DECIMAL(5,3),
    
    -- Risk management
    risk_tolerance VARCHAR(20) NOT NULL CHECK (risk_tolerance IN ('very_low', 'low', 'medium', 'high', 'very_high')),
    max_position_size DECIMAL(3,2) DEFAULT 0.1 CHECK (max_position_size >= 0.0 AND max_position_size <= 1.0),
    sector_limits JSONB DEFAULT '{}',
    
    -- Tracking
    benchmark VARCHAR(20),
    inception_date TIMESTAMP NOT NULL,
    last_rebalanced TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for portfolios
CREATE INDEX idx_portfolios_owner_id ON portfolios(owner_id);
CREATE INDEX idx_portfolios_name ON portfolios(name);
CREATE INDEX idx_portfolios_risk_tolerance ON portfolios(risk_tolerance);
CREATE INDEX idx_portfolios_benchmark ON portfolios(benchmark);
CREATE INDEX idx_portfolios_inception_date ON portfolios(inception_date);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_portfolios_holdings ON portfolios USING GIN(holdings);
CREATE INDEX idx_portfolios_sector_limits ON portfolios USING GIN(sector_limits);

-- Create trigger for portfolios updated_at
CREATE TRIGGER update_portfolios_updated_at 
    BEFORE UPDATE ON portfolios 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create recommendation performance table
CREATE TABLE recommendation_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id UUID REFERENCES investment_recommendations(id) ON DELETE CASCADE,
    
    -- Performance metrics
    actual_return DECIMAL(8,4),
    predicted_return DECIMAL(8,4),
    holding_period INTEGER, -- days held
    
    -- Risk metrics
    max_drawdown DECIMAL(5,3),
    volatility DECIMAL(5,3) CHECK (volatility >= 0),
    sharpe_ratio DECIMAL(5,3),
    
    -- Outcome tracking
    target_hit BOOLEAN,
    stop_loss_hit BOOLEAN,
    exit_reason VARCHAR(100),
    exit_date TIMESTAMP,
    
    -- Attribution
    performance_attribution JSONB DEFAULT '{}',
    factor_returns JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for recommendation performance
CREATE INDEX idx_recommendation_performance_rec_id ON recommendation_performance(recommendation_id);
CREATE INDEX idx_recommendation_performance_actual_return ON recommendation_performance(actual_return);
CREATE INDEX idx_recommendation_performance_holding_period ON recommendation_performance(holding_period);
CREATE INDEX idx_recommendation_performance_exit_date ON recommendation_performance(exit_date);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_recommendation_performance_attribution ON recommendation_performance USING GIN(performance_attribution);
CREATE INDEX idx_recommendation_performance_factor_returns ON recommendation_performance USING GIN(factor_returns);

-- Create trigger for recommendation performance updated_at
CREATE TRIGGER update_recommendation_performance_updated_at 
    BEFORE UPDATE ON recommendation_performance 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create recommendation alerts table
CREATE TABLE recommendation_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id UUID REFERENCES investment_recommendations(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    message TEXT NOT NULL,
    
    -- Alert triggers
    price_change DECIMAL(8,4),
    confidence_change DECIMAL(3,2),
    new_information TEXT,
    
    -- Delivery
    recipient_ids JSONB DEFAULT '[]',
    delivery_channels JSONB DEFAULT '[]',
    is_delivered BOOLEAN DEFAULT FALSE,
    delivered_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for recommendation alerts
CREATE INDEX idx_recommendation_alerts_rec_id ON recommendation_alerts(recommendation_id);
CREATE INDEX idx_recommendation_alerts_type ON recommendation_alerts(alert_type);
CREATE INDEX idx_recommendation_alerts_severity ON recommendation_alerts(severity);
CREATE INDEX idx_recommendation_alerts_delivered ON recommendation_alerts(is_delivered);
CREATE INDEX idx_recommendation_alerts_created_at ON recommendation_alerts(created_at);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_recommendation_alerts_recipients ON recommendation_alerts USING GIN(recipient_ids);
CREATE INDEX idx_recommendation_alerts_channels ON recommendation_alerts USING GIN(delivery_channels);

-- Create trigger for recommendation alerts updated_at
CREATE TRIGGER update_recommendation_alerts_updated_at 
    BEFORE UPDATE ON recommendation_alerts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_recommendation_alerts_updated_at ON recommendation_alerts;
DROP INDEX IF EXISTS idx_recommendation_alerts_channels;
DROP INDEX IF EXISTS idx_recommendation_alerts_recipients;
DROP INDEX IF EXISTS idx_recommendation_alerts_created_at;
DROP INDEX IF EXISTS idx_recommendation_alerts_delivered;
DROP INDEX IF EXISTS idx_recommendation_alerts_severity;
DROP INDEX IF EXISTS idx_recommendation_alerts_type;
DROP INDEX IF EXISTS idx_recommendation_alerts_rec_id;
DROP TABLE IF EXISTS recommendation_alerts;

DROP TRIGGER IF EXISTS update_recommendation_performance_updated_at ON recommendation_performance;
DROP INDEX IF EXISTS idx_recommendation_performance_factor_returns;
DROP INDEX IF EXISTS idx_recommendation_performance_attribution;
DROP INDEX IF EXISTS idx_recommendation_performance_exit_date;
DROP INDEX IF EXISTS idx_recommendation_performance_holding_period;
DROP INDEX IF EXISTS idx_recommendation_performance_actual_return;
DROP INDEX IF EXISTS idx_recommendation_performance_rec_id;
DROP TABLE IF EXISTS recommendation_performance;

DROP TRIGGER IF EXISTS update_portfolios_updated_at ON portfolios;
DROP INDEX IF EXISTS idx_portfolios_sector_limits;
DROP INDEX IF EXISTS idx_portfolios_holdings;
DROP INDEX IF EXISTS idx_portfolios_inception_date;
DROP INDEX IF EXISTS idx_portfolios_benchmark;
DROP INDEX IF EXISTS idx_portfolios_risk_tolerance;
DROP INDEX IF EXISTS idx_portfolios_name;
DROP INDEX IF EXISTS idx_portfolios_owner_id;
DROP TABLE IF EXISTS portfolios;

DROP TRIGGER IF EXISTS update_recommendation_explanations_updated_at ON recommendation_explanations;
DROP INDEX IF EXISTS idx_recommendation_explanations_feature_importance;
DROP INDEX IF EXISTS idx_recommendation_explanations_key_metrics;
DROP INDEX IF EXISTS idx_recommendation_explanations_created_at;
DROP INDEX IF EXISTS idx_recommendation_explanations_rec_id;
DROP TABLE IF EXISTS recommendation_explanations;

DROP TRIGGER IF EXISTS update_investment_recommendations_updated_at ON investment_recommendations;
DROP INDEX IF EXISTS idx_recommendations_anomaly_flags;
DROP INDEX IF EXISTS idx_recommendations_document_insights;
DROP INDEX IF EXISTS idx_recommendations_risk_factors;
DROP INDEX IF EXISTS idx_recommendations_supporting_factors;
DROP INDEX IF EXISTS idx_recommendations_expires_at;
DROP INDEX IF EXISTS idx_recommendations_created_at;
DROP INDEX IF EXISTS idx_recommendations_analyst;
DROP INDEX IF EXISTS idx_recommendations_status;
DROP INDEX IF EXISTS idx_recommendations_time_horizon;
DROP INDEX IF EXISTS idx_recommendations_risk_level;
DROP INDEX IF EXISTS idx_recommendations_confidence;
DROP INDEX IF EXISTS idx_recommendations_signal;
DROP INDEX IF EXISTS idx_recommendations_ticker;
DROP TABLE IF EXISTS investment_recommendations;