-- UP
-- Create users table for authentication and authorization
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    username VARCHAR(50) NOT NULL UNIQUE,
    full_name VARCHAR(200),
    
    -- Authentication
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP,
    
    -- Authorization
    role VARCHAR(50) DEFAULT 'viewer' CHECK (role IN ('admin', 'analyst', 'portfolio_manager', 'risk_manager', 'viewer')),
    permissions JSONB DEFAULT '[]',
    
    -- Subscription
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'basic', 'professional', 'enterprise')),
    subscription_expires TIMESTAMP,
    
    -- Profile
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    avatar_url VARCHAR(500),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_subscription_tier ON users(subscription_tier);
CREATE INDEX idx_users_is_active ON users(is_active);
CREATE INDEX idx_users_is_verified ON users(is_verified);
CREATE INDEX idx_users_last_login ON users(last_login);

-- Create GIN index for permissions
CREATE INDEX idx_users_permissions ON users USING GIN(permissions);

-- Create trigger for users updated_at
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create user preferences table
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Dashboard preferences
    default_dashboard VARCHAR(100),
    dashboard_layout JSONB DEFAULT '{}',
    chart_preferences JSONB DEFAULT '{}',
    
    -- Notification preferences
    email_notifications BOOLEAN DEFAULT TRUE,
    push_notifications BOOLEAN DEFAULT TRUE,
    alert_frequency VARCHAR(20) DEFAULT 'immediate' CHECK (alert_frequency IN ('immediate', 'daily', 'weekly')),
    
    -- Analysis preferences
    default_time_horizon VARCHAR(20) DEFAULT 'medium_term',
    risk_tolerance VARCHAR(20) DEFAULT 'medium',
    preferred_analysis_depth VARCHAR(20) DEFAULT 'standard',
    
    -- Data preferences
    preferred_data_sources JSONB DEFAULT '[]',
    currency VARCHAR(10) DEFAULT 'USD',
    date_format VARCHAR(20) DEFAULT 'YYYY-MM-DD',
    
    -- Privacy settings
    data_sharing_consent BOOLEAN DEFAULT FALSE,
    analytics_tracking BOOLEAN DEFAULT TRUE,
    marketing_consent BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for user preferences
CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX idx_user_preferences_default_dashboard ON user_preferences(default_dashboard);
CREATE INDEX idx_user_preferences_alert_frequency ON user_preferences(alert_frequency);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_user_preferences_dashboard_layout ON user_preferences USING GIN(dashboard_layout);
CREATE INDEX idx_user_preferences_chart_preferences ON user_preferences USING GIN(chart_preferences);
CREATE INDEX idx_user_preferences_data_sources ON user_preferences USING GIN(preferred_data_sources);

-- Create trigger for user preferences updated_at
CREATE TRIGGER update_user_preferences_updated_at 
    BEFORE UPDATE ON user_preferences 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create watchlists table
CREATE TABLE watchlists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Watchlist content
    tickers JSONB DEFAULT '[]',
    is_default BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    tags JSONB DEFAULT '[]',
    color VARCHAR(20),
    sort_order INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for watchlists
CREATE INDEX idx_watchlists_user_id ON watchlists(user_id);
CREATE INDEX idx_watchlists_name ON watchlists(name);
CREATE INDEX idx_watchlists_is_default ON watchlists(is_default);
CREATE INDEX idx_watchlists_is_public ON watchlists(is_public);
CREATE INDEX idx_watchlists_sort_order ON watchlists(sort_order);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_watchlists_tickers ON watchlists USING GIN(tickers);
CREATE INDEX idx_watchlists_tags ON watchlists USING GIN(tags);

-- Create trigger for watchlists updated_at
CREATE TRIGGER update_watchlists_updated_at 
    BEFORE UPDATE ON watchlists 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create user sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    refresh_token VARCHAR(255),
    
    -- Session details
    ip_address INET,
    user_agent TEXT,
    device_type VARCHAR(50),
    
    -- Session lifecycle
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Security
    login_method VARCHAR(20) DEFAULT 'password' CHECK (login_method IN ('password', 'oauth', 'sso')),
    is_suspicious BOOLEAN DEFAULT FALSE,
    failed_attempts INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for user sessions
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_session_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_is_active ON user_sessions(is_active);
CREATE INDEX idx_user_sessions_last_activity ON user_sessions(last_activity);
CREATE INDEX idx_user_sessions_is_suspicious ON user_sessions(is_suspicious);

-- Create trigger for user sessions updated_at
CREATE TRIGGER update_user_sessions_updated_at 
    BEFORE UPDATE ON user_sessions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create user activity table for audit logging
CREATE TABLE user_activity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    
    -- Activity context
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    
    -- Activity data
    metadata JSONB DEFAULT '{}',
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    -- Performance
    duration_ms INTEGER CHECK (duration_ms >= 0),
    response_size INTEGER CHECK (response_size >= 0),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for user activity
CREATE INDEX idx_user_activity_user_id ON user_activity(user_id);
CREATE INDEX idx_user_activity_type ON user_activity(activity_type);
CREATE INDEX idx_user_activity_resource_type ON user_activity(resource_type);
CREATE INDEX idx_user_activity_success ON user_activity(success);
CREATE INDEX idx_user_activity_created_at ON user_activity(created_at);

-- Create GIN index for metadata
CREATE INDEX idx_user_activity_metadata ON user_activity USING GIN(metadata);

-- Create API keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    
    -- Access control
    permissions JSONB DEFAULT '[]',
    rate_limit INTEGER, -- requests per minute
    allowed_ips JSONB DEFAULT '[]',
    
    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    usage_count INTEGER DEFAULT 0 CHECK (usage_count >= 0),
    
    -- Security
    created_by UUID REFERENCES users(id),
    revoked_at TIMESTAMP,
    revoked_by UUID REFERENCES users(id),
    revocation_reason TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for API keys
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at);
CREATE INDEX idx_api_keys_last_used ON api_keys(last_used);
CREATE INDEX idx_api_keys_created_by ON api_keys(created_by);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_api_keys_permissions ON api_keys USING GIN(permissions);
CREATE INDEX idx_api_keys_allowed_ips ON api_keys USING GIN(allowed_ips);

-- Create trigger for API keys updated_at
CREATE TRIGGER update_api_keys_updated_at 
    BEFORE UPDATE ON api_keys 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create user notifications table
CREATE TABLE user_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    
    -- Notification details
    notification_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    category VARCHAR(50) NOT NULL,
    
    -- Delivery
    delivery_channels JSONB DEFAULT '[]',
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    
    -- Action
    action_url VARCHAR(500),
    action_text VARCHAR(100),
    expires_at TIMESTAMP,
    
    -- Metadata
    source_id VARCHAR(100),
    source_type VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for user notifications
CREATE INDEX idx_user_notifications_user_id ON user_notifications(user_id);
CREATE INDEX idx_user_notifications_type ON user_notifications(notification_type);
CREATE INDEX idx_user_notifications_priority ON user_notifications(priority);
CREATE INDEX idx_user_notifications_category ON user_notifications(category);
CREATE INDEX idx_user_notifications_is_read ON user_notifications(is_read);
CREATE INDEX idx_user_notifications_expires_at ON user_notifications(expires_at);
CREATE INDEX idx_user_notifications_created_at ON user_notifications(created_at);

-- Create GIN indexes for JSONB fields
CREATE INDEX idx_user_notifications_delivery_channels ON user_notifications USING GIN(delivery_channels);
CREATE INDEX idx_user_notifications_metadata ON user_notifications USING GIN(metadata);

-- Create trigger for user notifications updated_at
CREATE TRIGGER update_user_notifications_updated_at 
    BEFORE UPDATE ON user_notifications 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- DOWN
DROP TRIGGER IF EXISTS update_user_notifications_updated_at ON user_notifications;
DROP INDEX IF EXISTS idx_user_notifications_metadata;
DROP INDEX IF EXISTS idx_user_notifications_delivery_channels;
DROP INDEX IF EXISTS idx_user_notifications_created_at;
DROP INDEX IF EXISTS idx_user_notifications_expires_at;
DROP INDEX IF EXISTS idx_user_notifications_is_read;
DROP INDEX IF EXISTS idx_user_notifications_category;
DROP INDEX IF EXISTS idx_user_notifications_priority;
DROP INDEX IF EXISTS idx_user_notifications_type;
DROP INDEX IF EXISTS idx_user_notifications_user_id;
DROP TABLE IF EXISTS user_notifications;

DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
DROP INDEX IF EXISTS idx_api_keys_allowed_ips;
DROP INDEX IF EXISTS idx_api_keys_permissions;
DROP INDEX IF EXISTS idx_api_keys_created_by;
DROP INDEX IF EXISTS idx_api_keys_last_used;
DROP INDEX IF EXISTS idx_api_keys_expires_at;
DROP INDEX IF EXISTS idx_api_keys_is_active;
DROP INDEX IF EXISTS idx_api_keys_key_hash;
DROP INDEX IF EXISTS idx_api_keys_user_id;
DROP TABLE IF EXISTS api_keys;

DROP INDEX IF EXISTS idx_user_activity_metadata;
DROP INDEX IF EXISTS idx_user_activity_created_at;
DROP INDEX IF EXISTS idx_user_activity_success;
DROP INDEX IF EXISTS idx_user_activity_resource_type;
DROP INDEX IF EXISTS idx_user_activity_type;
DROP INDEX IF EXISTS idx_user_activity_user_id;
DROP TABLE IF EXISTS user_activity;

DROP TRIGGER IF EXISTS update_user_sessions_updated_at ON user_sessions;
DROP INDEX IF EXISTS idx_user_sessions_is_suspicious;
DROP INDEX IF EXISTS idx_user_sessions_last_activity;
DROP INDEX IF EXISTS idx_user_sessions_is_active;
DROP INDEX IF EXISTS idx_user_sessions_expires_at;
DROP INDEX IF EXISTS idx_user_sessions_session_token;
DROP INDEX IF EXISTS idx_user_sessions_user_id;
DROP TABLE IF EXISTS user_sessions;

DROP TRIGGER IF EXISTS update_watchlists_updated_at ON watchlists;
DROP INDEX IF EXISTS idx_watchlists_tags;
DROP INDEX IF EXISTS idx_watchlists_tickers;
DROP INDEX IF EXISTS idx_watchlists_sort_order;
DROP INDEX IF EXISTS idx_watchlists_is_public;
DROP INDEX IF EXISTS idx_watchlists_is_default;
DROP INDEX IF EXISTS idx_watchlists_name;
DROP INDEX IF EXISTS idx_watchlists_user_id;
DROP TABLE IF EXISTS watchlists;

DROP TRIGGER IF EXISTS update_user_preferences_updated_at ON user_preferences;
DROP INDEX IF EXISTS idx_user_preferences_data_sources;
DROP INDEX IF EXISTS idx_user_preferences_chart_preferences;
DROP INDEX IF EXISTS idx_user_preferences_dashboard_layout;
DROP INDEX IF EXISTS idx_user_preferences_alert_frequency;
DROP INDEX IF EXISTS idx_user_preferences_default_dashboard;
DROP INDEX IF EXISTS idx_user_preferences_user_id;
DROP TABLE IF EXISTS user_preferences;

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP INDEX IF EXISTS idx_users_permissions;
DROP INDEX IF EXISTS idx_users_last_login;
DROP INDEX IF EXISTS idx_users_is_verified;
DROP INDEX IF EXISTS idx_users_is_active;
DROP INDEX IF EXISTS idx_users_subscription_tier;
DROP INDEX IF EXISTS idx_users_role;
DROP INDEX IF EXISTS idx_users_username;
DROP INDEX IF EXISTS idx_users_email;
DROP TABLE IF EXISTS users;