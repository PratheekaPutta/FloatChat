-- ARGO Oceanographic Data Analysis System
-- Database Initialization Script
-- Creates all necessary tables for users, ARGO data, and system functionality

-- Enable PostGIS extension if not already enabled (for spatial queries)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create users table for authentication and role management
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('Admin', 'Researcher', 'Viewer')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Create user sessions table for session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for user sessions
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);

-- Create user permissions table for fine-grained access control
CREATE TABLE IF NOT EXISTS user_permissions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER REFERENCES users(id),
    UNIQUE(user_id, resource_type, operation)
);

-- Create ARGO metadata table for float information
CREATE TABLE IF NOT EXISTS argo_metadata (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20) NOT NULL,
    project_name VARCHAR(100),
    pi_name VARCHAR(100),
    platform_type VARCHAR(50),
    wmo_inst_type VARCHAR(10),
    date_creation TIMESTAMP WITH TIME ZONE,
    date_update TIMESTAMP WITH TIME ZONE,
    lat_min DECIMAL(8,5),
    lat_max DECIMAL(8,5),
    lon_min DECIMAL(9,5),
    lon_max DECIMAL(9,5),
    data_centre VARCHAR(10),
    institution VARCHAR(100),
    n_profiles INTEGER DEFAULT 0,
    n_levels INTEGER DEFAULT 0,
    uploaded_by INTEGER REFERENCES users(id),
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    file_checksum VARCHAR(64),
    file_size BIGINT,
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for ARGO metadata
CREATE INDEX IF NOT EXISTS idx_argo_metadata_platform ON argo_metadata(platform_number);
CREATE INDEX IF NOT EXISTS idx_argo_metadata_project ON argo_metadata(project_name);
CREATE INDEX IF NOT EXISTS idx_argo_metadata_date ON argo_metadata(date_creation);
CREATE INDEX IF NOT EXISTS idx_argo_metadata_location ON argo_metadata(lat_min, lat_max, lon_min, lon_max);
CREATE INDEX IF NOT EXISTS idx_argo_metadata_uploaded ON argo_metadata(uploaded_at);
CREATE INDEX IF NOT EXISTS idx_argo_metadata_status ON argo_metadata(processing_status);

-- Create spatial index for geographic queries if PostGIS is available
-- CREATE INDEX IF NOT EXISTS idx_argo_metadata_geom ON argo_metadata USING GIST (ST_MakeEnvelope(lon_min, lat_min, lon_max, lat_max, 4326));

-- Create ARGO profiles table for individual profile data
CREATE TABLE IF NOT EXISTS argo_profiles (
    id SERIAL PRIMARY KEY,
    metadata_id INTEGER REFERENCES argo_metadata(id) ON DELETE CASCADE,
    cycle_number INTEGER,
    direction CHAR(1) CHECK (direction IN ('A', 'D')), -- A=Ascending, D=Descending
    data_mode CHAR(1) CHECK (data_mode IN ('R', 'A', 'D')), -- R=Real-time, A=Adjusted, D=Delayed
    latitude DECIMAL(8,5),
    longitude DECIMAL(9,5),
    juld TIMESTAMP WITH TIME ZONE, -- Julian date converted to timestamp
    juld_qc CHAR(1),
    juld_location TIMESTAMP WITH TIME ZONE,
    position_qc CHAR(1),
    n_levels INTEGER DEFAULT 0,
    profile_pres_qc CHAR(1),
    profile_temp_qc CHAR(1),
    profile_psal_qc CHAR(1),
    vertical_sampling_scheme TEXT,
    config_mission_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for ARGO profiles
CREATE INDEX IF NOT EXISTS idx_argo_profiles_metadata ON argo_profiles(metadata_id);
CREATE INDEX IF NOT EXISTS idx_argo_profiles_cycle ON argo_profiles(cycle_number);
CREATE INDEX IF NOT EXISTS idx_argo_profiles_date ON argo_profiles(juld);
CREATE INDEX IF NOT EXISTS idx_argo_profiles_location ON argo_profiles(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_argo_profiles_data_mode ON argo_profiles(data_mode);
CREATE INDEX IF NOT EXISTS idx_argo_profiles_direction ON argo_profiles(direction);

-- Create spatial index for geographic queries
-- CREATE INDEX IF NOT EXISTS idx_argo_profiles_geom ON argo_profiles USING GIST (ST_Point(longitude, latitude));

-- Create ARGO measurements table for depth-resolved data
CREATE TABLE IF NOT EXISTS argo_measurements (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE,
    depth_level INTEGER NOT NULL,
    pressure DECIMAL(8,3), -- dbar
    pressure_qc CHAR(1),
    pressure_adjusted DECIMAL(8,3),
    pressure_adjusted_qc CHAR(1),
    pressure_adjusted_error DECIMAL(8,3),
    temperature DECIMAL(8,4), -- Celsius
    temperature_qc CHAR(1),
    temperature_adjusted DECIMAL(8,4),
    temperature_adjusted_qc CHAR(1),
    temperature_adjusted_error DECIMAL(8,4),
    salinity DECIMAL(8,4), -- PSU
    salinity_qc CHAR(1),
    salinity_adjusted DECIMAL(8,4),
    salinity_adjusted_qc CHAR(1),
    salinity_adjusted_error DECIMAL(8,4),
    -- BGC parameters
    doxy DECIMAL(8,3), -- μmol/kg
    doxy_qc CHAR(1),
    doxy_adjusted DECIMAL(8,3),
    doxy_adjusted_qc CHAR(1),
    doxy_adjusted_error DECIMAL(8,3),
    chla DECIMAL(8,4), -- mg/m^3
    chla_qc CHAR(1),
    chla_adjusted DECIMAL(8,4),
    chla_adjusted_qc CHAR(1),
    chla_adjusted_error DECIMAL(8,4),
    bbp700 DECIMAL(10,8), -- m^-1
    bbp700_qc CHAR(1),
    bbp700_adjusted DECIMAL(10,8),
    bbp700_adjusted_qc CHAR(1),
    bbp700_adjusted_error DECIMAL(10,8),
    ph_in_situ_total DECIMAL(8,4),
    ph_in_situ_total_qc CHAR(1),
    ph_in_situ_total_adjusted DECIMAL(8,4),
    ph_in_situ_total_adjusted_qc CHAR(1),
    ph_in_situ_total_adjusted_error DECIMAL(8,4),
    nitrate DECIMAL(8,3), -- μmol/kg
    nitrate_qc CHAR(1),
    nitrate_adjusted DECIMAL(8,3),
    nitrate_adjusted_qc CHAR(1),
    nitrate_adjusted_error DECIMAL(8,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for ARGO measurements
CREATE INDEX IF NOT EXISTS idx_argo_measurements_profile ON argo_measurements(profile_id);
CREATE INDEX IF NOT EXISTS idx_argo_measurements_depth ON argo_measurements(depth_level);
CREATE INDEX IF NOT EXISTS idx_argo_measurements_pressure ON argo_measurements(pressure);
CREATE INDEX IF NOT EXISTS idx_argo_measurements_temp ON argo_measurements(temperature);
CREATE INDEX IF NOT EXISTS idx_argo_measurements_sal ON argo_measurements(salinity);
CREATE INDEX IF NOT EXISTS idx_argo_measurements_doxy ON argo_measurements(doxy);
CREATE INDEX IF NOT EXISTS idx_argo_measurements_chla ON argo_measurements(chla);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_argo_measurements_temp_sal ON argo_measurements(temperature, salinity) WHERE temperature IS NOT NULL AND salinity IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_argo_measurements_pressure_temp ON argo_measurements(pressure, temperature) WHERE pressure IS NOT NULL AND temperature IS NOT NULL;

-- Create ARGO BGC parameters table for biogeochemical data
CREATE TABLE IF NOT EXISTS argo_bgc (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE,
    depth_level INTEGER NOT NULL,
    pressure DECIMAL(8,3),
    -- Additional BGC parameters
    cdom DECIMAL(8,4), -- ppb
    cdom_qc CHAR(1),
    turbidity DECIMAL(8,4), -- NTU
    turbidity_qc CHAR(1),
    cp660 DECIMAL(10,8), -- m^-1
    cp660_qc CHAR(1),
    down_irradiance380 DECIMAL(12,6), -- W/m^2/nm
    down_irradiance380_qc CHAR(1),
    down_irradiance412 DECIMAL(12,6),
    down_irradiance412_qc CHAR(1),
    down_irradiance490 DECIMAL(12,6),
    down_irradiance490_qc CHAR(1),
    downwelling_par DECIMAL(12,6), -- μmol photons/m^2/s
    downwelling_par_qc CHAR(1),
    bisulfide DECIMAL(8,3), -- μmol/kg
    bisulfide_qc CHAR(1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for BGC data
CREATE INDEX IF NOT EXISTS idx_argo_bgc_profile ON argo_bgc(profile_id);
CREATE INDEX IF NOT EXISTS idx_argo_bgc_depth ON argo_bgc(depth_level);
CREATE INDEX IF NOT EXISTS idx_argo_bgc_pressure ON argo_bgc(pressure);

-- Create data quality control table
CREATE TABLE IF NOT EXISTS argo_quality_control (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES argo_profiles(id) ON DELETE CASCADE,
    parameter_name VARCHAR(50) NOT NULL,
    qc_flag CHAR(1) NOT NULL,
    qc_description TEXT,
    qc_performed_by VARCHAR(100),
    qc_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    qc_comment TEXT
);

-- Create system logs table for audit trail
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id INTEGER,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for system logs
CREATE INDEX IF NOT EXISTS idx_system_logs_user ON system_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_system_logs_action ON system_logs(action);
CREATE INDEX IF NOT EXISTS idx_system_logs_date ON system_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_system_logs_resource ON system_logs(resource_type, resource_id);

-- Create vector embeddings table for search functionality
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50) NOT NULL, -- 'metadata', 'profile', 'summary'
    content_id INTEGER NOT NULL,
    embedding_vector REAL[] NOT NULL,
    text_content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for vector embeddings
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_type ON vector_embeddings(content_type);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_content ON vector_embeddings(content_type, content_id);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_date ON vector_embeddings(created_at);

-- Create chat history table for conversation management
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB,
    sql_query TEXT,
    query_results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for chat history
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_date ON chat_history(created_at);

-- Create configuration table for system settings
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    config_type VARCHAR(20) DEFAULT 'string' CHECK (config_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, config_type, description, is_public) VALUES
('app_version', '1.0.0', 'string', 'Application version', TRUE),
('max_file_size_mb', '100', 'integer', 'Maximum file upload size in MB', TRUE),
('default_query_limit', '1000', 'integer', 'Default limit for database queries', TRUE),
('vector_similarity_threshold', '0.3', 'float', 'Minimum similarity threshold for vector search', FALSE),
('enable_bgc_data', 'true', 'boolean', 'Enable biogeochemical data processing', TRUE),
('data_retention_days', '365', 'integer', 'Number of days to retain uploaded data', FALSE)
ON CONFLICT (config_key) DO NOTHING;

-- Create functions for data maintenance

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_argo_metadata_updated_at ON argo_metadata;
CREATE TRIGGER update_argo_metadata_updated_at
    BEFORE UPDATE ON argo_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_argo_profiles_updated_at ON argo_profiles;
CREATE TRIGGER update_argo_profiles_updated_at
    BEFORE UPDATE ON argo_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_vector_embeddings_updated_at ON vector_embeddings;
CREATE TRIGGER update_vector_embeddings_updated_at
    BEFORE UPDATE ON vector_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_system_config_updated_at ON system_config;
CREATE TRIGGER update_system_config_updated_at
    BEFORE UPDATE ON system_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up old sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < CURRENT_TIMESTAMP 
    OR (created_at < CURRENT_TIMESTAMP - INTERVAL '30 days' AND is_active = FALSE);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate profile statistics
CREATE OR REPLACE FUNCTION update_profile_statistics(metadata_record_id INTEGER)
RETURNS VOID AS $$
BEGIN
    UPDATE argo_metadata 
    SET 
        n_profiles = (
            SELECT COUNT(*) 
            FROM argo_profiles 
            WHERE metadata_id = metadata_record_id
        ),
        n_levels = (
            SELECT COALESCE(SUM(n_levels), 0) 
            FROM argo_profiles 
            WHERE metadata_id = metadata_record_id
        )
    WHERE id = metadata_record_id;
END;
$$ LANGUAGE plpgsql;

-- Create views for common queries

-- View for profile summary
CREATE OR REPLACE VIEW profile_summary AS
SELECT 
    p.id,
    m.platform_number,
    m.project_name,
    p.cycle_number,
    p.latitude,
    p.longitude,
    p.juld,
    p.data_mode,
    p.direction,
    p.n_levels,
    m.date_creation as float_deployment_date,
    COUNT(am.id) as measurement_count,
    COUNT(CASE WHEN am.temperature IS NOT NULL THEN 1 END) as temp_measurements,
    COUNT(CASE WHEN am.salinity IS NOT NULL THEN 1 END) as sal_measurements,
    COUNT(CASE WHEN am.doxy IS NOT NULL THEN 1 END) as oxy_measurements,
    COUNT(CASE WHEN am.chla IS NOT NULL THEN 1 END) as chla_measurements
FROM argo_profiles p
JOIN argo_metadata m ON p.metadata_id = m.id
LEFT JOIN argo_measurements am ON p.id = am.profile_id
GROUP BY p.id, m.platform_number, m.project_name, p.cycle_number, 
         p.latitude, p.longitude, p.juld, p.data_mode, p.direction, 
         p.n_levels, m.date_creation;

-- View for float trajectory
CREATE OR REPLACE VIEW float_trajectory AS
SELECT 
    m.platform_number,
    m.project_name,
    p.cycle_number,
    p.latitude,
    p.longitude,
    p.juld,
    p.data_mode,
    LAG(p.latitude) OVER (PARTITION BY m.platform_number ORDER BY p.juld) as prev_lat,
    LAG(p.longitude) OVER (PARTITION BY m.platform_number ORDER BY p.juld) as prev_lon,
    LAG(p.juld) OVER (PARTITION BY m.platform_number ORDER BY p.juld) as prev_date
FROM argo_profiles p
JOIN argo_metadata m ON p.metadata_id = m.id
WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
ORDER BY m.platform_number, p.juld;

-- View for data quality summary
CREATE OR REPLACE VIEW data_quality_summary AS
SELECT 
    m.platform_number,
    p.id as profile_id,
    p.data_mode,
    COUNT(am.id) as total_measurements,
    COUNT(CASE WHEN am.temperature_qc IN ('1', '2') THEN 1 END) as good_temp_measurements,
    COUNT(CASE WHEN am.salinity_qc IN ('1', '2') THEN 1 END) as good_sal_measurements,
    COUNT(CASE WHEN am.pressure_qc IN ('1', '2') THEN 1 END) as good_pres_measurements,
    ROUND(
        COUNT(CASE WHEN am.temperature_qc IN ('1', '2') THEN 1 END)::NUMERIC / 
        NULLIF(COUNT(CASE WHEN am.temperature IS NOT NULL THEN 1 END), 0) * 100, 
        2
    ) as temp_quality_percentage,
    ROUND(
        COUNT(CASE WHEN am.salinity_qc IN ('1', '2') THEN 1 END)::NUMERIC / 
        NULLIF(COUNT(CASE WHEN am.salinity IS NOT NULL THEN 1 END), 0) * 100, 
        2
    ) as sal_quality_percentage
FROM argo_profiles p
JOIN argo_metadata m ON p.metadata_id = m.id
LEFT JOIN argo_measurements am ON p.id = am.profile_id
GROUP BY m.platform_number, p.id, p.data_mode;

-- Grant permissions to application user (adjust username as needed)
-- Note: In production, create a specific application user with limited permissions
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO argo_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO argo_app_user;

-- Create indexes for performance optimization on large datasets
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_argo_measurements_composite 
ON argo_measurements(profile_id, depth_level, pressure) 
WHERE temperature IS NOT NULL OR salinity IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_argo_profiles_location_date 
ON argo_profiles(latitude, longitude, juld) 
WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND juld IS NOT NULL;

-- Add constraints for data validation
ALTER TABLE argo_profiles 
ADD CONSTRAINT check_latitude_range CHECK (latitude >= -90 AND latitude <= 90),
ADD CONSTRAINT check_longitude_range CHECK (longitude >= -180 AND longitude <= 180);

ALTER TABLE argo_measurements 
ADD CONSTRAINT check_pressure_positive CHECK (pressure >= 0),
ADD CONSTRAINT check_temperature_range CHECK (temperature >= -5 AND temperature <= 40),
ADD CONSTRAINT check_salinity_range CHECK (salinity >= 0 AND salinity <= 45);

-- Create materialized view for performance on common aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_data_summary AS
SELECT 
    DATE_TRUNC('month', p.juld) as month,
    COUNT(DISTINCT m.platform_number) as active_floats,
    COUNT(p.id) as total_profiles,
    COUNT(am.id) as total_measurements,
    AVG(am.temperature) as avg_temperature,
    AVG(am.salinity) as avg_salinity,
    MIN(p.latitude) as min_lat,
    MAX(p.latitude) as max_lat,
    MIN(p.longitude) as min_lon,
    MAX(p.longitude) as max_lon
FROM argo_profiles p
JOIN argo_metadata m ON p.metadata_id = m.id
LEFT JOIN argo_measurements am ON p.id = am.profile_id
WHERE p.juld IS NOT NULL
GROUP BY DATE_TRUNC('month', p.juld)
ORDER BY month DESC;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_monthly_summary_month ON monthly_data_summary(month);

-- Add comment documentation
COMMENT ON TABLE users IS 'User accounts for authentication and role-based access control';
COMMENT ON TABLE argo_metadata IS 'ARGO float metadata including deployment information and spatial/temporal bounds';
COMMENT ON TABLE argo_profiles IS 'Individual ARGO profiles with position and time information';
COMMENT ON TABLE argo_measurements IS 'Depth-resolved measurements from ARGO profiles including core and BGC parameters';
COMMENT ON TABLE vector_embeddings IS 'Vector embeddings for semantic search functionality';
COMMENT ON TABLE chat_history IS 'Conversation history for AI assistant interactions';

-- Final maintenance
ANALYZE;

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE 'ARGO oceanographic database initialization completed successfully!';
    RAISE NOTICE 'Tables created: users, argo_metadata, argo_profiles, argo_measurements, argo_bgc, and supporting tables';
    RAISE NOTICE 'Views created: profile_summary, float_trajectory, data_quality_summary';
    RAISE NOTICE 'Materialized view created: monthly_data_summary';
    RAISE NOTICE 'Please create application users and grant appropriate permissions';
END $$;
