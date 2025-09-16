import pandas as pd
import numpy as np
from sqlalchemy import text
from config.database import get_sqlalchemy_engine, execute_query
from config.settings import settings
import streamlit as st

class DatabaseManager:
    """Manage database operations for ARGO data"""
    
    def __init__(self):
        self.engine = get_sqlalchemy_engine()
    
    def execute_user_query(self, sql_query, user_role, max_rows=None):
        """Execute user query with role-based restrictions"""
        try:
            # Apply role-based limitations
            limited_query = self.apply_role_restrictions(sql_query, user_role, max_rows)
            
            # Execute query using pandas for better data handling
            df = pd.read_sql_query(limited_query, self.engine)
            
            return True, df
            
        except Exception as e:
            return False, f"Query execution error: {str(e)}"
    
    def apply_role_restrictions(self, query, user_role, max_rows=None):
        """Apply role-based restrictions to SQL query"""
        # Get role limitations
        limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
        
        # Apply row limit
        if max_rows is None:
            max_rows = limits.get('max_rows', 1000)
        
        if max_rows and 'LIMIT' not in query.upper():
            query = f"{query} LIMIT {max_rows}"
        
        # Apply table restrictions (basic validation)
        allowed_tables = limits.get('allowed_tables', [])
        if '*' not in allowed_tables:
            # This is a basic check - in production, use a proper SQL parser
            query_upper = query.upper()
            for table in ['USERS', 'USER_SESSIONS', 'SENSITIVE_DATA']:
                if table in query_upper and table not in [t.upper() for t in allowed_tables]:
                    raise ValueError(f"Access to table {table} is not allowed for role {user_role}")
        
        return query
    
    def get_available_tables(self, user_role):
        """Get list of tables accessible to user role"""
        limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
        allowed_tables = limits.get('allowed_tables', [])
        
        if '*' in allowed_tables:
            # Get all ARGO-related tables
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'argo_%'
            ORDER BY table_name
            """
        else:
            # Get only allowed tables
            placeholders = ','.join(['%s'] * len(allowed_tables))
            query = f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ({placeholders})
            ORDER BY table_name
            """
        
        result = execute_query(query, allowed_tables if '*' not in allowed_tables else None)
        return [row['table_name'] for row in result] if result else []
    
    def get_table_schema(self, table_name, user_role):
        """Get table schema with role-based column filtering"""
        limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
        restricted_columns = limits.get('restricted_columns', [])
        
        query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = %s
        ORDER BY ordinal_position
        """
        
        result = execute_query(query, (table_name,))
        
        if result:
            # Filter out restricted columns
            filtered_columns = []
            for col in result:
                if col['column_name'] not in restricted_columns:
                    filtered_columns.append(col)
            return filtered_columns
        
        return []
    
    def get_recent_profiles(self, limit=10, user_role='Viewer'):
        """Get recent ARGO profiles"""
        query = """
        SELECT 
            p.id,
            m.platform_number,
            p.cycle_number,
            p.latitude,
            p.longitude,
            p.juld,
            p.n_levels,
            m.project_name
        FROM argo_profiles p
        JOIN argo_metadata m ON p.metadata_id = m.id
        ORDER BY p.juld DESC
        LIMIT %s
        """
        
        result = execute_query(query, (limit,))
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def search_profiles_by_location(self, lat_min, lat_max, lon_min, lon_max, user_role='Viewer'):
        """Search profiles by geographical bounds"""
        query = """
        SELECT 
            p.id,
            m.platform_number,
            p.cycle_number,
            p.latitude,
            p.longitude,
            p.juld,
            p.n_levels,
            m.project_name,
            ST_Distance(
                ST_Point(p.longitude, p.latitude)::geography,
                ST_Point(%s, %s)::geography
            ) / 1000 as distance_km
        FROM argo_profiles p
        JOIN argo_metadata m ON p.metadata_id = m.id
        WHERE p.latitude BETWEEN %s AND %s
        AND p.longitude BETWEEN %s AND %s
        ORDER BY distance_km
        """
        
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2
        
        # Apply role restrictions
        limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
        max_rows = limits.get('max_rows', 1000)
        
        if max_rows:
            query += f" LIMIT {max_rows}"
        
        result = execute_query(query, (center_lon, center_lat, lat_min, lat_max, lon_min, lon_max))
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def get_profile_measurements(self, profile_id, user_role='Viewer'):
        """Get measurements for a specific profile"""
        query = """
        SELECT 
            depth_level,
            pressure,
            temperature,
            salinity,
            doxy,
            chla,
            bbp700,
            ph_in_situ_total,
            nitrate
        FROM argo_measurements
        WHERE profile_id = %s
        ORDER BY depth_level
        """
        
        result = execute_query(query, (profile_id,))
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def get_float_trajectory(self, platform_number, user_role='Viewer'):
        """Get trajectory data for a specific float"""
        query = """
        SELECT 
            p.cycle_number,
            p.latitude,
            p.longitude,
            p.juld,
            p.n_levels
        FROM argo_profiles p
        JOIN argo_metadata m ON p.metadata_id = m.id
        WHERE m.platform_number = %s
        ORDER BY p.juld
        """
        
        result = execute_query(query, (platform_number,))
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def search_profiles_by_date(self, start_date, end_date, user_role='Viewer'):
        """Search profiles by date range"""
        query = """
        SELECT 
            p.id,
            m.platform_number,
            p.cycle_number,
            p.latitude,
            p.longitude,
            p.juld,
            p.n_levels,
            m.project_name
        FROM argo_profiles p
        JOIN argo_metadata m ON p.metadata_id = m.id
        WHERE p.juld BETWEEN %s AND %s
        ORDER BY p.juld DESC
        """
        
        # Apply role restrictions
        limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
        max_rows = limits.get('max_rows', 1000)
        
        if max_rows:
            query += f" LIMIT {max_rows}"
        
        result = execute_query(query, (start_date, end_date))
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def get_data_statistics(self, user_role='Viewer'):
        """Get data statistics for dashboard"""
        stats = {}
        
        # Basic counts
        count_queries = {
            'total_floats': "SELECT COUNT(DISTINCT platform_number) as count FROM argo_metadata",
            'total_profiles': "SELECT COUNT(*) as count FROM argo_profiles",
            'total_measurements': "SELECT COUNT(*) as count FROM argo_measurements"
        }
        
        for key, query in count_queries.items():
            result = execute_query(query)
            stats[key] = result[0]['count'] if result else 0
        
        # Recent activity
        recent_query = """
        SELECT DATE(juld) as date, COUNT(*) as profiles
        FROM argo_profiles
        WHERE juld >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(juld)
        ORDER BY date DESC
        LIMIT 30
        """
        
        result = execute_query(recent_query)
        stats['recent_activity'] = result if result else []
        
        # Geographical distribution
        geo_query = """
        SELECT 
            FLOOR(latitude/10)*10 as lat_bin,
            FLOOR(longitude/10)*10 as lon_bin,
            COUNT(*) as count
        FROM argo_profiles
        GROUP BY lat_bin, lon_bin
        ORDER BY count DESC
        LIMIT 100
        """
        
        result = execute_query(geo_query)
        stats['geo_distribution'] = result if result else []
        
        return stats
