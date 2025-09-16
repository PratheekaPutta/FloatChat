import os
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    """Get database URL from environment variables"""
    return os.getenv("DATABASE_URL") or f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT"),
            database=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def get_sqlalchemy_engine():
    """Get SQLAlchemy engine for pandas operations"""
    database_url = get_database_url()
    return create_engine(database_url)

def init_database():
    """Initialize database tables"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            # Read and execute SQL initialization script
            with open('sql/init_tables.sql', 'r') as f:
                sql_commands = f.read()
            
            cursor.execute(sql_commands)
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False

def execute_query(query, params=None, fetch=True):
    """Execute a database query safely"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
                
            conn.commit()
            cursor.close()
            conn.close()
            
            return result
    except Exception as e:
        st.error(f"Query execution failed: {e}")
        return None

def check_user_permissions(user_id, resource_type, operation):
    """Check if user has permission for specific operation"""
    query = """
    SELECT COUNT(*) as count FROM user_permissions up
    JOIN users u ON up.user_id = u.id
    WHERE up.user_id = %s AND up.resource_type = %s AND up.operation = %s
    """
    result = execute_query(query, (user_id, resource_type, operation))
    return result[0]['count'] > 0 if result else False
