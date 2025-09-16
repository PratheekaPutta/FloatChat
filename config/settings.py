import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL")
    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = os.getenv("PGPORT", "5432")
    PGDATABASE = os.getenv("PGDATABASE", "argo_db")
    PGUSER = os.getenv("PGUSER", "postgres")
    PGPASSWORD = os.getenv("PGPASSWORD", "")
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # Authentication
    SESSION_SECRET = os.getenv("SESSION_SECRET", "default_secret_key")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_jwt_secret")
    
    # Application settings
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3-8b-8192")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Role-based access control
    ROLE_PERMISSIONS = {
        'Admin': {
            'data_access': 'full',
            'user_management': True,
            'file_upload': True,
            'complex_queries': True,
            'export_data': True
        },
        'Researcher': {
            'data_access': 'research',
            'user_management': False,
            'file_upload': False,
            'complex_queries': True,
            'export_data': True
        },
        'Viewer': {
            'data_access': 'basic',
            'user_management': False,
            'file_upload': False,
            'complex_queries': False,
            'export_data': False
        }
    }
    
    # Query limitations by role
    QUERY_LIMITS = {
        'Admin': {
            'max_rows': None,
            'allowed_tables': ['*'],
            'restricted_columns': []
        },
        'Researcher': {
            'max_rows': 50000,
            'allowed_tables': ['argo_profiles', 'argo_metadata', 'argo_bgc'],
            'restricted_columns': ['user_id', 'internal_notes']
        },
        'Viewer': {
            'max_rows': 1000,
            'allowed_tables': ['argo_profiles', 'argo_metadata'],
            'restricted_columns': ['user_id', 'internal_notes', 'raw_data']
        }
    }

# Create global settings instance
settings = Settings()
