import streamlit as st
import bcrypt
import jwt
from datetime import datetime, timedelta
import os
from config.database import execute_query
from config.settings import settings

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(user_data):
    """Create a JWT token for user session"""
    payload = {
        'user_id': user_data['id'],
        'username': user_data['username'],
        'role': user_data['role'],
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm='HS256')
    return token

def verify_jwt_token(token):
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def authenticate_user(username, password):
    """Authenticate user with username and password"""
    query = "SELECT id, username, password_hash, role, email, is_active FROM users WHERE username = %s"
    result = execute_query(query, (username,))   # fixed tuple
    
    if result and len(result) > 0:
        user = result[0]
        
        if not user['is_active']:
            st.error("Account is deactivated. Please contact administrator.")
            return None
        
        if verify_password(password, user['password_hash']):
            # Create JWT token
            token = create_jwt_token(user)
            
            # Store in session state
            st.session_state.auth_token = token
            st.session_state.current_user = {
                'id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'email': user['email']
            }
            
            # Update last login
            update_query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s"
            execute_query(update_query, (user['id'],), fetch=False)   # fixed tuple
            
            return st.session_state.current_user
    
    return None

def get_current_user():
    """Get currently authenticated user from session"""
    if 'auth_token' in st.session_state:
        token_data = verify_jwt_token(st.session_state.auth_token)
        if token_data:
            # Refresh user data from database
            query = "SELECT id, username, role, email, is_active FROM users WHERE id = %s"
            result = execute_query(query, (token_data['user_id'],))   # fixed tuple
            
            if result and len(result) > 0:
                user = result[0]
                if user['is_active']:
                    st.session_state.current_user = {
                        'id': user['id'],
                        'username': user['username'],
                        'role': user['role'],
                        'email': user['email']
                    }
                    return st.session_state.current_user
    
    # Clear invalid session
    if 'auth_token' in st.session_state:
        del st.session_state.auth_token
    if 'current_user' in st.session_state:
        del st.session_state.current_user
    
    return None

def logout_user():
    """Logout current user"""
    if 'auth_token' in st.session_state:
        del st.session_state.auth_token
    if 'current_user' in st.session_state:
        del st.session_state.current_user
    
    # Clear other session data
    for key in list(st.session_state.keys()):
        if key.startswith('chat_') or key.startswith('query_'):
            del st.session_state[key]

def require_role(required_roles):
    """Decorator to require specific roles for access"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_user = get_current_user()
            if not current_user:
                st.error("Authentication required")
                return None
            
            if isinstance(required_roles, str):
                roles = [required_roles]
            else:
                roles = required_roles
            
            if current_user['role'] not in roles:
                st.error("Insufficient permissions")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_permission(user, permission_type):
    """Check if user has specific permission"""
    user_role = user['role']
    permissions = settings.ROLE_PERMISSIONS.get(user_role, {})
    return permissions.get(permission_type, False)
