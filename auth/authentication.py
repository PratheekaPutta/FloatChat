import streamlit as st
import bcrypt
import jwt
from datetime import datetime, timedelta
import pyotp
from config.database import execute_query
from config.settings import settings
from auth.security_enhancements import check_geoip_alert
import requests
from security_enhancements import (
    record_login_attempt,
    check_login_rate,
    check_geoip_alert,
    send_email_alert
)
# ------------------------------
# Password & JWT helpers
# ------------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(user_data):
    payload = {
        'user_id': user_data['id'],
        'username': user_data['username'],
        'role': user_data['role'],
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm='HS256')

def verify_jwt_token(token):
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# ------------------------------
# TOTP 2FA helpers
# ------------------------------
def generate_totp_secret():
    return pyotp.random_base32()

def verify_totp(secret, token):
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

# ------------------------------
# Audit logging
# ------------------------------
def log_audit(user_id, action, ip_address=None):
    query = """
    INSERT INTO audit_logs (user_id, action, ip_address, timestamp)
    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
    """
    execute_query(query, (user_id, action, ip_address), fetch=False)

# ------------------------------
# Authentication
# ------------------------------
# login.py
import streamlit as st
import requests
from auth.authentication import authenticate_user

# ------------------------------
# Helper to get public IP (Option 3)
# ------------------------------
def get_user_ip():
    try:
        return requests.get("https://api.ipify.org").text
    except:
        return "unknown"

# ------------------------------
# Streamlit Login Form
# ------------------------------
st.title("ARGO App Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")
totp_code = st.text_input("2FA Code (if enabled)")

if st.button("Login"):
    user_ip = get_user_ip()  # Fetch public IP
    user = authenticate_user(username, password, totp_token=totp_code, ip_address=user_ip)
    
    if user:
        st.success(f"Welcome {user['username']}!")
        st.write(f"Your IP: {user_ip}")
    else:
        st.error("Login failed! Check username, password, or 2FA code.")


def get_current_user():
    if 'auth_token' not in st.session_state:
        return None
    token_data = verify_jwt_token(st.session_state.auth_token)
    if not token_data:
        del st.session_state.auth_token
        return None
    query = "SELECT id, username, role, email, is_active FROM users WHERE id = %s"
    result = execute_query(query, (token_data['user_id'],))
    if result and result[0]['is_active']:
        user = result[0]
        st.session_state.current_user = {
            'id': user['id'],
            'username': user['username'],
            'role': user['role'],
            'email': user['email']
        }
        return st.session_state.current_user
    return None

def logout_user():
    for key in list(st.session_state.keys()):
        if key.startswith('auth_') or key.startswith('current_') or key.startswith('chat_') or key.startswith('query_'):
            del st.session_state[key]

def require_role(required_roles):
    def decorator(func):
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                st.error("Authentication required")
                return None
            if isinstance(required_roles, str):
                roles = [required_roles]
            else:
                roles = required_roles
            if user['role'] not in roles:
                st.error("Insufficient permissions")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_permission(user, permission_type):
    permissions = settings.ROLE_PERMISSIONS.get(user['role'], {})
    return permissions.get(permission_type, False)
