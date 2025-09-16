import streamlit as st
from config.database import execute_query
from auth.authentication import hash_password
from config.settings import settings
import re

def create_admin_user():
    """Create default admin user if it doesn't exist"""
    # Check if admin user already exists
    query = "SELECT COUNT(*) as count FROM users WHERE username = 'admin'"
    result = execute_query(query)
    
    if result and result[0]['count'] == 0:
        # Create admin user
        hashed_password = hash_password(settings.ADMIN_PASSWORD)
        
        insert_query = """
        INSERT INTO users (username, email, password_hash, role, is_active, created_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """
        
        execute_query(insert_query, (
            'admin',
            settings.ADMIN_EMAIL,
            hashed_password,
            'Admin',
            True
        ), fetch=False)
        
        # Create default demo users
        create_demo_users()

def create_demo_users():
    """Create demo users for testing"""
    demo_users = [
        {
            'username': 'researcher',
            'email': 'researcher@example.com',
            'password': 'research123',
            'role': 'Researcher'
        },
        {
            'username': 'viewer',
            'email': 'viewer@example.com',
            'password': 'view123',
            'role': 'Viewer'
        }
    ]
    
    for user_data in demo_users:
        # Check if user exists
        query = "SELECT COUNT(*) as count FROM users WHERE username = %s"
        result = execute_query(query, (user_data['username'],))
        
        if result and result[0]['count'] == 0:
            hashed_password = hash_password(user_data['password'])
            
            insert_query = """
            INSERT INTO users (username, email, password_hash, role, is_active, created_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            
            execute_query(insert_query, (
                user_data['username'],
                user_data['email'],
                hashed_password,
                user_data['role'],
                True
            ), fetch=False)

def get_all_users():
    """Get all users from database"""
    query = """
    SELECT id, username, email, role, is_active, created_at, last_login
    FROM users
    ORDER BY created_at DESC
    """
    return execute_query(query)

def create_user(username, email, password, role):
    """Create a new user"""
    # Validate input
    if not username or not email or not password or not role:
        return False, "All fields are required"
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    
    # Check if username already exists
    query = "SELECT COUNT(*) as count FROM users WHERE username = %s"
    result = execute_query(query, (username,))
    
    if result and result[0]['count'] > 0:
        return False, "Username already exists"
    
    # Check if email already exists
    query = "SELECT COUNT(*) as count FROM users WHERE email = %s"
    result = execute_query(query, (email,))
    
    if result and result[0]['count'] > 0:
        return False, "Email already exists"
    
    # Validate role
    valid_roles = ['Admin', 'Researcher', 'Viewer']
    if role not in valid_roles:
        return False, "Invalid role"
    
    # Create user
    hashed_password = hash_password(password)
    
    insert_query = """
    INSERT INTO users (username, email, password_hash, role, is_active, created_at)
    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """
    
    result = execute_query(insert_query, (
        username,
        email,
        hashed_password,
        role,
        True
    ), fetch=False)
    
    if result is not None:
        return True, "User created successfully"
    else:
        return False, "Failed to create user"

def update_user_role(user_id, new_role):
    """Update user role"""
    valid_roles = ['Admin', 'Researcher', 'Viewer']
    if new_role not in valid_roles:
        return False, "Invalid role"
    
    query = "UPDATE users SET role = %s WHERE id = %s"
    result = execute_query(query, (new_role, user_id), fetch=False)
    
    if result is not None:
        return True, "User role updated successfully"
    else:
        return False, "Failed to update user role"

def toggle_user_status(user_id):
    """Toggle user active status"""
    # Get current status
    query = "SELECT is_active FROM users WHERE id = %s"
    result = execute_query(query, (user_id,))
    
    if result and len(result) > 0:
        current_status = result[0]['is_active']
        new_status = not current_status
        
        update_query = "UPDATE users SET is_active = %s WHERE id = %s"
        update_result = execute_query(update_query, (new_status, user_id), fetch=False)
        
        if update_result is not None:
            status_text = "activated" if new_status else "deactivated"
            return True, f"User {status_text} successfully"
        else:
            return False, "Failed to update user status"
    else:
        return False, "User not found"

def delete_user(user_id):
    """Delete a user (soft delete by deactivating)"""
    # Don't allow deletion of admin user
    query = "SELECT role FROM users WHERE id = %s"
    result = execute_query(query, (user_id,))
    
    if result and len(result) > 0:
        if result[0]['role'] == 'Admin':
            return False, "Cannot delete admin user"
        
        # Deactivate instead of hard delete
        update_query = "UPDATE users SET is_active = false WHERE id = %s"
        update_result = execute_query(update_query, (user_id,), fetch=False)
        
        if update_result is not None:
            return True, "User deactivated successfully"
        else:
            return False, "Failed to deactivate user"
    else:
        return False, "User not found"

def get_user_activity_stats():
    """Get user activity statistics"""
    queries = {
        'total_users': "SELECT COUNT(*) as count FROM users",
        'active_users': "SELECT COUNT(*) as count FROM users WHERE is_active = true",
        'users_by_role': """
            SELECT role, COUNT(*) as count 
            FROM users 
            WHERE is_active = true 
            GROUP BY role
        """,
        'recent_logins': """
            SELECT COUNT(*) as count 
            FROM users 
            WHERE last_login >= CURRENT_DATE - INTERVAL '7 days'
        """
    }
    
    stats = {}
    for key, query in queries.items():
        result = execute_query(query)
        if result:
            if key == 'users_by_role':
                stats[key] = {row['role']: row['count'] for row in result}
            else:
                stats[key] = result[0]['count']
        else:
            stats[key] = 0
    
    return stats
