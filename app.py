import streamlit as st
import os
from dotenv import load_dotenv
from auth.authentication import authenticate_user, logout_user, get_current_user
from frontend.chat_interface import render_chat_interface
from frontend.admin_dashboard import render_admin_dashboard
from frontend.visualizations import render_visualization_dashboard
from config.database import init_database
from auth.user_management import create_admin_user

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ARGO AI Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Initialize database and create admin user if needed
    if 'db_initialized' not in st.session_state:
        init_database()
        create_admin_user()
        st.session_state.db_initialized = True
    
    # Check authentication status
    current_user = get_current_user()
    
    if not current_user:
        # Show login page
        render_login_page()
    else:
        # Show main application
        render_main_app(current_user)

def render_login_page():
    """Render the login interface"""
    st.title("🌊 ARGO AI Assistant")
    st.markdown("### AI-Powered Oceanographic Data Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("#### Login to Continue")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    user = authenticate_user(username, password)
                    if user:
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        st.markdown("---")
        st.markdown("**Demo Accounts:**")
        st.markdown("- Admin: `admin` / `admin123`")
        st.markdown("- Researcher: `researcher` / `research123`")
        st.markdown("- Viewer: `viewer` / `view123`")

def render_main_app(current_user):
    """Render the main application interface"""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🌊 ARGO AI")
        st.markdown(f"**Welcome, {current_user['username']}**")
        st.markdown(f"*Role: {current_user['role']}*")
        
        st.markdown("---")
        
        # Navigation menu
        if current_user['role'] == 'Admin':
            pages = ["Chat Interface", "Visualizations", "Admin Dashboard"]
        elif current_user['role'] == 'Researcher':
            pages = ["Chat Interface", "Visualizations"]
        else:  # Viewer
            pages = ["Chat Interface", "Basic Visualizations"]
        
        selected_page = st.selectbox("Navigate to:", pages)
        
        st.markdown("---")
        
        if st.button("Logout"):
            logout_user()
            st.rerun()
    
    # Main content area
    if selected_page == "Chat Interface":
        render_chat_interface(current_user)
    elif selected_page == "Visualizations" or selected_page == "Basic Visualizations":
        render_visualization_dashboard(current_user)
    elif selected_page == "Admin Dashboard" and current_user['role'] == 'Admin':
        render_admin_dashboard(current_user)

if __name__ == "__main__":
    main()
