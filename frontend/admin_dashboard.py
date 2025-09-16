import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from auth.user_management import (
    get_all_users, create_user, update_user_role, 
    toggle_user_status, delete_user, get_user_activity_stats
)
from data_processing.netcdf_processor import NetCDFProcessor
from vectorstore.faiss_manager import FAISSManager
from config.database import execute_query
from auth.authentication import check_permission
import tempfile
import os

def render_admin_dashboard(current_user):
    """Render the admin dashboard"""
    
    st.title("🔧 Admin Dashboard")
    st.markdown("System administration and user management")
    
    # Check admin permissions
    if current_user['role'] != 'Admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    # Dashboard tabs
    tabs = st.tabs([
        "User Management",
        "Data Management", 
        "System Statistics",
        "Vector Database",
        "System Health"
    ])
    
    with tabs[0]:
        render_user_management()
    
    with tabs[1]:
        render_data_management(current_user)
    
    with tabs[2]:
        render_system_statistics()
    
    with tabs[3]:
        render_vector_database_management()
    
    with tabs[4]:
        render_system_health()

def render_user_management():
    """Render user management interface"""
    
    st.header("👥 User Management")
    
    # Get user statistics
    stats = get_user_activity_stats()
    
    # Display user statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", stats.get('total_users', 0))
    with col2:
        st.metric("Active Users", stats.get('active_users', 0))
    with col3:
        st.metric("Recent Logins", stats.get('recent_logins', 0))
    with col4:
        inactive_users = stats.get('total_users', 0) - stats.get('active_users', 0)
        st.metric("Inactive Users", inactive_users)
    
    # Users by role chart
    if stats.get('users_by_role'):
        roles_df = pd.DataFrame(list(stats['users_by_role'].items()), columns=['Role', 'Count'])
        fig = px.pie(roles_df, values='Count', names='Role', title='Users by Role')
        st.plotly_chart(fig, use_container_width=True)
    
    # User management sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("User List")
        
        # Get all users
        users = get_all_users()
        
        if users:
            users_df = pd.DataFrame(users)
            users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d')
            users_df['last_login'] = pd.to_datetime(users_df['last_login'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            
            # Display users with action buttons
            for _, user in users_df.iterrows():
                with st.container():
                    user_col1, user_col2, user_col3, user_col4 = st.columns([3, 2, 2, 2])
                    
                    with user_col1:
                        status_icon = "🟢" if user['is_active'] else "🔴"
                        st.write(f"{status_icon} **{user['username']}** ({user['email']})")
                        st.write(f"Role: {user['role']} | Created: {user['created_at']}")
                    
                    with user_col2:
                        new_role = st.selectbox(
                            "Role",
                            ["Admin", "Researcher", "Viewer"],
                            index=["Admin", "Researcher", "Viewer"].index(user['role']),
                            key=f"role_{user['id']}"
                        )
                        
                        if new_role != user['role']:
                            if st.button("Update Role", key=f"update_role_{user['id']}"):
                                success, message = update_user_role(user['id'], new_role)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    
                    with user_col3:
                        status_text = "Deactivate" if user['is_active'] else "Activate"
                        if st.button(status_text, key=f"toggle_{user['id']}"):
                            success, message = toggle_user_status(user['id'])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with user_col4:
                        if user['role'] != 'Admin':  # Don't allow deleting admin users
                            if st.button("Delete", key=f"delete_{user['id']}", type="secondary"):
                                success, message = delete_user(user['id'])
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    
                    st.divider()
        else:
            st.info("No users found.")
    
    with col2:
        st.subheader("Create New User")
        
        with st.form("create_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["Researcher", "Viewer", "Admin"])
            
            submit_button = st.form_submit_button("Create User")
            
            if submit_button:
                if new_username and new_email and new_password:
                    success, message = create_user(new_username, new_email, new_password, new_role)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")

def render_data_management(current_user):
    """Render data management interface"""
    
    st.header("📊 Data Management")
    
    # Data upload section
    st.subheader("NetCDF File Upload")
    
    uploaded_file = st.file_uploader(
        "Upload ARGO NetCDF file",
        type=['nc', 'netcdf'],
        help="Upload ARGO float NetCDF files for processing"
    )
    
    if uploaded_file is not None:
        if st.button("Process NetCDF File"):
            with st.spinner("Processing NetCDF file..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the file
                    processor = NetCDFProcessor()
                    success, message = processor.process_argo_file(tmp_file_path, current_user['id'])
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    if success:
                        st.success(message)
                        # Rebuild vector index after new data
                        faiss_manager = FAISSManager()
                        faiss_manager.rebuild_index()
                        st.info("Vector index updated with new data")
                    else:
                        st.error(message)
                        
                except Exception as e:
                    st.error(f"Error processing file: {e}")
    
    # Data summary
    st.subheader("Data Summary")
    
    try:
        processor = NetCDFProcessor()
        summary = processor.get_data_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Floats", summary.get('total_floats', 0))
        with col2:
            st.metric("Total Profiles", summary.get('total_profiles', 0))
        with col3:
            st.metric("Total Measurements", summary.get('total_measurements', 0))
        with col4:
            if summary.get('date_range'):
                date_range = summary['date_range']
                if date_range.get('min_date') and date_range.get('max_date'):
                    duration = (date_range['max_date'] - date_range['min_date']).days
                    st.metric("Date Range", f"{duration} days")
        
        # Spatial coverage
        if summary.get('spatial_coverage'):
            spatial = summary['spatial_coverage']
            if all(k in spatial for k in ['min_lat', 'max_lat', 'min_lon', 'max_lon']):
                st.write("**Spatial Coverage:**")
                st.write(f"Latitude: {spatial['min_lat']:.2f}°N to {spatial['max_lat']:.2f}°N")
                st.write(f"Longitude: {spatial['min_lon']:.2f}°E to {spatial['max_lon']:.2f}°E")
        
    except Exception as e:
        st.error(f"Error loading data summary: {e}")
    
    # Recent uploads
    st.subheader("Recent Data Uploads")
    
    recent_uploads_query = """
    SELECT m.platform_number, m.project_name, m.n_profiles, m.uploaded_at, u.username
    FROM argo_metadata m
    JOIN users u ON m.uploaded_by = u.id
    ORDER BY m.uploaded_at DESC
    LIMIT 10
    """
    
    try:
        recent_uploads = execute_query(recent_uploads_query)
        if recent_uploads:
            uploads_df = pd.DataFrame(recent_uploads)
            uploads_df['uploaded_at'] = pd.to_datetime(uploads_df['uploaded_at']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(uploads_df, use_container_width=True)
        else:
            st.info("No recent uploads found.")
    except Exception as e:
        st.error(f"Error loading recent uploads: {e}")

def render_system_statistics():
    """Render system statistics and analytics"""
    
    st.header("📈 System Statistics")
    
    # Database statistics
    try:
        # Get comprehensive database statistics
        stats_queries = {
            'profiles_by_day': """
                SELECT DATE(juld) as date, COUNT(*) as count
                FROM argo_profiles
                WHERE juld >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(juld)
                ORDER BY date DESC
            """,
            'floats_by_project': """
                SELECT project_name, COUNT(*) as count
                FROM argo_metadata
                GROUP BY project_name
                ORDER BY count DESC
                LIMIT 10
            """,
            'measurements_by_parameter': """
                SELECT 
                    SUM(CASE WHEN temperature IS NOT NULL THEN 1 ELSE 0 END) as temperature,
                    SUM(CASE WHEN salinity IS NOT NULL THEN 1 ELSE 0 END) as salinity,
                    SUM(CASE WHEN doxy IS NOT NULL THEN 1 ELSE 0 END) as oxygen,
                    SUM(CASE WHEN chla IS NOT NULL THEN 1 ELSE 0 END) as chlorophyll,
                    SUM(CASE WHEN nitrate IS NOT NULL THEN 1 ELSE 0 END) as nitrate
                FROM argo_measurements
            """,
            'data_quality_distribution': """
                SELECT data_mode, COUNT(*) as count
                FROM argo_profiles
                WHERE data_mode IS NOT NULL
                GROUP BY data_mode
            """
        }
        
        # Execute queries and create visualizations
        for query_name, query in stats_queries.items():
            try:
                result = execute_query(query)
                if result and len(result) > 0:
                    df = pd.DataFrame(result)
                    
                    if query_name == 'profiles_by_day':
                        st.subheader("Daily Profile Activity (Last 30 Days)")
                        df['date'] = pd.to_datetime(df['date'])
                        fig = px.line(df, x='date', y='count', title='Profiles per Day')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif query_name == 'floats_by_project':
                        st.subheader("Floats by Project")
                        fig = px.bar(df, x='project_name', y='count', title='Number of Floats by Project')
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif query_name == 'measurements_by_parameter':
                        st.subheader("Measurements by Parameter Type")
                        # Reshape data for plotting
                        param_data = []
                        for col in df.columns:
                            param_data.append({'Parameter': col.title(), 'Count': df[col].iloc[0]})
                        param_df = pd.DataFrame(param_data)
                        
                        fig = px.bar(param_df, x='Parameter', y='Count', title='Available Measurements by Parameter')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif query_name == 'data_quality_distribution':
                        st.subheader("Data Quality Distribution")
                        fig = px.pie(df, values='count', names='data_mode', title='Data Mode Distribution')
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error executing {query_name}: {e}")
    
    except Exception as e:
        st.error(f"Error loading system statistics: {e}")
    
    # User activity statistics
    st.subheader("User Activity")
    
    user_activity_query = """
    SELECT 
        u.role,
        COUNT(CASE WHEN u.last_login >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as last_7_days,
        COUNT(CASE WHEN u.last_login >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as last_30_days,
        COUNT(*) as total_users
    FROM users u
    WHERE u.is_active = true
    GROUP BY u.role
    """
    
    try:
        user_activity = execute_query(user_activity_query)
        if user_activity:
            activity_df = pd.DataFrame(user_activity)
            
            fig = go.Figure(data=[
                go.Bar(name='Last 7 days', x=activity_df['role'], y=activity_df['last_7_days']),
                go.Bar(name='Last 30 days', x=activity_df['role'], y=activity_df['last_30_days']),
                go.Bar(name='Total users', x=activity_df['role'], y=activity_df['total_users'])
            ])
            
            fig.update_layout(
                title='User Activity by Role',
                xaxis_title='Role',
                yaxis_title='Number of Users',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading user activity: {e}")

def render_vector_database_management():
    """Render vector database management interface"""
    
    st.header("🔍 Vector Database Management")
    
    try:
        faiss_manager = FAISSManager()
        
        # Get index statistics
        stats = faiss_manager.get_index_stats()
        
        # Display index statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Vectors", stats['total_vectors'])
        with col2:
            st.metric("Vector Dimension", stats['dimension'])
        with col3:
            st.metric("Metadata Entries", stats['metadata_count'])
        with col4:
            st.metric("Index Size", f"{stats['index_size_mb']} MB")
        
        # Index management controls
        st.subheader("Index Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Rebuild Index"):
                with st.spinner("Rebuilding FAISS index..."):
                    success = faiss_manager.rebuild_index()
                    if success:
                        st.success("Index rebuilt successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to rebuild index")
        
        with col2:
            if st.button("Save Index"):
                with st.spinner("Saving index..."):
                    success = faiss_manager.save_index()
                    if success:
                        st.success("Index saved successfully!")
                    else:
                        st.error("Failed to save index")
        
        with col3:
            if st.button("Load Index"):
                with st.spinner("Loading index..."):
                    success = faiss_manager.load_index()
                    if success:
                        st.success("Index loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load index")
        
        # Test search functionality
        st.subheader("Test Vector Search")
        
        test_query = st.text_input("Enter test query:", "temperature profiles near equator")
        
        if st.button("Test Search") and test_query:
            with st.spinner("Searching..."):
                try:
                    results = faiss_manager.query_similar_data(test_query, k=5)
                    
                    if results:
                        st.success(f"Found {len(results)} relevant results")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1} (Similarity: {result['similarity']:.3f})"):
                                st.write(f"**Summary:** {result['metadata']['summary']}")
                                st.write(f"**Type:** {result['metadata']['type']}")
                                if 'platform_number' in result['metadata']:
                                    st.write(f"**Platform:** {result['metadata']['platform_number']}")
                    else:
                        st.warning("No relevant results found")
                        
                except Exception as e:
                    st.error(f"Search error: {e}")
        
        # MCP Status
        st.subheader("MCP Client Status")
        
        try:
            from rag.mcp_client import MCPClient
            mcp_client = MCPClient()
            mcp_status = mcp_client.get_mcp_status()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Protocol Version:** {mcp_status['protocol_version']}")
                st.write(f"**Model:** {mcp_status['model']}")
                st.write(f"**Status:** {mcp_status['status']}")
            
            with col2:
                st.write("**Available Tools:**")
                for tool in mcp_status['available_tools']:
                    st.write(f"• {tool}")
        
        except Exception as e:
            st.error(f"Error getting MCP status: {e}")
    
    except Exception as e:
        st.error(f"Error loading vector database management: {e}")

def render_system_health():
    """Render system health monitoring"""
    
    st.header("🏥 System Health")
    
    # Database connectivity test
    st.subheader("Database Health")
    
    try:
        from config.database import get_db_connection
        
        conn = get_db_connection()
        if conn:
            st.success("✅ Database connection successful")
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM users")
            result = cursor.fetchone()
            st.write(f"Database query test: {result['count']} users found")
            
            cursor.close()
            conn.close()
        else:
            st.error("❌ Database connection failed")
    
    except Exception as e:
        st.error(f"❌ Database error: {e}")
    
    # API connectivity tests
    st.subheader("External APIs")
    
    # Test OpenAI API
    try:
        from openai import OpenAI
        from config.settings import settings
        
        if settings.OPENAI_API_KEY:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a simpler model for testing
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            st.success("✅ OpenAI API connection successful")
        else:
            st.warning("⚠️ OpenAI API key not configured")
    
    except Exception as e:
        st.error(f"❌ OpenAI API error: {e}")
    
    # System resources (basic info)
    st.subheader("System Resources")
    
    try:
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        st.write(f"**Memory Usage:** {memory.percent}% ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        st.write(f"**CPU Usage:** {cpu_percent}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        st.write(f"**Disk Usage:** {disk.percent}% ({disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB)")
        
    except ImportError:
        st.info("Install psutil for detailed system resource monitoring")
    except Exception as e:
        st.error(f"Error getting system resources: {e}")
    
    # Application logs (if available)
    st.subheader("Recent System Events")
    
    # This would typically read from application logs
    # For now, we'll show recent database activity
    try:
        recent_activity_query = """
        SELECT 
            'User Login' as event_type,
            u.username as details,
            u.last_login as timestamp
        FROM users u
        WHERE u.last_login IS NOT NULL
        AND u.last_login >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        
        UNION ALL
        
        SELECT 
            'Data Upload' as event_type,
            CONCAT('Platform ', m.platform_number, ' - ', m.n_profiles, ' profiles') as details,
            m.uploaded_at as timestamp
        FROM argo_metadata m
        WHERE m.uploaded_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        
        ORDER BY timestamp DESC
        LIMIT 20
        """
        
        events = execute_query(recent_activity_query)
        if events:
            events_df = pd.DataFrame(events)
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No recent system events found")
    
    except Exception as e:
        st.error(f"Error loading system events: {e}")
