import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import folium
# from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_processing.database_manager import DatabaseManager
from auth.authentication import check_permission

# Temporary fallback for map functionality
FOLIUM_AVAILABLE = False

def render_visualization_dashboard(current_user):
    """Render the main visualization dashboard"""
    
    st.title("📊 ARGO Data Visualizations")
    st.markdown("Interactive visualizations of oceanographic data")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Sidebar for visualization controls
    with st.sidebar:
        st.markdown("### Visualization Options")
        
        viz_type = st.selectbox(
            "Choose Visualization Type:",
            [
                "Geographic Distribution",
                "Temperature Profiles", 
                "Salinity Analysis",
                "Time Series",
                "Data Statistics",
                "Float Trajectories"
            ]
        )
        
        # Date range selector
        st.markdown("#### Date Range")
        
        # Get date range from database
        date_range_query = """
        SELECT MIN(juld) as min_date, MAX(juld) as max_date 
        FROM argo_profiles 
        WHERE juld IS NOT NULL
        """
        date_result = db_manager.execute_query(date_range_query)
        
        if date_result and len(date_result) > 0:
            min_date = date_result[0].get('min_date')
            max_date = date_result[0].get('max_date')
            
            if min_date and max_date:
                start_date = st.date_input(
                    "Start Date",
                    value=max_date - timedelta(days=365),  # Last year by default
                    min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                    max_value=max_date.date() if hasattr(max_date, 'date') else max_date
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=max_date.date() if hasattr(max_date, 'date') else max_date,
                    min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                    max_value=max_date.date() if hasattr(max_date, 'date') else max_date
                )
            else:
                start_date = datetime.now().date() - timedelta(days=365)
                end_date = datetime.now().date()
        else:
            start_date = datetime.now().date() - timedelta(days=365)
            end_date = datetime.now().date()
        
        # Geographic bounds
        st.markdown("#### Geographic Bounds")
        col1, col2 = st.columns(2)
        with col1:
            lat_min = st.number_input("Min Latitude", value=-90.0, min_value=-90.0, max_value=90.0)
            lon_min = st.number_input("Min Longitude", value=-180.0, min_value=-180.0, max_value=180.0)
        with col2:
            lat_max = st.number_input("Max Latitude", value=90.0, min_value=-90.0, max_value=90.0)
            lon_max = st.number_input("Max Longitude", value=180.0, min_value=-180.0, max_value=180.0)
    
    # Main visualization area
    if viz_type == "Geographic Distribution":
        render_geographic_visualization(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max)
    elif viz_type == "Temperature Profiles":
        render_temperature_profiles(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max)
    elif viz_type == "Salinity Analysis":
        render_salinity_analysis(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max)
    elif viz_type == "Time Series":
        render_time_series(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max)
    elif viz_type == "Data Statistics":
        render_data_statistics(db_manager, current_user)
    elif viz_type == "Float Trajectories":
        render_float_trajectories(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max)

def render_geographic_visualization(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """Render geographic distribution of ARGO floats"""
    
    st.header("🗺️ Geographic Distribution of ARGO Floats")
    
    # Query for float locations
    query = """
    SELECT 
        p.latitude, p.longitude, p.juld,
        m.platform_number, m.project_name,
        p.n_levels, p.cycle_number
    FROM argo_profiles p
    JOIN argo_metadata m ON p.metadata_id = m.id
    WHERE p.latitude BETWEEN %s AND %s
    AND p.longitude BETWEEN %s AND %s
    AND p.juld BETWEEN %s AND %s
    AND p.latitude IS NOT NULL
    AND p.longitude IS NOT NULL
    ORDER BY p.juld DESC
    """
    
    with st.spinner("Loading geographic data..."):
        try:
            data = db_manager.execute_query(query, (lat_min, lat_max, lon_min, lon_max, start_date, end_date))
            df = pd.DataFrame(data) if data else pd.DataFrame()
            
            if df.empty:
                st.warning("No data found for the selected criteria")
                return
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Profiles", len(df))
            with col2:
                st.metric("Unique Floats", df['platform_number'].nunique())
            with col3:
                st.metric("Date Range", f"{df['juld'].min().strftime('%Y-%m-%d')} to {df['juld'].max().strftime('%Y-%m-%d')}")
            with col4:
                st.metric("Avg Depth Levels", f"{df['n_levels'].mean():.0f}")
            
            # Create map
            st.subheader("Interactive Map")
            
            # Sample data if too large for performance
            if len(df) > 1000:
                st.info(f"Showing 1000 random samples from {len(df)} total profiles")
                df_map = df.sample(1000)
            else:
                df_map = df
            
            # Create Folium map
            center_lat = df_map['latitude'].mean()
            center_lon = df_map['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Add markers for each profile
            for _, row in df_map.iterrows():
                popup_text = f"""
                Platform: {row['platform_number']}<br>
                Cycle: {row['cycle_number']}<br>
                Date: {row['juld'].strftime('%Y-%m-%d')}<br>
                Location: {row['latitude']:.2f}°N, {row['longitude']:.2f}°E<br>
                Levels: {row['n_levels']}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=popup_text,
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.7
                ).add_to(m)
            
            # Display map
            map_data = st_folium(m, width=700, height=500)
            
            # Plotly scatter plot
            st.subheader("Density Plot")
            
            fig = px.density_mapbox(
                df_map, 
                lat='latitude', 
                lon='longitude',
                z='n_levels',
                hover_data=['platform_number', 'cycle_number', 'juld'],
                mapbox_style='open-street-map',
                title='ARGO Float Density by Depth Levels',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=3,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data export
            if check_permission(current_user, 'export_data'):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Geographic Data",
                    data=csv,
                    file_name=f"argo_geographic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error loading geographic data: {e}")

def render_temperature_profiles(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """Render temperature profile visualizations"""
    
    st.header("🌡️ Temperature Profiles")
    
    # Query for temperature data
    query = """
    SELECT 
        p.id as profile_id, p.latitude, p.longitude, p.juld,
        m.platform_number, m.project_name,
        am.pressure, am.temperature, am.depth_level
    FROM argo_profiles p
    JOIN argo_metadata m ON p.metadata_id = m.id
    JOIN argo_measurements am ON p.id = am.profile_id
    WHERE p.latitude BETWEEN %s AND %s
    AND p.longitude BETWEEN %s AND %s
    AND p.juld BETWEEN %s AND %s
    AND am.temperature IS NOT NULL
    AND am.pressure IS NOT NULL
    ORDER BY p.juld DESC, am.pressure
    LIMIT 10000
    """
    
    with st.spinner("Loading temperature profile data..."):
        try:
            data = db_manager.execute_query(query, (lat_min, lat_max, lon_min, lon_max, start_date, end_date))
            df = pd.DataFrame(data) if data else pd.DataFrame()
            
            if df.empty:
                st.warning("No temperature profile data found for the selected criteria")
                return
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Profiles", df['profile_id'].nunique())
            with col2:
                st.metric("Temperature Range", f"{df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
            with col3:
                st.metric("Depth Range", f"{df['pressure'].min():.0f} to {df['pressure'].max():.0f} dbar")
            with col4:
                st.metric("Measurements", len(df))
            
            # Profile selection
            unique_profiles = df[['profile_id', 'platform_number', 'latitude', 'longitude', 'juld']].drop_duplicates()
            
            st.subheader("Select Profiles to Display")
            
            if len(unique_profiles) > 10:
                st.info(f"Showing first 10 profiles from {len(unique_profiles)} available")
                selected_profiles = unique_profiles.head(10)['profile_id'].tolist()
            else:
                selected_profiles = unique_profiles['profile_id'].tolist()
            
            # Create temperature profile plot
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set3
            
            for i, profile_id in enumerate(selected_profiles[:10]):  # Limit to 10 profiles
                profile_data = df[df['profile_id'] == profile_id]
                
                if not profile_data.empty:
                    profile_info = unique_profiles[unique_profiles['profile_id'] == profile_id].iloc[0]
                    
                    fig.add_trace(go.Scatter(
                        x=profile_data['temperature'],
                        y=-profile_data['pressure'],  # Negative for depth
                        mode='lines+markers',
                        name=f"Float {profile_info['platform_number']} ({profile_info['juld'].strftime('%Y-%m-%d')})",
                        line=dict(color=colors[i % len(colors)]),
                        hovertemplate=
                        f"Float: {profile_info['platform_number']}<br>" +
                        "Temperature: %{x:.2f}°C<br>" +
                        "Pressure: %{y:.0f} dbar<br>" +
                        f"Date: {profile_info['juld'].strftime('%Y-%m-%d')}<br>" +
                        f"Location: {profile_info['latitude']:.2f}°N, {profile_info['longitude']:.2f}°E<br>" +
                        "<extra></extra>"
                    ))
            
            fig.update_layout(
                title="Temperature Profiles vs Pressure",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (dbar, negative for depth)",
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Average temperature profile
            st.subheader("Average Temperature Profile")
            
            # Bin pressure data and calculate mean temperature
            df['pressure_bin'] = pd.cut(df['pressure'], bins=50, labels=False)
            temp_profile = df.groupby('pressure_bin').agg({
                'pressure': 'mean',
                'temperature': ['mean', 'std', 'count']
            }).reset_index()
            
            temp_profile.columns = ['pressure_bin', 'pressure', 'temp_mean', 'temp_std', 'count']
            temp_profile = temp_profile[temp_profile['count'] >= 5]  # At least 5 measurements per bin
            
            fig_avg = go.Figure()
            
            # Add mean temperature line
            fig_avg.add_trace(go.Scatter(
                x=temp_profile['temp_mean'],
                y=-temp_profile['pressure'],
                mode='lines+markers',
                name='Average Temperature',
                line=dict(color='red', width=3),
                error_x=dict(array=temp_profile['temp_std'], visible=True),
                hovertemplate="Avg Temperature: %{x:.2f}±%{error_x:.2f}°C<br>Pressure: %{y:.0f} dbar<extra></extra>"
            ))
            
            fig_avg.update_layout(
                title="Average Temperature Profile with Standard Deviation",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (dbar, negative for depth)",
                height=500
            )
            
            st.plotly_chart(fig_avg, use_container_width=True)
            
            # Temperature statistics by depth
            st.subheader("Temperature Statistics by Depth Range")
            
            depth_ranges = [
                (0, 100, "Surface (0-100 dbar)"),
                (100, 500, "Intermediate (100-500 dbar)"),
                (500, 1000, "Deep (500-1000 dbar)"),
                (1000, 2000, "Deeper (1000-2000 dbar)"),
                (2000, float('inf'), "Abyssal (>2000 dbar)")
            ]
            
            stats_data = []
            for min_depth, max_depth, label in depth_ranges:
                depth_data = df[(df['pressure'] >= min_depth) & (df['pressure'] < max_depth)]
                if not depth_data.empty:
                    stats_data.append({
                        'Depth Range': label,
                        'Count': len(depth_data),
                        'Mean Temp (°C)': depth_data['temperature'].mean(),
                        'Std Temp (°C)': depth_data['temperature'].std(),
                        'Min Temp (°C)': depth_data['temperature'].min(),
                        'Max Temp (°C)': depth_data['temperature'].max()
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.round(2), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading temperature profile data: {e}")

def render_salinity_analysis(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """Render salinity analysis visualizations"""
    
    st.header("🧂 Salinity Analysis")
    
    # Query for temperature-salinity data
    query = """
    SELECT 
        p.id as profile_id, p.latitude, p.longitude, p.juld,
        m.platform_number, m.project_name,
        am.pressure, am.temperature, am.salinity
    FROM argo_profiles p
    JOIN argo_metadata m ON p.metadata_id = m.id
    JOIN argo_measurements am ON p.id = am.profile_id
    WHERE p.latitude BETWEEN %s AND %s
    AND p.longitude BETWEEN %s AND %s
    AND p.juld BETWEEN %s AND %s
    AND am.temperature IS NOT NULL
    AND am.salinity IS NOT NULL
    AND am.pressure IS NOT NULL
    ORDER BY p.juld DESC
    LIMIT 10000
    """
    
    with st.spinner("Loading salinity data..."):
        try:
            data = db_manager.execute_query(query, (lat_min, lat_max, lon_min, lon_max, start_date, end_date))
            df = pd.DataFrame(data) if data else pd.DataFrame()
            
            if df.empty:
                st.warning("No salinity data found for the selected criteria")
                return
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Profiles", df['profile_id'].nunique())
            with col2:
                st.metric("Salinity Range", f"{df['salinity'].min():.2f} to {df['salinity'].max():.2f} PSU")
            with col3:
                st.metric("Temperature Range", f"{df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
            with col4:
                st.metric("Measurements", len(df))
            
            # Temperature-Salinity Diagram
            st.subheader("Temperature-Salinity Diagram")
            
            # Sample data for performance if too large
            if len(df) > 5000:
                st.info(f"Showing 5000 random samples from {len(df)} total measurements")
                df_sample = df.sample(5000)
            else:
                df_sample = df
            
            fig_ts = px.scatter(
                df_sample,
                x='salinity',
                y='temperature',
                color='pressure',
                color_continuous_scale='viridis',
                title='Temperature-Salinity Diagram (colored by pressure)',
                labels={
                    'salinity': 'Salinity (PSU)',
                    'temperature': 'Temperature (°C)',
                    'pressure': 'Pressure (dbar)'
                },
                hover_data=['platform_number', 'juld']
            )
            
            fig_ts.update_layout(height=600)
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Salinity profiles
            st.subheader("Salinity Profiles")
            
            # Select a few profiles to display
            unique_profiles = df[['profile_id', 'platform_number', 'juld']].drop_duplicates().head(5)
            
            fig_sal = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            for i, (_, profile) in enumerate(unique_profiles.iterrows()):
                profile_data = df[df['profile_id'] == profile['profile_id']].sort_values('pressure')
                
                fig_sal.add_trace(go.Scatter(
                    x=profile_data['salinity'],
                    y=-profile_data['pressure'],
                    mode='lines+markers',
                    name=f"Float {profile['platform_number']} ({profile['juld'].strftime('%Y-%m-%d')})",
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig_sal.update_layout(
                title="Salinity Profiles vs Pressure",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Pressure (dbar, negative for depth)",
                height=500
            )
            
            st.plotly_chart(fig_sal, use_container_width=True)
            
            # Salinity distribution by depth
            st.subheader("Salinity Distribution by Depth")
            
            # Create depth bins
            df['depth_bin'] = pd.cut(df['pressure'], bins=[0, 100, 500, 1000, 2000, float('inf')], 
                                   labels=['0-100m', '100-500m', '500-1000m', '1000-2000m', '>2000m'])
            
            fig_box = px.box(
                df.dropna(subset=['depth_bin']),
                x='depth_bin',
                y='salinity',
                title='Salinity Distribution by Depth Range',
                labels={'depth_bin': 'Depth Range', 'salinity': 'Salinity (PSU)'}
            )
            
            fig_box.update_layout(height=500)
            st.plotly_chart(fig_box, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading salinity data: {e}")

def render_time_series(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """Render time series visualizations"""
    
    st.header("📈 Time Series Analysis")
    
    # Query for time series data
    query = """
    SELECT 
        DATE(p.juld) as date,
        AVG(am.temperature) as avg_temperature,
        AVG(am.salinity) as avg_salinity,
        COUNT(*) as measurement_count,
        COUNT(DISTINCT p.id) as profile_count
    FROM argo_profiles p
    JOIN argo_measurements am ON p.id = am.profile_id
    WHERE p.latitude BETWEEN %s AND %s
    AND p.longitude BETWEEN %s AND %s
    AND p.juld BETWEEN %s AND %s
    AND am.temperature IS NOT NULL
    AND am.salinity IS NOT NULL
    GROUP BY DATE(p.juld)
    ORDER BY date
    """
    
    with st.spinner("Loading time series data..."):
        try:
            data = db_manager.execute_query(query, (lat_min, lat_max, lon_min, lon_max, start_date, end_date))
            df = pd.DataFrame(data) if data else pd.DataFrame()
            
            if df.empty:
                st.warning("No time series data found for the selected criteria")
                return
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Days with Data", len(df))
            with col2:
                st.metric("Avg Daily Profiles", f"{df['profile_count'].mean():.1f}")
            with col3:
                st.metric("Total Measurements", df['measurement_count'].sum())
            with col4:
                st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
            
            # Time series plots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Average Temperature Over Time', 'Average Salinity Over Time', 'Data Availability Over Time'),
                vertical_spacing=0.08
            )
            
            # Temperature time series
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['avg_temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='red'),
                    hovertemplate="Date: %{x}<br>Temperature: %{y:.2f}°C<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Salinity time series
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['avg_salinity'],
                    mode='lines+markers',
                    name='Salinity',
                    line=dict(color='blue'),
                    hovertemplate="Date: %{x}<br>Salinity: %{y:.3f} PSU<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Data availability
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['profile_count'],
                    name='Profiles per Day',
                    marker_color='green',
                    hovertemplate="Date: %{x}<br>Profiles: %{y}<extra></extra>"
                ),
                row=3, col=1
            )
            
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Salinity (PSU)", row=2, col=1)
            fig.update_yaxes(title_text="Profile Count", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            
            fig.update_layout(height=800, showlegend=False, title_text="Time Series Analysis")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly aggregation
            if len(df) > 30:  # Only show if we have enough data
                st.subheader("Monthly Trends")
                
                df['month'] = df['date'].dt.to_period('M')
                monthly_data = df.groupby('month').agg({
                    'avg_temperature': 'mean',
                    'avg_salinity': 'mean',
                    'profile_count': 'sum',
                    'measurement_count': 'sum'
                }).reset_index()
                
                monthly_data['month'] = monthly_data['month'].astype(str)
                
                fig_monthly = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Monthly Average Temperature', 'Monthly Average Salinity')
                )
                
                fig_monthly.add_trace(
                    go.Scatter(
                        x=monthly_data['month'],
                        y=monthly_data['avg_temperature'],
                        mode='lines+markers',
                        name='Monthly Temp',
                        line=dict(color='red', width=3)
                    ),
                    row=1, col=1
                )
                
                fig_monthly.add_trace(
                    go.Scatter(
                        x=monthly_data['month'],
                        y=monthly_data['avg_salinity'],
                        mode='lines+markers',
                        name='Monthly Salinity',
                        line=dict(color='blue', width=3)
                    ),
                    row=2, col=1
                )
                
                fig_monthly.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_monthly, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading time series data: {e}")

def render_data_statistics(db_manager, current_user):
    """Render data statistics dashboard"""
    
    st.header("📊 Data Statistics")
    
    # Get comprehensive statistics
    stats = db_manager.get_data_statistics(current_user['role'])
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Floats", stats.get('total_floats', 0))
    with col2:
        st.metric("Total Profiles", stats.get('total_profiles', 0))
    with col3:
        st.metric("Total Measurements", stats.get('total_measurements', 0))
    with col4:
        avg_per_profile = stats.get('total_measurements', 0) / max(stats.get('total_profiles', 1), 1)
        st.metric("Avg Measurements/Profile", f"{avg_per_profile:.0f}")
    
    # Recent activity
    if stats.get('recent_activity'):
        st.subheader("Recent Data Activity (Last 30 Days)")
        
        activity_df = pd.DataFrame(stats['recent_activity'])
        activity_df['date'] = pd.to_datetime(activity_df['date'])
        
        fig_activity = px.bar(
            activity_df,
            x='date',
            y='profiles',
            title='Daily Profile Count (Last 30 Days)',
            labels={'profiles': 'Number of Profiles', 'date': 'Date'}
        )
        
        st.plotly_chart(fig_activity, use_container_width=True)
    
    # Geographic distribution
    if stats.get('geo_distribution'):
        st.subheader("Geographic Distribution")
        
        geo_df = pd.DataFrame(stats['geo_distribution'])
        
        fig_geo = px.scatter(
            geo_df.head(50),  # Top 50 locations
            x='lon_bin',
            y='lat_bin',
            size='count',
            title='Profile Density by Location (10° x 10° bins)',
            labels={'lon_bin': 'Longitude Bin', 'lat_bin': 'Latitude Bin', 'count': 'Profile Count'}
        )
        
        st.plotly_chart(fig_geo, use_container_width=True)

def render_float_trajectories(db_manager, current_user, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """Render float trajectory visualizations"""
    
    st.header("🛰️ Float Trajectories")
    
    # Get list of floats in the area
    float_query = """
    SELECT DISTINCT m.platform_number, COUNT(p.id) as profile_count,
           MIN(p.juld) as first_profile, MAX(p.juld) as last_profile
    FROM argo_metadata m
    JOIN argo_profiles p ON m.id = p.metadata_id
    WHERE p.latitude BETWEEN %s AND %s
    AND p.longitude BETWEEN %s AND %s
    AND p.juld BETWEEN %s AND %s
    GROUP BY m.platform_number
    ORDER BY profile_count DESC
    LIMIT 20
    """
    
    floats_data = db_manager.execute_query(float_query, (lat_min, lat_max, lon_min, lon_max, start_date, end_date))
    
    if not floats_data:
        st.warning("No float data found for the selected criteria")
        return
    
    floats_df = pd.DataFrame(floats_data)
    
    st.subheader("Available Floats")
    st.dataframe(floats_df, use_container_width=True)
    
    # Select float for trajectory
    selected_float = st.selectbox(
        "Select Float for Trajectory:",
        floats_df['platform_number'].tolist()
    )
    
    if selected_float:
        # Get trajectory data
        trajectory_query = """
        SELECT p.latitude, p.longitude, p.juld, p.cycle_number
        FROM argo_profiles p
        JOIN argo_metadata m ON p.metadata_id = m.id
        WHERE m.platform_number = %s
        ORDER BY p.juld
        """
        
        trajectory_data = db_manager.execute_query(trajectory_query, (selected_float,))
        
        if trajectory_data:
            traj_df = pd.DataFrame(trajectory_data)
            
            # Create trajectory map
            st.subheader(f"Trajectory for Float {selected_float}")
            
            fig_traj = px.line_mapbox(
                traj_df,
                lat='latitude',
                lon='longitude',
                hover_data=['cycle_number', 'juld'],
                mapbox_style='open-street-map',
                title=f'Float {selected_float} Trajectory',
                height=600
            )
            
            # Add start and end markers
            fig_traj.add_trace(
                go.Scattermapbox(
                    lat=[traj_df.iloc[0]['latitude']],
                    lon=[traj_df.iloc[0]['longitude']],
                    mode='markers',
                    marker=dict(size=15, color='green'),
                    name='Start',
                    hovertemplate="Start Position<br>Date: %{customdata}<extra></extra>",
                    customdata=[traj_df.iloc[0]['juld']]
                )
            )
            
            fig_traj.add_trace(
                go.Scattermapbox(
                    lat=[traj_df.iloc[-1]['latitude']],
                    lon=[traj_df.iloc[-1]['longitude']],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='End',
                    hovertemplate="End Position<br>Date: %{customdata}<extra></extra>",
                    customdata=[traj_df.iloc[-1]['juld']]
                )
            )
            
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # Trajectory statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Profiles", len(traj_df))
            with col2:
                duration = (traj_df['juld'].max() - traj_df['juld'].min()).days
                st.metric("Mission Duration", f"{duration} days")
            with col3:
                # Calculate approximate distance
                from geopy.distance import geodesic
                total_distance = 0
                for i in range(1, len(traj_df)):
                    coord1 = (traj_df.iloc[i-1]['latitude'], traj_df.iloc[i-1]['longitude'])
                    coord2 = (traj_df.iloc[i]['latitude'], traj_df.iloc[i]['longitude'])
                    total_distance += geodesic(coord1, coord2).kilometers
                st.metric("Approx Distance", f"{total_distance:.0f} km")
            with col4:
                avg_speed = total_distance / max(duration, 1) * 24  # km/day to km/h
                st.metric("Avg Speed", f"{avg_speed:.2f} km/h")

def create_quick_visualization(data, viz_type, current_user):
    """Create quick visualization based on data and type"""
    
    if data is None or data.empty:
        return None
    
    try:
        if viz_type == "Map View" and 'latitude' in data.columns and 'longitude' in data.columns:
            fig = px.scatter_mapbox(
                data,
                lat='latitude',
                lon='longitude',
                mapbox_style='open-street-map',
                title='Geographic Distribution',
                height=500
            )
            return fig
        
        elif viz_type == "Profile Plot" and any(col in data.columns for col in ['pressure', 'depth']):
            y_col = 'pressure' if 'pressure' in data.columns else 'depth'
            
            if 'temperature' in data.columns:
                fig = px.line(
                    data,
                    x='temperature',
                    y=y_col,
                    title='Temperature Profile',
                    labels={'temperature': 'Temperature (°C)', y_col: f'{y_col.title()} (dbar)'}
                )
                fig.update_yaxes(autorange="reversed")  # Depth increases downward
                return fig
        
        elif viz_type == "Time Series" and any(col in data.columns for col in ['juld', 'date', 'time']):
            time_col = 'juld' if 'juld' in data.columns else ('date' if 'date' in data.columns else 'time')
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                y_col = numeric_cols[0]
                fig = px.line(
                    data,
                    x=time_col,
                    y=y_col,
                    title=f'{y_col.title()} Over Time'
                )
                return fig
        
        elif viz_type == "Scatter Plot":
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(
                    data,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f'{numeric_cols[1].title()} vs {numeric_cols[0].title()}'
                )
                return fig
        
        elif viz_type == "Histogram":
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(
                    data,
                    x=numeric_cols[0],
                    title=f'Distribution of {numeric_cols[0].title()}'
                )
                return fig
        
        elif viz_type == "Box Plot":
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.box(
                    data,
                    y=numeric_cols[0],
                    title=f'Box Plot of {numeric_cols[0].title()}'
                )
                return fig
        
    except Exception as e:
        st.error(f"Error creating {viz_type}: {e}")
        return None
    
    return None
