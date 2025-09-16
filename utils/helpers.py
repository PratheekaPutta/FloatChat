import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
import streamlit as st
from typing import Union, List, Tuple, Optional, Any
import json

def safe_float_conversion(value) -> Optional[float]:
    """Safely convert value to float, return None for invalid values"""
    try:
        if value is None or pd.isna(value):
            return None
        
        # Handle string inputs
        if isinstance(value, str):
            value = value.strip()
            if value == '' or value.lower() in ['nan', 'null', 'none']:
                return None
        
        # Convert to float
        result = float(value)
        
        # Check for invalid float values
        if np.isnan(result) or np.isinf(result):
            return None
        
        return result
        
    except (ValueError, TypeError, OverflowError):
        return None

def safe_int_conversion(value) -> Optional[int]:
    """Safely convert value to int, return None for invalid values"""
    try:
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, str):
            value = value.strip()
            if value == '' or value.lower() in ['nan', 'null', 'none']:
                return None
        
        # Convert to int
        result = int(float(value))  # Convert through float to handle strings like "123.0"
        return result
        
    except (ValueError, TypeError, OverflowError):
        return None

def validate_coordinates(latitude: Any, longitude: Any) -> Tuple[Optional[float], Optional[float]]:
    """Validate and convert latitude/longitude coordinates"""
    
    # Validate latitude
    lat = safe_float_conversion(latitude)
    if lat is not None:
        if not (-90 <= lat <= 90):
            lat = None
    
    # Validate longitude
    lon = safe_float_conversion(longitude)
    if lon is not None:
        if not (-180 <= lon <= 180):
            # Try to normalize longitude to -180 to 180 range
            while lon > 180:
                lon -= 360
            while lon < -180:
                lon += 360
            
            if not (-180 <= lon <= 180):
                lon = None
    
    return lat, lon

def validate_date(date_value: Any) -> Optional[datetime]:
    """Validate and convert date value to datetime"""
    try:
        if date_value is None or pd.isna(date_value):
            return None
        
        # If already datetime
        if isinstance(date_value, datetime):
            return date_value
        
        # If pandas datetime
        if hasattr(date_value, 'to_pydatetime'):
            return date_value.to_pydatetime()
        
        # If string, try to parse
        if isinstance(date_value, str):
            date_value = date_value.strip()
            if date_value == '':
                return None
            
            # Try common date formats
            formats = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
        
        # Try pandas to_datetime as last resort
        return pd.to_datetime(date_value).to_pydatetime()
        
    except Exception:
        return None

def validate_pressure_depth(pressure: Any) -> Optional[float]:
    """Validate oceanographic pressure/depth values"""
    press = safe_float_conversion(pressure)
    
    if press is not None:
        # Pressure should be positive and reasonable for ocean depths
        if press < 0:
            return None
        if press > 11000:  # Deepest ocean is ~11km, so 11000 dbar is max reasonable
            return None
    
    return press

def validate_temperature(temperature: Any) -> Optional[float]:
    """Validate oceanographic temperature values"""
    temp = safe_float_conversion(temperature)
    
    if temp is not None:
        # Ocean temperature typically ranges from -2°C to 35°C
        if temp < -5 or temp > 40:
            return None
    
    return temp

def validate_salinity(salinity: Any) -> Optional[float]:
    """Validate oceanographic salinity values"""
    sal = safe_float_conversion(salinity)
    
    if sal is not None:
        # Practical salinity typically ranges from 0 to 42 PSU
        if sal < 0 or sal > 45:
            return None
    
    return sal

def validate_oxygen(oxygen: Any) -> Optional[float]:
    """Validate dissolved oxygen values"""
    oxy = safe_float_conversion(oxygen)
    
    if oxy is not None:
        # Dissolved oxygen typically ranges from 0 to 500 μmol/kg
        if oxy < 0 or oxy > 600:
            return None
    
    return oxy

def clean_string_value(value: Any) -> Optional[str]:
    """Clean and validate string values"""
    if value is None or pd.isna(value):
        return None
    
    if not isinstance(value, str):
        value = str(value)
    
    # Clean the string
    value = value.strip()
    
    # Remove null bytes and other problematic characters
    value = value.replace('\x00', '')
    
    # Return None for empty strings
    if value == '' or value.lower() in ['nan', 'null', 'none']:
        return None
    
    return value

def format_coordinate(coord: float, is_latitude: bool = True) -> str:
    """Format coordinate for display"""
    if coord is None:
        return "N/A"
    
    if is_latitude:
        direction = "N" if coord >= 0 else "S"
    else:
        direction = "E" if coord >= 0 else "W"
    
    return f"{abs(coord):.3f}°{direction}"

def format_date_for_display(date_value: Any) -> str:
    """Format date for display"""
    date_obj = validate_date(date_value)
    if date_obj:
        return date_obj.strftime("%Y-%m-%d %H:%M")
    return "N/A"

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """Calculate distance between two coordinates using Haversine formula"""
    try:
        # Validate coordinates
        lat1, lon1 = validate_coordinates(lat1, lon1)
        lat2, lon2 = validate_coordinates(lat2, lon2)
        
        if None in [lat1, lon1, lat2, lon2]:
            return None
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
        
    except Exception:
        return None

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from text"""
    if not isinstance(text, str):
        return []
    
    # Pattern to match numbers (including floats and negatives)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers

def parse_coordinate_string(coord_str: str) -> Optional[float]:
    """Parse coordinate string like '45.123N' or '-123.456' to decimal degrees"""
    if not isinstance(coord_str, str):
        return None
    
    coord_str = coord_str.strip().upper()
    
    # Extract numbers from the string
    numbers = extract_numbers_from_text(coord_str)
    if not numbers:
        return None
    
    coord = numbers[0]
    
    # Check for direction indicators
    if 'S' in coord_str or 'W' in coord_str:
        coord = -abs(coord)
    elif 'N' in coord_str or 'E' in coord_str:
        coord = abs(coord)
    
    return coord

def convert_julian_day(julian_day: float, reference_date: str = "1950-01-01") -> Optional[datetime]:
    """Convert Julian day to datetime"""
    try:
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        result_date = ref_date + timedelta(days=float(julian_day))
        return result_date
    except Exception:
        return None

def create_summary_statistics(data: pd.DataFrame, numeric_only: bool = True) -> dict:
    """Create summary statistics for a DataFrame"""
    if data.empty:
        return {}
    
    stats = {}
    
    if numeric_only:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data_subset = data[numeric_cols]
    else:
        data_subset = data
    
    for col in data_subset.columns:
        col_stats = {}
        
        if data_subset[col].dtype in [np.int64, np.float64]:
            # Numeric statistics
            col_stats['count'] = int(data_subset[col].count())
            col_stats['mean'] = float(data_subset[col].mean()) if col_stats['count'] > 0 else None
            col_stats['std'] = float(data_subset[col].std()) if col_stats['count'] > 1 else None
            col_stats['min'] = float(data_subset[col].min()) if col_stats['count'] > 0 else None
            col_stats['max'] = float(data_subset[col].max()) if col_stats['count'] > 0 else None
            col_stats['median'] = float(data_subset[col].median()) if col_stats['count'] > 0 else None
            col_stats['null_count'] = int(data_subset[col].isnull().sum())
        else:
            # Categorical statistics
            col_stats['count'] = int(data_subset[col].count())
            col_stats['unique'] = int(data_subset[col].nunique())
            col_stats['null_count'] = int(data_subset[col].isnull().sum())
            if col_stats['count'] > 0:
                col_stats['top'] = str(data_subset[col].mode().iloc[0])
                col_stats['freq'] = int(data_subset[col].value_counts().iloc[0])
        
        stats[col] = col_stats
    
    return stats

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """Detect outliers in a data series"""
    if data.empty or data.isnull().all():
        return pd.Series([], dtype=bool)
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        return pd.Series([False] * len(data))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    if not isinstance(filename, str):
        filename = str(filename)
    
    # Remove problematic characters
    filename = re.sub(r'[^\w\s-.]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 255 - len(ext) - 1
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename.strip('_')

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """Split DataFrame into smaller chunks"""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size])
    return chunks

def safe_json_serialize(obj: Any) -> str:
    """Safely serialize object to JSON string"""
    def json_serializer(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_serializer, ensure_ascii=False, indent=2)
    except Exception as e:
        return f'{{"error": "Serialization failed: {str(e)}"}}'

def validate_email(email: str) -> bool:
    """Validate email address format"""
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def generate_color_palette(n_colors: int, palette_type: str = 'qualitative') -> List[str]:
    """Generate color palette for visualizations"""
    if palette_type == 'qualitative':
        # Use plotly qualitative colors
        import plotly.colors as pc
        colors = pc.qualitative.Plotly + pc.qualitative.Set3 + pc.qualitative.Pastel
        return (colors * ((n_colors // len(colors)) + 1))[:n_colors]
    
    elif palette_type == 'sequential':
        # Generate sequential colors
        import plotly.colors as pc
        return pc.sample_colorscale('viridis', [i/(n_colors-1) for i in range(n_colors)])
    
    else:
        # Default to simple color cycle
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return (base_colors * ((n_colors // len(base_colors)) + 1))[:n_colors]

def create_download_link(data: Any, filename: str, mime_type: str = 'text/csv') -> None:
    """Create download link using Streamlit"""
    if isinstance(data, pd.DataFrame):
        if mime_type == 'text/csv':
            data_str = data.to_csv(index=False)
        elif mime_type == 'application/json':
            data_str = data.to_json(orient='records', indent=2)
        else:
            data_str = str(data)
    else:
        data_str = str(data)
    
    st.download_button(
        label=f"Download {filename}",
        data=data_str,
        file_name=filename,
        mime=mime_type
    )

def log_error(error_message: str, context: dict = None) -> None:
    """Log error with context (simple logging for now)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ERROR: {error_message}"
    
    if context:
        log_entry += f" | Context: {context}"
    
    # For now, just print to console
    # In production, this would write to a log file or logging service
    print(log_entry)
    
    # Also display in Streamlit if available
    try:
        st.error(f"Error: {error_message}")
    except:
        pass  # If not in Streamlit context

def performance_timer(func_name: str = None):
    """Decorator to measure function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                name = func_name or func.__name__
                print(f"Performance: {name} took {duration:.3f} seconds")
                
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                name = func_name or func.__name__
                print(f"Performance: {name} failed after {duration:.3f} seconds: {e}")
                raise
        return wrapper
    return decorator
