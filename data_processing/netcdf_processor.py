import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import os
import streamlit as st
from config.database import get_sqlalchemy_engine, execute_query
from utils.helpers import safe_float_conversion, validate_coordinates

class NetCDFProcessor:
    """Process ARGO NetCDF files and store in database"""
    
    def __init__(self):
        self.engine = get_sqlalchemy_engine()
    
    def process_argo_file(self, file_path, user_id):
        """Process an ARGO NetCDF file and store in database"""
        try:
            # Open NetCDF file
            ds = xr.open_dataset(file_path)
            
            # Extract metadata
            metadata = self.extract_metadata(ds)
            
            # Extract profile data
            profiles = self.extract_profiles(ds)
            
            # Store in database
            metadata_id = self.store_metadata(metadata, user_id)
            if metadata_id:
                self.store_profiles(profiles, metadata_id)
                return True, "File processed successfully"
            else:
                return False, "Failed to store metadata"
                
        except Exception as e:
            return False, f"Error processing file: {str(e)}"
    
    def extract_metadata(self, ds):
        """Extract metadata from NetCDF dataset"""
        metadata = {}
        
        # Basic float information
        metadata['platform_number'] = str(ds.PLATFORM_NUMBER.values[0]) if 'PLATFORM_NUMBER' in ds else None
        metadata['project_name'] = str(ds.PROJECT_NAME.values[0]) if 'PROJECT_NAME' in ds else None
        metadata['pi_name'] = str(ds.PI_NAME.values[0]) if 'PI_NAME' in ds else None
        metadata['platform_type'] = str(ds.PLATFORM_TYPE.values[0]) if 'PLATFORM_TYPE' in ds else None
        metadata['wmo_inst_type'] = str(ds.WMO_INST_TYPE.values[0]) if 'WMO_INST_TYPE' in ds else None
        
        # Temporal information
        if 'JULD' in ds:
            reference_date = pd.to_datetime('1950-01-01')
            metadata['date_creation'] = reference_date + pd.Timedelta(days=float(ds.JULD.values[0]))
        
        # Spatial bounds
        if 'LATITUDE' in ds and 'LONGITUDE' in ds:
            lats = ds.LATITUDE.values
            lons = ds.LONGITUDE.values
            
            # Remove NaN values
            valid_lats = lats[~np.isnan(lats)]
            valid_lons = lons[~np.isnan(lons)]
            
            if len(valid_lats) > 0 and len(valid_lons) > 0:
                metadata['lat_min'] = float(np.min(valid_lats))
                metadata['lat_max'] = float(np.max(valid_lats))
                metadata['lon_min'] = float(np.min(valid_lons))
                metadata['lon_max'] = float(np.max(valid_lons))
        
        # Data center and institution
        metadata['data_centre'] = str(ds.DATA_CENTRE.values[0]) if 'DATA_CENTRE' in ds else None
        metadata['institution'] = str(ds.INSTITUTION.values[0]) if 'INSTITUTION' in ds else None
        
        # Number of profiles
        metadata['n_profiles'] = len(ds.N_PROF) if 'N_PROF' in ds else 0
        
        return metadata
    
    def extract_profiles(self, ds):
        """Extract profile data from NetCDF dataset"""
        profiles = []
        
        # Get dimensions
        n_prof = ds.dims.get('N_PROF', 0)
        n_levels = ds.dims.get('N_LEVELS', 0)
        
        for prof_idx in range(n_prof):
            try:
                profile = {}
                
                # Basic profile info
                profile['cycle_number'] = int(ds.CYCLE_NUMBER.values[prof_idx]) if 'CYCLE_NUMBER' in ds else None
                profile['direction'] = str(ds.DIRECTION.values[prof_idx]) if 'DIRECTION' in ds else None
                profile['data_mode'] = str(ds.DATA_MODE.values[prof_idx]) if 'DATA_MODE' in ds else None
                
                # Position and time
                if 'LATITUDE' in ds:
                    lat = float(ds.LATITUDE.values[prof_idx])
                    profile['latitude'] = lat if validate_coordinates(lat, None)[0] else None
                
                if 'LONGITUDE' in ds:
                    lon = float(ds.LONGITUDE.values[prof_idx])
                    profile['longitude'] = lon if validate_coordinates(None, lon)[1] else None
                
                if 'JULD' in ds:
                    reference_date = pd.to_datetime('1950-01-01')
                    profile['juld'] = reference_date + pd.Timedelta(days=float(ds.JULD.values[prof_idx]))
                
                # Extract measurement data
                measurements = {}
                
                # Temperature
                if 'TEMP' in ds:
                    temp_data = ds.TEMP.values[prof_idx, :]
                    measurements['temperature'] = [safe_float_conversion(t) for t in temp_data if not np.isnan(t)]
                
                # Salinity
                if 'PSAL' in ds:
                    sal_data = ds.PSAL.values[prof_idx, :]
                    measurements['salinity'] = [safe_float_conversion(s) for s in sal_data if not np.isnan(s)]
                
                # Pressure/Depth
                if 'PRES' in ds:
                    pres_data = ds.PRES.values[prof_idx, :]
                    measurements['pressure'] = [safe_float_conversion(p) for p in pres_data if not np.isnan(p)]
                
                # BGC parameters if available
                bgc_params = ['DOXY', 'CHLA', 'BBP700', 'PH_IN_SITU_TOTAL', 'NITRATE']
                for param in bgc_params:
                    if param in ds:
                        param_data = ds[param].values[prof_idx, :]
                        measurements[param.lower()] = [safe_float_conversion(p) for p in param_data if not np.isnan(p)]
                
                profile['measurements'] = measurements
                profile['n_levels'] = len(measurements.get('pressure', []))
                
                profiles.append(profile)
                
            except Exception as e:
                st.warning(f"Error processing profile {prof_idx}: {str(e)}")
                continue
        
        return profiles
    
    def store_metadata(self, metadata, user_id):
        """Store metadata in database"""
        try:
            insert_query = """
            INSERT INTO argo_metadata (
                platform_number, project_name, pi_name, platform_type, wmo_inst_type,
                date_creation, lat_min, lat_max, lon_min, lon_max, data_centre,
                institution, n_profiles, uploaded_by, uploaded_at
            ) VALUES (
                %(platform_number)s, %(project_name)s, %(pi_name)s, %(platform_type)s, %(wmo_inst_type)s,
                %(date_creation)s, %(lat_min)s, %(lat_max)s, %(lon_min)s, %(lon_max)s, %(data_centre)s,
                %(institution)s, %(n_profiles)s, %(uploaded_by)s, CURRENT_TIMESTAMP
            ) RETURNING id
            """
            
            metadata['uploaded_by'] = user_id
            
            result = execute_query(insert_query, metadata)
            if result and len(result) > 0:
                return result[0]['id']
            return None
            
        except Exception as e:
            st.error(f"Error storing metadata: {str(e)}")
            return None
    
    def store_profiles(self, profiles, metadata_id):
        """Store profile data in database"""
        try:
            for profile in profiles:
                # Store basic profile info
                profile_insert = """
                INSERT INTO argo_profiles (
                    metadata_id, cycle_number, direction, data_mode, latitude, longitude,
                    juld, n_levels, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                ) RETURNING id
                """
                
                profile_result = execute_query(profile_insert, (
                    metadata_id,
                    profile['cycle_number'],
                    profile['direction'],
                    profile['data_mode'],
                    profile['latitude'],
                    profile['longitude'],
                    profile['juld'],
                    profile['n_levels']
                ))
                
                if profile_result and len(profile_result) > 0:
                    profile_id = profile_result[0]['id']
                    
                    # Store measurement data
                    measurements = profile['measurements']
                    
                    for i, pressure in enumerate(measurements.get('pressure', [])):
                        measurement_insert = """
                        INSERT INTO argo_measurements (
                            profile_id, depth_level, pressure, temperature, salinity,
                            doxy, chla, bbp700, ph_in_situ_total, nitrate
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        temp = measurements.get('temperature', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('temperature', [])) else None
                        sal = measurements.get('salinity', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('salinity', [])) else None
                        doxy = measurements.get('doxy', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('doxy', [])) else None
                        chla = measurements.get('chla', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('chla', [])) else None
                        bbp700 = measurements.get('bbp700', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('bbp700', [])) else None
                        ph = measurements.get('ph_in_situ_total', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('ph_in_situ_total', [])) else None
                        nitrate = measurements.get('nitrate', [None] * len(measurements.get('pressure', [])))[i] if i < len(measurements.get('nitrate', [])) else None
                        
                        execute_query(measurement_insert, (
                            profile_id, i + 1, pressure, temp, sal, doxy, chla, bbp700, ph, nitrate
                        ), fetch=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error storing profiles: {str(e)}")
            return False
    
    def get_data_summary(self):
        """Get summary of stored ARGO data"""
        summary_queries = {
            'total_floats': "SELECT COUNT(DISTINCT platform_number) as count FROM argo_metadata",
            'total_profiles': "SELECT COUNT(*) as count FROM argo_profiles",
            'total_measurements': "SELECT COUNT(*) as count FROM argo_measurements",
            'date_range': """
                SELECT 
                    MIN(date_creation) as min_date, 
                    MAX(date_creation) as max_date 
                FROM argo_metadata
            """,
            'spatial_coverage': """
                SELECT 
                    MIN(lat_min) as min_lat, MAX(lat_max) as max_lat,
                    MIN(lon_min) as min_lon, MAX(lon_max) as max_lon
                FROM argo_metadata
            """
        }
        
        summary = {}
        for key, query in summary_queries.items():
            result = execute_query(query)
            if result:
                if key in ['total_floats', 'total_profiles', 'total_measurements']:
                    summary[key] = result[0]['count']
                else:
                    summary[key] = dict(result[0])
            else:
                summary[key] = 0 if key.startswith('total_') else {}
        
        return summary
