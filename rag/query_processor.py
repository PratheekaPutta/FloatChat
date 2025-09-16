import re
import json
import streamlit as st
from groq import Groq
from config.settings import settings
from data_processing.database_manager import DatabaseManager
from vectorstore.faiss_manager import FAISSManager

class QueryProcessor:
    """Process natural language queries and convert to SQL"""
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.db_manager = DatabaseManager()
        self.faiss_manager = FAISSManager()
        
        # Model configuration - using Llama3 on Groq for fast inference
        self.model = settings.DEFAULT_MODEL
    
    def translate_to_sql(self, natural_query, context, user_role):
        """Translate natural language query to SQL"""
        try:
            # Get database schema information
            schema_info = self._get_schema_info(user_role)
            
            # Build SQL generation prompt
            prompt = self._build_sql_prompt(natural_query, context, schema_info, user_role)
            
            # Generate SQL using LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            sql_query = result.get('sql_query', '')
            query_type = result.get('query_type', 'data_query')
            confidence = result.get('confidence', 0.5)
            
            # Validate and sanitize SQL
            if sql_query and confidence > 0.3:
                sanitized_sql = self._sanitize_sql(sql_query, user_role)
                return sanitized_sql, query_type
            else:
                return None, 'information'
            
        except Exception as e:
            st.error(f"Error translating query to SQL: {e}")
            return None, 'error'
    
    def _build_sql_prompt(self, query, context, schema_info, user_role):
        """Build prompt for SQL generation"""
        prompt = f"""You are an expert SQL generator for ARGO oceanographic data queries.
        
        User Role: {user_role}
        User Query: {query}
        
        Database Schema:
        {schema_info}
        
        Relevant Data Context:
        {context}
        
        Generate a SQL query that:
        1. Answers the user's question accurately
        2. Uses appropriate JOINs between tables
        3. Includes relevant WHERE clauses for filtering
        4. Uses proper aggregations when needed
        5. Follows role-based access restrictions
        6. Limits results appropriately (max 1000 rows for Viewer, 50000 for Researcher)
        
        Common ARGO query patterns:
        - Geographic filtering: latitude BETWEEN lat1 AND lat2 AND longitude BETWEEN lon1 AND lon2
        - Time filtering: juld BETWEEN 'start_date' AND 'end_date'
        - Parameter filtering: temperature IS NOT NULL, salinity > threshold
        - Float filtering: platform_number = 'float_id'
        - Profile filtering: cycle_number, direction = 'A'/'D'
        
        Respond in JSON format:
        {{
            "sql_query": "SELECT statement with proper syntax",
            "query_type": "data_query|visualization|information|statistics",
            "confidence": 0.0-1.0,
            "explanation": "Brief explanation of the query logic"
        }}
        
        If the query cannot be translated to SQL (e.g., general questions), set confidence to 0.0.
        """
        
        return prompt
    
    def _get_schema_info(self, user_role):
        """Get database schema information for SQL generation"""
        schema_parts = []
        
        # Get accessible tables
        tables = self.db_manager.get_available_tables(user_role)
        
        for table in tables:
            columns = self.db_manager.get_table_schema(table, user_role)
            if columns:
                schema_parts.append(f"\nTable: {table}")
                for col in columns:
                    schema_parts.append(f"  - {col['column_name']} ({col['data_type']})")
        
        # Add table relationships
        relationships = """
        
        Table Relationships:
        - argo_profiles.metadata_id → argo_metadata.id
        - argo_measurements.profile_id → argo_profiles.id
        
        Key Fields:
        - platform_number: ARGO float identifier
        - cycle_number: Profile cycle number
        - juld: Julian date (timestamp)
        - latitude, longitude: Geographic coordinates
        - pressure: Water pressure (dbar)
        - temperature: Water temperature (°C)
        - salinity: Practical salinity (PSU)
        - doxy: Dissolved oxygen (μmol/kg)
        """
        
        schema_parts.append(relationships)
        
        return "\n".join(schema_parts)
    
    def _sanitize_sql(self, sql_query, user_role):
        """Sanitize and validate SQL query"""
        try:
            # Remove comments and normalize whitespace
            sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
            sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
            sql_query = ' '.join(sql_query.split())
            
            # Check for dangerous operations
            dangerous_keywords = [
                'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER',
                'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
            ]
            
            sql_upper = sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    raise ValueError(f"Dangerous SQL operation detected: {keyword}")
            
            # Ensure query starts with SELECT
            if not sql_upper.strip().startswith('SELECT'):
                raise ValueError("Only SELECT queries are allowed")
            
            # Apply role-based row limits
            limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
            max_rows = limits.get('max_rows')
            
            if max_rows and 'LIMIT' not in sql_upper:
                sql_query = f"{sql_query} LIMIT {max_rows}"
            
            return sql_query
            
        except Exception as e:
            st.error(f"SQL sanitization error: {e}")
            return None
    
    def execute_safe_query(self, sql_query, user_role):
        """Execute SQL query safely with role-based restrictions"""
        try:
            if not sql_query:
                return False, "No SQL query provided"
            
            # Execute query through database manager
            success, result = self.db_manager.execute_user_query(sql_query, user_role)
            
            if success:
                return True, result
            else:
                return False, result
            
        except Exception as e:
            return False, f"Query execution error: {str(e)}"
    
    def analyze_query_intent(self, query):
        """Analyze user query intent"""
        try:
            prompt = f"""Analyze this ARGO oceanographic data query and determine the user's intent:

            Query: {query}
            
            Classify the intent as one of:
            1. data_retrieval - User wants specific data records
            2. statistics - User wants statistical summaries
            3. visualization - User wants charts/plots
            4. comparison - User wants to compare different datasets
            5. information - User wants general information about ARGO/oceanography
            6. location_search - User wants data from specific geographic areas
            7. time_series - User wants temporal analysis
            8. profile_analysis - User wants vertical profile analysis
            
            Also extract key parameters:
            - Geographic bounds (if mentioned)
            - Time periods (if mentioned)
            - Parameters of interest (temperature, salinity, etc.)
            - Float IDs (if mentioned)
            - Analysis type requested
            
            Respond in JSON format:
            {{
                "intent": "category",
                "confidence": 0.0-1.0,
                "parameters": {{
                    "geographic": {{"lat_min": null, "lat_max": null, "lon_min": null, "lon_max": null}},
                    "temporal": {{"start_date": null, "end_date": null}},
                    "oceanographic": ["list of parameters"],
                    "float_ids": ["list of platform numbers"],
                    "analysis_type": "description"
                }},
                "keywords": ["relevant keywords"],
                "complexity": "simple|moderate|complex"
            }}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error analyzing query intent: {e}")
            return {
                "intent": "information",
                "confidence": 0.5,
                "parameters": {},
                "keywords": [],
                "complexity": "simple"
            }
    
    def extract_oceanographic_entities(self, query):
        """Extract oceanographic entities from query"""
        entities = {
            'parameters': [],
            'locations': [],
            'time_references': [],
            'float_references': [],
            'depth_references': []
        }
        
        query_lower = query.lower()
        
        # Extract parameters
        parameters = {
            'temperature': ['temperature', 'temp', 'thermal'],
            'salinity': ['salinity', 'salt', 'psal'],
            'oxygen': ['oxygen', 'dissolved oxygen', 'doxy', 'o2'],
            'pressure': ['pressure', 'depth', 'pres'],
            'chlorophyll': ['chlorophyll', 'chla', 'chl'],
            'ph': ['ph', 'acidity', 'alkalinity'],
            'nitrate': ['nitrate', 'nitrogen', 'nutrients']
        }
        
        for param, keywords in parameters.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['parameters'].append(param)
        
        # Extract locations (basic pattern matching)
        location_patterns = [
            r'(\d+(?:\.\d+)?)[°]?\s*[ns]',  # Latitude
            r'(\d+(?:\.\d+)?)[°]?\s*[ew]',  # Longitude
            r'equator', r'tropical', r'subtropical', r'polar',
            r'atlantic', r'pacific', r'indian', r'ocean',
            r'arabian\s+sea', r'bay\s+of\s+bengal', r'mediterranean'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower)
            entities['locations'].extend(matches)
        
        # Extract time references
        time_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'last\s+(\d+)\s+(days?|weeks?|months?|years?)',
            r'recent', r'latest', r'current'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            entities['time_references'].extend(matches)
        
        # Extract float references
        float_patterns = [
            r'float\s+(\d+)',
            r'platform\s+(\d+)',
            r'wmo\s+(\d+)'
        ]
        
        for pattern in float_patterns:
            matches = re.findall(pattern, query_lower)
            entities['float_references'].extend(matches)
        
        # Extract depth references
        depth_patterns = [
            r'(\d+)\s*(?:m|meters?|dbar|pressure)',
            r'surface', r'deep', r'bottom', r'shallow'
        ]
        
        for pattern in depth_patterns:
            matches = re.findall(pattern, query_lower)
            entities['depth_references'].extend(matches)
        
        return entities
    
    def suggest_sql_improvements(self, sql_query, user_role):
        """Suggest improvements to SQL query"""
        try:
            prompt = f"""Review this SQL query for ARGO oceanographic data and suggest improvements:

            SQL Query: {sql_query}
            User Role: {user_role}
            
            Consider:
            1. Performance optimization (indexes, query structure)
            2. Data quality filters (NULL handling, outlier detection)
            3. Oceanographic best practices
            4. Role-appropriate data access
            5. Result clarity and usefulness
            
            Respond in JSON format:
            {{
                "improvements": ["list of suggestions"],
                "optimized_query": "improved SQL query",
                "performance_tips": ["performance suggestions"],
                "data_quality_notes": ["data quality considerations"]
            }}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error suggesting SQL improvements: {e}")
            return {
                "improvements": [],
                "optimized_query": sql_query,
                "performance_tips": [],
                "data_quality_notes": []
            }
