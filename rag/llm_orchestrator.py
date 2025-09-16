import os
import json
from groq import Groq
import streamlit as st
from config.settings import settings
from vectorstore.faiss_manager import FAISSManager
from rag.query_processor import QueryProcessor

class LLMOrchestrator:
    """Orchestrate LLM interactions for ARGO data queries"""
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.faiss_manager = FAISSManager()
        self.query_processor = QueryProcessor()
        
        # Model configuration - using Llama3 on Groq for fast inference
        self.model = settings.DEFAULT_MODEL
        self.max_tokens = 2000
        self.temperature = 0.1  # Low temperature for more precise responses
    
    def process_natural_language_query(self, user_query, user_role, conversation_history=None):
        """Process natural language query and return structured response"""
        try:
            # Step 1: Get relevant context from vector database
            context = self.faiss_manager.get_context_for_query(user_query)
            
            # Step 2: Analyze query intent and generate SQL
            sql_query, query_type = self.query_processor.translate_to_sql(
                user_query, context, user_role
            )
            
            # Step 3: Execute query if SQL was generated
            query_results = None
            if sql_query:
                success, results = self.query_processor.execute_safe_query(sql_query, user_role)
                if success:
                    query_results = results
                else:
                    return {
                        'success': False,
                        'error': f"Query execution failed: {results}",
                        'sql_query': sql_query
                    }
            
            # Step 4: Generate natural language response
            response = self.generate_response(
                user_query, context, sql_query, query_results, 
                query_type, conversation_history
            )
            
            return {
                'success': True,
                'response': response,
                'sql_query': sql_query,
                'query_type': query_type,
                'data': query_results,
                'context': context
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"LLM orchestration error: {str(e)}"
            }
    
    def generate_response(self, user_query, context, sql_query, query_results, 
                         query_type, conversation_history=None):
        """Generate natural language response using Groq"""
        try:
            # Build conversation context
            messages = []
            
            # System prompt
            system_prompt = self._build_system_prompt(query_type)
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history if available
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append(msg)
            
            # Build user message with context and results
            user_message = self._build_user_message(
                user_query, context, sql_query, query_results, query_type
            )
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)
            
            return parsed_response
            
        except Exception as e:
            return {
                'text': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'explanation': "There was a technical issue with the response generation.",
                'suggestions': ["Please try rephrasing your query", "Check if the data exists in the database"]
            }
    
    def _build_system_prompt(self, query_type):
        """Build system prompt based on query type"""
        base_prompt = """You are an AI assistant specialized in ARGO oceanographic data analysis. 
        You help users understand and explore ocean temperature, salinity, and biogeochemical data 
        from autonomous profiling floats.

        Your capabilities include:
        - Analyzing ARGO float profiles and trajectories
        - Explaining oceanographic parameters and measurements
        - Providing insights about ocean conditions and trends
        - Helping with data interpretation and visualization

        Guidelines:
        - Always provide accurate, scientifically sound information
        - Explain technical terms when addressing non-experts
        - Suggest relevant follow-up queries when appropriate
        - If data is limited, explain the limitations clearly
        - Use metric units (°C, PSU, dbar, etc.) for oceanographic measurements
        - Be helpful but honest about data availability and quality

        Response format: Always respond in JSON format with these fields:
        {
            "text": "Main response text",
            "explanation": "Brief technical explanation if needed",
            "suggestions": ["List of suggested follow-up queries"],
            "data_summary": "Summary of data used (if applicable)",
            "visualization_hints": "Suggestions for charts/plots (if applicable)"
        }"""
        
        # Add query-specific context
        if query_type == 'data_query':
            base_prompt += "\n\nFocus on data analysis and interpretation."
        elif query_type == 'visualization':
            base_prompt += "\n\nProvide visualization guidance and chart recommendations."
        elif query_type == 'information':
            base_prompt += "\n\nProvide educational information about oceanography and ARGO data."
        
        return base_prompt
    
    def _build_user_message(self, user_query, context, sql_query, query_results, query_type):
        """Build user message with all relevant information"""
        message_parts = []
        
        message_parts.append(f"User Query: {user_query}")
        
        if context:
            message_parts.append(f"\nRelevant Data Context:\n{context}")
        
        if sql_query:
            message_parts.append(f"\nGenerated SQL Query:\n{sql_query}")
        
        if query_results is not None:
            if hasattr(query_results, 'shape'):
                message_parts.append(f"\nQuery Results: {query_results.shape[0]} rows returned")
                
                # Add sample of results for small datasets
                if query_results.shape[0] <= 10:
                    message_parts.append(f"\nData Sample:\n{query_results.to_string()}")
                else:
                    message_parts.append(f"\nData Sample (first 5 rows):\n{query_results.head().to_string()}")
                    
                # Add basic statistics for numerical columns
                numeric_cols = query_results.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    stats = query_results[numeric_cols].describe()
                    message_parts.append(f"\nData Statistics:\n{stats.to_string()}")
            else:
                message_parts.append(f"\nQuery Results: {str(query_results)}")
        
        message_parts.append(f"\nQuery Type: {query_type}")
        
        message_parts.append("""\nPlease provide a helpful response that:
        1. Directly answers the user's question
        2. Explains the oceanographic significance of any findings
        3. Suggests relevant follow-up questions
        4. Provides visualization suggestions if appropriate
        5. Explains any limitations or caveats in the data""")
        
        return "\n".join(message_parts)
    
    def suggest_queries(self, current_query, user_role):
        """Suggest related queries based on current query"""
        try:
            prompt = f"""Based on this ARGO oceanographic data query: "{current_query}"
            
            Suggest 5 related queries that would be interesting for a {user_role} to explore.
            Consider:
            - Different time periods or locations
            - Related oceanographic parameters
            - Comparative analysis opportunities
            - Data quality or methodology questions
            
            Respond in JSON format:
            {{
                "suggestions": ["query1", "query2", "query3", "query4", "query5"]
            }}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('suggestions', [])
            
        except Exception as e:
            st.error(f"Error generating query suggestions: {e}")
            return []
    
    def explain_query_results(self, query, results, user_role):
        """Provide detailed explanation of query results"""
        try:
            if results is None or (hasattr(results, 'empty') and results.empty):
                return {
                    'text': "No data was found matching your query criteria.",
                    'explanation': "This could be due to the specific location, time period, or parameters requested.",
                    'suggestions': [
                        "Try expanding the geographical area",
                        "Use a broader time range",
                        "Check if the requested parameters are available"
                    ]
                }
            
            # Generate explanation prompt
            data_summary = ""
            if hasattr(results, 'shape'):
                data_summary = f"Dataset contains {results.shape[0]} records with {results.shape[1]} columns."
                
                # Add column information
                columns = list(results.columns)
                data_summary += f"\nColumns: {', '.join(columns)}"
                
                # Add value ranges for numeric columns
                numeric_cols = results.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    ranges = []
                    for col in numeric_cols:
                        min_val = results[col].min()
                        max_val = results[col].max()
                        ranges.append(f"{col}: {min_val:.2f} to {max_val:.2f}")
                    data_summary += f"\nValue ranges: {'; '.join(ranges)}"
            
            prompt = f"""Explain these ARGO oceanographic query results for a {user_role}:

            Original Query: {query}
            
            Data Summary: {data_summary}
            
            Provide an expert interpretation that includes:
            1. What the data shows about ocean conditions
            2. Oceanographic significance of the patterns/values
            3. Data quality considerations
            4. Potential applications or implications
            5. Limitations or caveats
            
            Respond in JSON format with fields: text, explanation, suggestions, data_summary, visualization_hints"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                'text': f"Error generating explanation: {str(e)}",
                'explanation': "Technical issue with response generation",
                'suggestions': []
            }
    
    def generate_visualization_suggestions(self, query, results):
        """Generate specific visualization suggestions based on query and results"""
        try:
            if results is None or (hasattr(results, 'empty') and results.empty):
                return []
            
            suggestions = []
            
            # Check data characteristics
            if hasattr(results, 'columns'):
                columns = list(results.columns)
                
                # Geospatial visualizations
                if 'latitude' in columns and 'longitude' in columns:
                    suggestions.append({
                        'type': 'map',
                        'title': 'Geographic Distribution',
                        'description': 'Plot float locations on a map'
                    })
                
                # Time series
                if any(col in columns for col in ['juld', 'date', 'time']):
                    suggestions.append({
                        'type': 'timeseries',
                        'title': 'Time Series Plot',
                        'description': 'Show data trends over time'
                    })
                
                # Profile plots
                if 'pressure' in columns or 'depth' in columns:
                    if 'temperature' in columns:
                        suggestions.append({
                            'type': 'profile',
                            'title': 'Temperature Profile',
                            'description': 'Temperature vs depth/pressure plot'
                        })
                    if 'salinity' in columns:
                        suggestions.append({
                            'type': 'profile',
                            'title': 'Salinity Profile',
                            'description': 'Salinity vs depth/pressure plot'
                        })
                
                # Scatter plots for correlations
                numeric_cols = results.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    suggestions.append({
                        'type': 'scatter',
                        'title': 'Parameter Correlation',
                        'description': 'Explore relationships between parameters'
                    })
            
            return suggestions
            
        except Exception as e:
            st.error(f"Error generating visualization suggestions: {e}")
            return []
