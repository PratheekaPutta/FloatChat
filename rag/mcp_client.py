import json
import asyncio
from typing import Dict, Any, List, Optional
import streamlit as st
from groq import Groq
from config.settings import settings
import logging

class MCPClient:
    """Model Context Protocol client for orchestrating LLM interactions"""
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        # Using Llama3 on Groq for fast inference
        self.model = settings.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)
        
        # MCP protocol configuration
        self.protocol_version = "1.0"
        self.max_context_length = 16000
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for the LLM"""
        return {
            "database_query": {
                "name": "database_query",
                "description": "Execute SQL queries on ARGO oceanographic database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {"type": "string", "description": "SQL query to execute"},
                        "user_role": {"type": "string", "description": "User role for access control"}
                    },
                    "required": ["sql_query", "user_role"]
                }
            },
            "vector_search": {
                "name": "vector_search",
                "description": "Search for similar ARGO data using vector embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "Natural language search query"},
                        "top_k": {"type": "integer", "description": "Number of results to return", "default": 10}
                    },
                    "required": ["query_text"]
                }
            },
            "data_visualization": {
                "name": "data_visualization",
                "description": "Generate visualization recommendations for oceanographic data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string", "description": "Type of data to visualize"},
                        "parameters": {"type": "array", "items": {"type": "string"}, "description": "Data parameters"},
                        "visualization_type": {"type": "string", "description": "Preferred visualization type"}
                    },
                    "required": ["data_type"]
                }
            },
            "oceanographic_analysis": {
                "name": "oceanographic_analysis",
                "description": "Provide oceanographic domain expertise and analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_summary": {"type": "string", "description": "Summary of data to analyze"},
                        "analysis_type": {"type": "string", "description": "Type of analysis requested"},
                        "context": {"type": "string", "description": "Additional context for analysis"}
                    },
                    "required": ["data_summary"]
                }
            }
        }
    
    async def process_request(self, user_message: str, context: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """Process user request using MCP protocol"""
        try:
            # Prepare MCP request
            mcp_request = self._prepare_mcp_request(user_message, context, user_role)
            
            # Send request to LLM
            response = await self._send_llm_request(mcp_request)
            
            # Process tool calls if any
            if response.get("tool_calls"):
                tool_results = await self._execute_tool_calls(response["tool_calls"], user_role)
                response["tool_results"] = tool_results
            
            return {
                "success": True,
                "response": response,
                "protocol_version": self.protocol_version
            }
            
        except Exception as e:
            self.logger.error(f"MCP request processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "protocol_version": self.protocol_version
            }
    
    def _prepare_mcp_request(self, user_message: str, context: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """Prepare MCP-compliant request"""
        return {
            "protocol_version": self.protocol_version,
            "request_id": f"req_{hash(user_message) % 1000000}",
            "user_message": user_message,
            "context": context,
            "user_role": user_role,
            "available_tools": list(self.tools.keys()),
            "constraints": self._get_role_constraints(user_role),
            "timestamp": str(pd.Timestamp.now())
        }
    
    async def _send_llm_request(self, mcp_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to LLM with MCP protocol"""
        try:
            # Build system prompt with MCP context
            system_prompt = self._build_mcp_system_prompt(mcp_request)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._format_user_message(mcp_request)}
            ]
            
            # Add context messages if available
            if mcp_request.get("context", {}).get("conversation_history"):
                history = mcp_request["context"]["conversation_history"][-4:]  # Last 4 messages
                messages.extend(history)
            
            # Send to OpenAI with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": tool} for tool in self.tools.values()],
                tool_choice="auto",
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            message = response.choices[0].message
            
            result = {
                "content": message.content,
                "tool_calls": []
            }
            
            # Process tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "function": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM request error: {e}")
            raise
    
    def _build_mcp_system_prompt(self, mcp_request: Dict[str, Any]) -> str:
        """Build system prompt for MCP interaction"""
        user_role = mcp_request.get("user_role", "Viewer")
        constraints = mcp_request.get("constraints", {})
        
        prompt = f"""You are an AI assistant operating under the Model Context Protocol (MCP) v{self.protocol_version}.
        
        ROLE: ARGO Oceanographic Data Analysis Expert
        USER_ROLE: {user_role}
        
        CAPABILITIES:
        - Analyze ARGO float data (temperature, salinity, pressure, BGC parameters)
        - Generate SQL queries for data retrieval
        - Provide oceanographic domain expertise
        - Suggest appropriate data visualizations
        - Explain oceanographic phenomena and data patterns
        
        AVAILABLE TOOLS:
        {json.dumps(list(self.tools.keys()), indent=2)}
        
        ROLE CONSTRAINTS:
        {json.dumps(constraints, indent=2)}
        
        PROTOCOL RULES:
        1. Always respond in valid JSON format
        2. Use tools when data access or analysis is needed
        3. Provide scientifically accurate oceanographic information
        4. Respect user role limitations for data access
        5. Explain technical concepts appropriately for user level
        6. Suggest follow-up queries when relevant
        
        RESPONSE FORMAT:
        {{
            "text": "Main response to user",
            "explanation": "Technical explanation if needed",
            "suggestions": ["Follow-up query suggestions"],
            "tool_usage": "Description of tools used",
            "confidence": 0.0-1.0,
            "data_quality_notes": "Any data quality considerations"
        }}
        
        OCEANOGRAPHIC CONTEXT:
        - ARGO floats measure ocean properties globally
        - Temperature in °C, Salinity in PSU, Pressure in dbar
        - BGC parameters include oxygen, chlorophyll, pH, nitrate
        - Data quality flags: A=adjusted, D=delayed mode, R=real-time
        - Geographical coordinates in decimal degrees
        """
        
        return prompt
    
    def _format_user_message(self, mcp_request: Dict[str, Any]) -> str:
        """Format user message with context"""
        message_parts = [
            f"USER QUERY: {mcp_request['user_message']}",
            f"REQUEST ID: {mcp_request['request_id']}"
        ]
        
        # Add context information
        context = mcp_request.get("context", {})
        if context.get("relevant_data"):
            message_parts.append(f"RELEVANT DATA CONTEXT:\n{context['relevant_data']}")
        
        if context.get("previous_query"):
            message_parts.append(f"PREVIOUS QUERY: {context['previous_query']}")
        
        if context.get("user_preferences"):
            message_parts.append(f"USER PREFERENCES: {context['user_preferences']}")
        
        return "\n\n".join(message_parts)
    
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]], user_role: str) -> List[Dict[str, Any]]:
        """Execute tool calls and return results"""
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call["function"]
                arguments = tool_call["arguments"]
                
                # Execute tool based on name
                if tool_name == "database_query":
                    result = await self._execute_database_query(arguments, user_role)
                elif tool_name == "vector_search":
                    result = await self._execute_vector_search(arguments)
                elif tool_name == "data_visualization":
                    result = await self._execute_data_visualization(arguments)
                elif tool_name == "oceanographic_analysis":
                    result = await self._execute_oceanographic_analysis(arguments)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
                
                results.append({
                    "tool_call_id": tool_call["id"],
                    "function": tool_name,
                    "result": result
                })
                
            except Exception as e:
                self.logger.error(f"Tool execution error for {tool_call}: {e}")
                results.append({
                    "tool_call_id": tool_call["id"],
                    "function": tool_call.get("function", "unknown"),
                    "error": str(e)
                })
        
        return results
    
    async def _execute_database_query(self, arguments: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """Execute database query tool"""
        try:
            from data_processing.database_manager import DatabaseManager
            
            db_manager = DatabaseManager()
            sql_query = arguments.get("sql_query", "")
            
            success, result = db_manager.execute_user_query(sql_query, user_role)
            
            if success:
                # Convert DataFrame to dict for JSON serialization
                if hasattr(result, 'to_dict'):
                    result_dict = result.to_dict('records')
                    return {
                        "success": True,
                        "data": result_dict,
                        "row_count": len(result_dict),
                        "columns": list(result.columns) if hasattr(result, 'columns') else []
                    }
                else:
                    return {"success": True, "data": result}
            else:
                return {"success": False, "error": result}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_vector_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector search tool"""
        try:
            from vectorstore.faiss_manager import FAISSManager
            
            faiss_manager = FAISSManager()
            query_text = arguments.get("query_text", "")
            top_k = arguments.get("top_k", 10)
            
            results = faiss_manager.query_similar_data(query_text, top_k)
            
            return {
                "success": True,
                "results": results,
                "query": query_text,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_data_visualization(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data visualization tool"""
        try:
            data_type = arguments.get("data_type", "")
            parameters = arguments.get("parameters", [])
            viz_type = arguments.get("visualization_type", "")
            
            # Generate visualization recommendations
            recommendations = []
            
            if "temperature" in parameters or "profile" in data_type.lower():
                recommendations.append({
                    "type": "line_plot",
                    "title": "Temperature vs Depth Profile",
                    "description": "Vertical temperature profile showing ocean stratification"
                })
            
            if "salinity" in parameters:
                recommendations.append({
                    "type": "scatter_plot",
                    "title": "Temperature-Salinity Diagram",
                    "description": "T-S diagram for water mass identification"
                })
            
            if "latitude" in parameters and "longitude" in parameters:
                recommendations.append({
                    "type": "map",
                    "title": "Geographic Distribution",
                    "description": "Map showing float locations and trajectories"
                })
            
            return {
                "success": True,
                "recommendations": recommendations,
                "data_type": data_type,
                "parameters": parameters
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_oceanographic_analysis(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute oceanographic analysis tool"""
        try:
            data_summary = arguments.get("data_summary", "")
            analysis_type = arguments.get("analysis_type", "general")
            context = arguments.get("context", "")
            
            # Generate oceanographic insights
            analysis_prompt = f"""Provide oceanographic analysis for this data:
            
            Data Summary: {data_summary}
            Analysis Type: {analysis_type}
            Context: {context}
            
            Provide insights about:
            1. Oceanographic significance
            2. Physical processes involved
            3. Data quality considerations
            4. Seasonal or regional patterns
            5. Implications for ocean research
            
            Respond in JSON format."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=1000,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            return {
                "success": True,
                "analysis": analysis,
                "data_summary": data_summary,
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_role_constraints(self, user_role: str) -> Dict[str, Any]:
        """Get constraints based on user role"""
        role_limits = settings.QUERY_LIMITS.get(user_role, settings.QUERY_LIMITS['Viewer'])
        
        return {
            "max_rows": role_limits.get("max_rows"),
            "allowed_tables": role_limits.get("allowed_tables", []),
            "restricted_columns": role_limits.get("restricted_columns", []),
            "can_export": settings.ROLE_PERMISSIONS.get(user_role, {}).get("export_data", False),
            "can_upload": settings.ROLE_PERMISSIONS.get(user_role, {}).get("file_upload", False),
            "complex_queries": settings.ROLE_PERMISSIONS.get(user_role, {}).get("complex_queries", False)
        }
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """Get MCP client status"""
        return {
            "protocol_version": self.protocol_version,
            "model": self.model,
            "available_tools": list(self.tools.keys()),
            "max_context_length": self.max_context_length,
            "status": "active"
        }
