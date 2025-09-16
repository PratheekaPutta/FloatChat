import streamlit as st
import pandas as pd
from datetime import datetime
import json
from rag.llm_orchestrator import LLMOrchestrator
from rag.mcp_client import MCPClient
from auth.authentication import check_permission
from frontend.visualizations import create_quick_visualization
import asyncio

def render_chat_interface(current_user):
    """Render the main chat interface for ARGO data queries"""
    
    st.title("🌊 ARGO AI Assistant")
    st.markdown("Ask questions about oceanographic data in natural language")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'llm_orchestrator' not in st.session_state:
        st.session_state.llm_orchestrator = LLMOrchestrator()
    
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = MCPClient()
    
    # Display user info and capabilities
    with st.sidebar:
        st.markdown("### Your Access Level")
        st.write(f"**Role:** {current_user['role']}")
        
        # Show role capabilities
        permissions = check_permission(current_user, 'data_access')
        if current_user['role'] == 'Admin':
            st.success("✅ Full data access")
            st.success("✅ Complex queries")
            st.success("✅ Data export")
            st.success("✅ File upload")
        elif current_user['role'] == 'Researcher':
            st.success("✅ Research data access")
            st.success("✅ Complex queries")
            st.success("✅ Data export")
            st.warning("❌ File upload restricted")
        else:  # Viewer
            st.info("ℹ️ Basic data access")
            st.warning("❌ Complex queries restricted")
            st.warning("❌ Data export restricted")
            st.warning("❌ File upload restricted")
        
        st.markdown("---")
        
        # Query examples based on role
        st.markdown("### Example Queries")
        
        if current_user['role'] in ['Admin', 'Researcher']:
            examples = [
                "Show temperature profiles near the equator in 2023",
                "Compare salinity in the Arabian Sea vs Bay of Bengal",
                "Find ARGO floats with BGC data in the Indian Ocean",
                "Plot oxygen levels at 1000m depth over time",
                "What are the seasonal temperature variations at 30°N?"
            ]
        else:
            examples = [
                "Show recent ARGO float locations",
                "What is the average temperature at 100m depth?",
                "Explain how ARGO floats work",
                "Show basic temperature profiles",
                "Where are ARGO floats currently active?"
            ]
        
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                st.session_state.pending_query = example
    
    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    render_assistant_message(message, current_user)
                else:
                    st.write(message["content"])
    
    # Handle pending query from examples
    if hasattr(st.session_state, 'pending_query'):
        query = st.session_state.pending_query
        del st.session_state.pending_query
        process_user_query(query, current_user)
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask about ARGO oceanographic data..."):
        process_user_query(prompt, current_user)

def process_user_query(query, current_user):
    """Process user query and generate response"""
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    
    # Process query and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            
            # Prepare context
            context = {
                "conversation_history": st.session_state.messages[-6:],  # Last 6 messages
                "user_preferences": {
                    "role": current_user['role'],
                    "visualization_preference": "plotly"
                }
            }
            
            # Process using MCP orchestrator
            try:
                # Use asyncio to run MCP client
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                mcp_result = loop.run_until_complete(
                    st.session_state.mcp_client.process_request(
                        query, context, current_user['role']
                    )
                )
                
                if mcp_result["success"]:
                    response_data = mcp_result["response"]
                    
                    # Parse LLM response
                    try:
                        if response_data.get("content"):
                            llm_response = json.loads(response_data["content"])
                        else:
                            llm_response = {"text": "I apologize, but I couldn't process your query properly."}
                    except json.JSONDecodeError:
                        llm_response = {"text": response_data.get("content", "Response parsing error")}
                    
                    # Display response
                    assistant_message = {
                        "role": "assistant",
                        "content": llm_response.get("text", "No response generated"),
                        "data": None,
                        "sql_query": None,
                        "suggestions": llm_response.get("suggestions", []),
                        "visualization_hints": llm_response.get("visualization_hints", []),
                        "tool_results": response_data.get("tool_results", [])
                    }
                    
                    # Process tool results
                    for tool_result in response_data.get("tool_results", []):
                        if tool_result.get("function") == "database_query":
                            result = tool_result.get("result", {})
                            if result.get("success") and result.get("data"):
                                assistant_message["data"] = pd.DataFrame(result["data"])
                                assistant_message["sql_query"] = "Database query executed"
                    
                    render_assistant_message(assistant_message, current_user)
                    st.session_state.messages.append(assistant_message)
                
                else:
                    error_message = {
                        "role": "assistant",
                        "content": f"I encountered an error: {mcp_result.get('error', 'Unknown error')}",
                        "error": True
                    }
                    
                    st.error(error_message["content"])
                    st.session_state.messages.append(error_message)
                
            except Exception as e:
                error_message = {
                    "role": "assistant",
                    "content": f"I apologize, but I encountered a technical error: {str(e)}",
                    "error": True
                }
                
                st.error(error_message["content"])
                st.session_state.messages.append(error_message)

def render_assistant_message(message, current_user):
    """Render assistant message with data and visualizations"""
    
    # Main response text
    st.write(message["content"])
    
    # Show explanation if available
    if message.get("explanation"):
        with st.expander("Technical Explanation"):
            st.write(message["explanation"])
    
    # Display data if available
    if message.get("data") is not None:
        data = message["data"]
        
        st.markdown("### Query Results")
        
        # Data summary
        if hasattr(data, 'shape'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                if hasattr(data, 'memory_usage'):
                    memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                    st.metric("Memory", f"{memory_mb:.1f} MB")
        
        # Display data table
        if not data.empty:
            # Show data with pagination for large datasets
            if len(data) > 100:
                st.warning(f"Large dataset ({len(data)} rows). Showing first 100 rows.")
                st.dataframe(data.head(100), use_container_width=True)
                
                if check_permission(current_user, 'export_data'):
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download full dataset as CSV",
                        data=csv,
                        file_name=f"argo_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.dataframe(data, use_container_width=True)
                
                if check_permission(current_user, 'export_data'):
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"argo_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Quick visualization options
        if not data.empty and len(data.columns) >= 2:
            st.markdown("### Quick Visualization")
            
            viz_options = []
            columns = list(data.columns)
            
            # Suggest visualizations based on column names
            if any(col in ['latitude', 'longitude'] for col in columns):
                viz_options.append("Map View")
            
            if any(col in ['temperature', 'salinity', 'pressure', 'depth'] for col in columns):
                viz_options.append("Profile Plot")
            
            if any(col in ['juld', 'date', 'time'] for col in columns):
                viz_options.append("Time Series")
            
            viz_options.extend(["Scatter Plot", "Histogram", "Box Plot"])
            
            selected_viz = st.selectbox("Choose visualization:", ["None"] + viz_options)
            
            if selected_viz != "None":
                try:
                    fig = create_quick_visualization(data, selected_viz, current_user)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization error: {e}")
    
    # Show SQL query if available
    if message.get("sql_query") and current_user['role'] in ['Admin', 'Researcher']:
        with st.expander("SQL Query"):
            st.code(message["sql_query"], language="sql")
    
    # Show suggestions
    if message.get("suggestions") and len(message["suggestions"]) > 0:
        st.markdown("### Follow-up Questions")
        cols = st.columns(min(len(message["suggestions"]), 3))
        
        for i, suggestion in enumerate(message["suggestions"][:3]):
            with cols[i % 3]:
                if st.button(suggestion, key=f"suggestion_{hash(suggestion)}_{len(st.session_state.messages)}"):
                    st.session_state.pending_query = suggestion
                    st.rerun()
    
    # Show visualization hints
    if message.get("visualization_hints"):
        with st.expander("Visualization Suggestions"):
            for hint in message["visualization_hints"]:
                st.write(f"• {hint}")

def clear_chat_history():
    """Clear chat history"""
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Add chat controls
def render_chat_controls():
    """Render chat control buttons"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear History"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Export Chat"):
            if st.session_state.messages:
                chat_export = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="Download Chat JSON",
                    data=json.dumps(chat_export, indent=2, default=str),
                    file_name=f"argo_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col3:
        # Show message count
        st.write(f"Messages: {len(st.session_state.messages)}")
