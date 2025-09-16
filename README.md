# AI-Powered ARGO Oceanographic Data Analysis System

This application provides an AI-powered conversational interface for analyzing ARGO oceanographic data with role-based access control and natural language querying capabilities.

## Features

- **Role-based Authentication**: Admin, Researcher, and Viewer roles with different access levels
- **NetCDF Data Processing**: Automated ingestion and conversion of ARGO NetCDF files to PostgreSQL
- **Vector Search**: Sentence transformer embeddings with FAISS vector database
- **RAG Pipeline**: LangChain-powered natural language to SQL query translation
- **Interactive Visualizations**: Geospatial plots, depth-time profiles, and trajectory maps
- **Chat Interface**: Conversational data querying with context awareness
- **Admin Dashboard**: User management and role assignment interface

## Technology Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL + FAISS Vector Store
- **ML/AI**: OpenAI GPT-5, Sentence Transformers, LangChain
- **Visualization**: Plotly, Folium
- **Authentication**: JWT with bcrypt hashing

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd argo-ai-system
pip install -r requirements.txt
```
2. **Run streamlit app**
```bash
streamlit run app.py
```
