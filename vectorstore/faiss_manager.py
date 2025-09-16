try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

import numpy as np
import pickle
import os
import streamlit as st
from config.settings import settings
from config.database import execute_query
import json

class FAISSManager:
    """Manage FAISS vector database for ARGO data retrieval"""
    
    def __init__(self):
        self.vector_db_path = settings.VECTOR_DB_PATH
        self.index = None
        self.metadata_store = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        if not FAISS_AVAILABLE:
            st.warning("FAISS not available. Vector search functionality disabled.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Try to load existing index
        self.load_index()
    
    def initialize_index(self, dimension=None):
        """Initialize a new FAISS index"""
        if dimension:
            self.dimension = dimension
        
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        
        st.info(f"Initialized new FAISS index with dimension {self.dimension}")
    
    def add_vectors(self, embeddings, metadata_list):
        """Add vectors and metadata to the index"""
        try:
            if self.index is None:
                self.initialize_index(embeddings.shape[1])
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.metadata_store.extend(metadata_list)
            
            return True
            
        except Exception as e:
            st.error(f"Error adding vectors to index: {e}")
            return False
    
    def search(self, query_embedding, k=10):
        """Search for similar vectors"""
        try:
            if self.index is None or self.index.ntotal == 0:
                return [], []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Get metadata for results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata_store):
                    result = {
                        'metadata': self.metadata_store[idx],
                        'score': float(scores[0][i]),
                        'index': int(idx)
                    }
                    results.append(result)
            
            return results, scores[0]
            
        except Exception as e:
            st.error(f"Error searching index: {e}")
            return [], []
    
    def build_index_from_database(self):
        """Build FAISS index from existing database data"""
        try:
            st.info("Building FAISS index from database...")
            
            # Get metadata summaries
            metadata_query = """
            SELECT 
                id, platform_number, project_name, pi_name, platform_type,
                date_creation, lat_min, lat_max, lon_min, lon_max,
                data_centre, institution, n_profiles
            FROM argo_metadata
            ORDER BY id
            """
            
            metadata_results = execute_query(metadata_query)
            
            if not metadata_results:
                st.warning("No metadata found in database")
                return False
            
            # Create summaries and generate embeddings
            summaries = []
            metadata_list = []
            
            for row in metadata_results:
                summary = self.embedding_manager.create_metadata_summary(dict(row))
                summaries.append(summary)
                
                metadata_list.append({
                    'type': 'metadata',
                    'id': row['id'],
                    'platform_number': row['platform_number'],
                    'summary': summary,
                    'data': dict(row)
                })
            
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings(summaries)
            
            if embeddings is None:
                st.error("Failed to generate embeddings")
                return False
            
            # Initialize new index
            self.initialize_index(embeddings.shape[1])
            
            # Add to index
            if self.add_vectors(embeddings, metadata_list):
                st.success(f"Added {len(summaries)} metadata entries to FAISS index")
                
                # Also add profile summaries
                self.add_profile_summaries()
                
                # Save index
                self.save_index()
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error building index from database: {e}")
            return False
    
    def add_profile_summaries(self, limit=1000):
        """Add profile summaries to the index"""
        try:
            # Get recent profiles with metadata
            profile_query = """
            SELECT 
                p.id, p.cycle_number, p.latitude, p.longitude, p.juld,
                p.n_levels, p.data_mode, p.direction,
                m.platform_number, m.project_name
            FROM argo_profiles p
            JOIN argo_metadata m ON p.metadata_id = m.id
            ORDER BY p.juld DESC
            LIMIT %s
            """
            
            profile_results = execute_query(profile_query, (limit,))
            
            if not profile_results:
                return
            
            # Create summaries
            summaries = []
            metadata_list = []
            
            for row in profile_results:
                # Add parameter availability info
                row_dict = dict(row)
                
                # Check for parameter availability
                param_query = """
                SELECT 
                    COUNT(CASE WHEN temperature IS NOT NULL THEN 1 END) > 0 as has_temperature,
                    COUNT(CASE WHEN salinity IS NOT NULL THEN 1 END) > 0 as has_salinity,
                    COUNT(CASE WHEN doxy IS NOT NULL THEN 1 END) > 0 as has_oxygen,
                    COUNT(CASE WHEN chla IS NOT NULL OR bbp700 IS NOT NULL OR 
                               ph_in_situ_total IS NOT NULL OR nitrate IS NOT NULL THEN 1 END) > 0 as has_bgc
                FROM argo_measurements
                WHERE profile_id = %s
                """
                
                param_result = execute_query(param_query, (row['id'],))
                if param_result:
                    row_dict.update(dict(param_result[0]))
                
                summary = self.embedding_manager.create_profile_summary(row_dict)
                summaries.append(summary)
                
                metadata_list.append({
                    'type': 'profile',
                    'id': row['id'],
                    'platform_number': row['platform_number'],
                    'summary': summary,
                    'data': row_dict
                })
            
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings(summaries)
            
            if embeddings is not None:
                self.add_vectors(embeddings, metadata_list)
                st.success(f"Added {len(summaries)} profile summaries to FAISS index")
            
        except Exception as e:
            st.error(f"Error adding profile summaries: {e}")
    
    def query_similar_data(self, query_text, k=10, similarity_threshold=0.3):
        """Query for similar data using natural language"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query_text])
            
            if query_embedding is None:
                return []
            
            # Search index
            results, scores = self.search(query_embedding[0], k)
            
            # Filter by similarity threshold
            filtered_results = []
            for result in results:
                if result['score'] >= similarity_threshold:
                    result['similarity'] = result['score']
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Error querying similar data: {e}")
            return []
    
    def get_context_for_query(self, query_text, max_results=5):
        """Get relevant context for RAG query processing"""
        try:
            # Search for similar data
            similar_data = self.query_similar_data(query_text, k=max_results)
            
            if not similar_data:
                return "No relevant ARGO data found for this query."
            
            # Build context
            context_parts = []
            context_parts.append("Relevant ARGO Data Context:")
            
            for i, item in enumerate(similar_data):
                metadata = item['metadata']
                context_parts.append(f"\n{i+1}. {metadata['summary']}")
                
                if metadata['type'] == 'metadata':
                    data = metadata['data']
                    context_parts.append(f"   Platform: {data.get('platform_number', 'Unknown')}")
                    context_parts.append(f"   Project: {data.get('project_name', 'Unknown')}")
                    context_parts.append(f"   Profiles: {data.get('n_profiles', 0)}")
                
                elif metadata['type'] == 'profile':
                    data = metadata['data']
                    context_parts.append(f"   Profile ID: {data.get('id')}")
                    context_parts.append(f"   Location: {data.get('latitude', 'N/A'):.2f}°N, {data.get('longitude', 'N/A'):.2f}°E")
                    context_parts.append(f"   Date: {data.get('juld', 'Unknown')}")
                
                context_parts.append(f"   Relevance Score: {item['similarity']:.3f}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            st.error(f"Error getting context for query: {e}")
            return "Error retrieving context data."
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                # Save FAISS index
                index_path = os.path.join(self.vector_db_path, "argo_index.faiss")
                faiss.write_index(self.index, index_path)
                
                # Save metadata
                metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata_store, f)
                
                # Save index info
                info_path = os.path.join(self.vector_db_path, "index_info.json")
                info = {
                    'dimension': self.dimension,
                    'total_vectors': self.index.ntotal,
                    'created_at': str(pd.Timestamp.now())
                }
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
                
                st.success("FAISS index saved successfully")
                return True
            
        except Exception as e:
            st.error(f"Error saving index: {e}")
            return False
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            index_path = os.path.join(self.vector_db_path, "argo_index.faiss")
            metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
            info_path = os.path.join(self.vector_db_path, "index_info.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                
                # Load index info
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        self.dimension = info.get('dimension', 384)
                
                st.success(f"Loaded FAISS index with {self.index.ntotal} vectors")
                return True
            
        except Exception as e:
            st.warning(f"Could not load existing index: {e}")
            return False
    
    def get_index_stats(self):
        """Get statistics about the FAISS index"""
        if self.index is None:
            return {
                'total_vectors': 0,
                'dimension': 0,
                'metadata_count': 0,
                'index_size_mb': 0
            }
        
        # Calculate approximate index size
        index_size_bytes = self.index.ntotal * self.dimension * 4  # 4 bytes per float32
        index_size_mb = index_size_bytes / (1024 * 1024)
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata_store),
            'index_size_mb': round(index_size_mb, 2)
        }
    
    def rebuild_index(self):
        """Rebuild the entire FAISS index from database"""
        try:
            # Clear existing index
            self.index = None
            self.metadata_store = []
            
            # Rebuild from database
            success = self.build_index_from_database()
            
            if success:
                st.success("FAISS index rebuilt successfully")
            else:
                st.error("Failed to rebuild FAISS index")
            
            return success
            
        except Exception as e:
            st.error(f"Error rebuilding index: {e}")
            return False
