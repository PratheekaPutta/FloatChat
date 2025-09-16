import numpy as np
import pandas as pd
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
import streamlit as st
from config.settings import settings
import json
import os

class EmbeddingManager:
    """Manage text embeddings for ARGO metadata"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.warning("Sentence Transformers not available. Vector embeddings functionality disabled.")
            return
            
        try:
            self.model = SentenceTransformer(self.model_name)
            st.success(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            # Fallback to a smaller model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                st.warning("Using fallback embedding model: all-MiniLM-L6-v2")
            except Exception as e2:
                st.error(f"Failed to load fallback model: {e2}")
    
    def create_metadata_summary(self, metadata_row):
        """Create a text summary from metadata for embedding"""
        summary_parts = []
        
        # Platform information
        if metadata_row.get('platform_number'):
            summary_parts.append(f"ARGO Float {metadata_row['platform_number']}")
        
        if metadata_row.get('project_name'):
            summary_parts.append(f"Project: {metadata_row['project_name']}")
        
        if metadata_row.get('platform_type'):
            summary_parts.append(f"Type: {metadata_row['platform_type']}")
        
        # Temporal information
        if metadata_row.get('date_creation'):
            summary_parts.append(f"Deployed: {metadata_row['date_creation']}")
        
        # Spatial information
        if all(k in metadata_row for k in ['lat_min', 'lat_max', 'lon_min', 'lon_max']):
            summary_parts.append(
                f"Location: {metadata_row['lat_min']:.2f}°N to {metadata_row['lat_max']:.2f}°N, "
                f"{metadata_row['lon_min']:.2f}°E to {metadata_row['lon_max']:.2f}°E"
            )
        
        # Data information
        if metadata_row.get('n_profiles'):
            summary_parts.append(f"Profiles: {metadata_row['n_profiles']}")
        
        if metadata_row.get('pi_name'):
            summary_parts.append(f"PI: {metadata_row['pi_name']}")
        
        if metadata_row.get('institution'):
            summary_parts.append(f"Institution: {metadata_row['institution']}")
        
        return " | ".join(summary_parts)
    
    def create_profile_summary(self, profile_row):
        """Create a text summary from profile data for embedding"""
        summary_parts = []
        
        # Basic profile info
        if profile_row.get('platform_number'):
            summary_parts.append(f"Float {profile_row['platform_number']}")
        
        if profile_row.get('cycle_number'):
            summary_parts.append(f"Cycle {profile_row['cycle_number']}")
        
        # Location and time
        if profile_row.get('latitude') and profile_row.get('longitude'):
            summary_parts.append(f"Location: {profile_row['latitude']:.2f}°N, {profile_row['longitude']:.2f}°E")
        
        if profile_row.get('juld'):
            summary_parts.append(f"Date: {profile_row['juld']}")
        
        # Measurement info
        if profile_row.get('n_levels'):
            summary_parts.append(f"Depth levels: {profile_row['n_levels']}")
        
        if profile_row.get('data_mode'):
            summary_parts.append(f"Mode: {profile_row['data_mode']}")
        
        if profile_row.get('direction'):
            summary_parts.append(f"Direction: {profile_row['direction']}")
        
        # Add parameter availability info
        param_info = []
        if profile_row.get('has_temperature'):
            param_info.append("Temperature")
        if profile_row.get('has_salinity'):
            param_info.append("Salinity")
        if profile_row.get('has_oxygen'):
            param_info.append("Oxygen")
        if profile_row.get('has_bgc'):
            param_info.append("BGC parameters")
        
        if param_info:
            summary_parts.append(f"Parameters: {', '.join(param_info)}")
        
        return " | ".join(summary_parts)
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts"""
        if not self.model:
            st.error("Embedding model not loaded")
            return None
        
        try:
            # Convert to list if single string
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            return embeddings
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None
    
    def compute_similarity(self, query_embedding, document_embeddings):
        """Compute cosine similarity between query and document embeddings"""
        try:
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            
            return similarities
            
        except Exception as e:
            st.error(f"Error computing similarity: {e}")
            return None
    
    def find_similar_content(self, query_text, embeddings_data, top_k=10):
        """Find most similar content to query"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query_text])
            if query_embedding is None:
                return []
            
            query_embedding = query_embedding[0]  # Get first (and only) embedding
            
            # Extract embeddings and metadata
            document_embeddings = np.array([item['embedding'] for item in embeddings_data])
            
            # Compute similarities
            similarities = self.compute_similarity(query_embedding, document_embeddings)
            
            if similarities is None:
                return []
            
            # Get top-k most similar items
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'metadata': embeddings_data[idx]['metadata'],
                    'summary': embeddings_data[idx]['summary'],
                    'similarity': float(similarities[idx]),
                    'id': embeddings_data[idx].get('id')
                })
            
            return results
            
        except Exception as e:
            st.error(f"Error finding similar content: {e}")
            return []
    
    def create_enhanced_context(self, query, similar_items, max_context_length=2000):
        """Create enhanced context from similar items for RAG"""
        context_parts = []
        
        context_parts.append(f"User Query: {query}")
        context_parts.append("\nRelevant ARGO Data Context:")
        
        for i, item in enumerate(similar_items):
            if len(" ".join(context_parts)) > max_context_length:
                break
            
            context_parts.append(f"\n{i+1}. {item['summary']} (Similarity: {item['similarity']:.3f})")
            
            # Add specific metadata if available
            if 'metadata' in item and item['metadata']:
                metadata = item['metadata']
                if isinstance(metadata, dict):
                    key_info = []
                    for key in ['platform_number', 'latitude', 'longitude', 'juld', 'n_profiles']:
                        if key in metadata and metadata[key] is not None:
                            key_info.append(f"{key}: {metadata[key]}")
                    
                    if key_info:
                        context_parts.append(f"   Details: {', '.join(key_info)}")
        
        return "\n".join(context_parts)
    
    def extract_keywords(self, text):
        """Extract relevant keywords from text for search enhancement"""
        # Simple keyword extraction - in production, use more sophisticated NLP
        keywords = set()
        
        # Common oceanographic terms
        ocean_terms = [
            'temperature', 'salinity', 'oxygen', 'pressure', 'depth',
            'float', 'profile', 'argo', 'bgc', 'trajectory',
            'atlantic', 'pacific', 'indian', 'ocean', 'sea',
            'equator', 'tropical', 'subtropical', 'polar',
            'chlorophyll', 'nitrate', 'ph', 'dissolved', 'conductivity'
        ]
        
        text_lower = text.lower()
        for term in ocean_terms:
            if term in text_lower:
                keywords.add(term)
        
        # Extract potential coordinates
        import re
        coord_pattern = r'[-+]?\d*\.?\d+[°]?[NS]?[EW]?'
        coords = re.findall(coord_pattern, text)
        keywords.update(coords)
        
        # Extract years/dates
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        keywords.update(years)
        
        return list(keywords)
    
    def save_embeddings(self, embeddings_data, filename):
        """Save embeddings to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            for item in embeddings_data:
                if isinstance(item.get('embedding'), np.ndarray):
                    item['embedding'] = item['embedding'].tolist()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save to JSON file
            with open(filename, 'w') as f:
                json.dump(embeddings_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, filename):
        """Load embeddings from file"""
        try:
            if not os.path.exists(filename):
                return []
            
            with open(filename, 'r') as f:
                embeddings_data = json.load(f)
            
            # Convert lists back to numpy arrays
            for item in embeddings_data:
                if isinstance(item.get('embedding'), list):
                    item['embedding'] = np.array(item['embedding'])
            
            return embeddings_data
            
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return []
