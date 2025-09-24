"""
Embedding Model Manager for efficient model loading and caching.
Prevents runtime downloads and provides fast model access.
"""

import os
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModelManager:
    """Singleton manager for embedding models to avoid repeated loading."""
    
    _instance: Optional['EmbeddingModelManager'] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __new__(cls) -> 'EmbeddingModelManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model if not already loaded."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"✅ Model loaded successfully (dimension: {self._model.get_sentence_embedding_dimension()})")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts) -> list:
        """Encode texts to embeddings."""
        return self.model.encode(texts)
    
    def encode_single(self, text: str) -> list:
        """Encode a single text to embedding."""
        return self.model.encode(text).tolist()
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if self._model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self._model_name,
            "dimension": self._model.get_sentence_embedding_dimension(),
            "max_seq_length": self._model.max_seq_length
        }


# Global instance for easy access
embedding_manager = EmbeddingModelManager()
