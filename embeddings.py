"""
Embedding Manager
Handles initialization and execution of the SentenceTransformer model.
FIXED: Type error using Optional for attribute initialization.
"""
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional # CRITICAL: Import Optional

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages the loading and usage of the SentenceTransformer model.
    """
    
    # CRITICAL FIX: Annotate the attribute at the class level to allow None initialization
    model: Optional[SentenceTransformer] = None 
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer model.
        """
        self.model_name = model_name
        # The assignment happens here, resolving the error
        self.model = None 
        self._load_model()

    def _load_model(self):
        """Loads the SentenceTransformer model."""
        try:
            logger.info(f"⏳ Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generates a single embedding vector for a given text query.
        """
        if self.model is None:
            raise RuntimeError("Embedding model is not initialized.")
            
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generates embedding vectors for a list of texts.
        """
        if self.model is None:
            raise RuntimeError("Embedding model is not initialized.")
            
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

if __name__ == '__main__':
    # Example usage for direct testing
    logging.basicConfig(level=logging.INFO)
    try:
        manager = EmbeddingManager()
        test_query = "Accident near a major intersection with poor lighting."
        embedding = manager.embed_single(test_query)
        print(f"Test Query: {test_query}")
        print(f"Embedding shape: {embedding.shape}")
        print("✅ Embedding test complete.")
    except Exception as e:
        print(f"Embedding test failed: {e}")