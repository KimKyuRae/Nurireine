"""
Embedding Function Module

Provides GGUF-based embedding generation for ChromaDB using llama-cpp-python.
Primarily targets Korean embedding models like KURE-v1.

Features:
- Automatic model downloading from HuggingFace Hub
- GPU acceleration when available
- LRU caching for frequently queried texts
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from chromadb import Documents, EmbeddingFunction, Embeddings
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .. import config

logger = logging.getLogger(__name__)

# Cache size for embeddings (number of unique texts to cache)
EMBEDDING_CACHE_SIZE = 256


class GGUFEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB embedding function using a GGUF model via llama-cpp-python.
    
    This class handles:
    - Automatic model downloading from HuggingFace Hub
    - GPU acceleration when available
    - Embedding generation for text documents
    - LRU caching for frequently queried texts (#12)
    """
    
    def __init__(
        self, 
        repo_id: str = None, 
        filename: str = None,
        model_dir: Path = None
    ):
        """
        Initialize the embedding function.
        
        Args:
            repo_id: HuggingFace repository ID (default from config)
            filename: Model filename (default from config)
            model_dir: Directory to store models (default from config)
        """
        self.repo_id = repo_id or config.embedding.model_repo
        self.filename = filename or config.embedding.model_filename
        self.model_dir = model_dir or config.MODELS_DIR
        
        self.model_path = self._ensure_model()
        
        logger.info(f"Loading embedding model from {self.model_path}...")
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                embedding=True,
                verbose=False,    # Enable verbose for debug usage
                n_gpu_layers=0   # Force CPU for stability (small model)
            )
            logger.info("Embedding model ready.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _ensure_model(self) -> Path:
        """
        Ensure the model file exists locally, downloading if necessary.
        
        Returns:
            Path to the local model file
        """
        local_path = self.model_dir / self.filename
        
        if local_path.exists():
            return local_path
        
        logger.info(f"Downloading embedding model {self.filename} from {self.repo_id}...")
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=str(self.model_dir)
            )
            return Path(path)
        except Exception as e:
            logger.error(f"Embedding model download failed: {type(e).__name__}: {e}")
            raise
    
    def _create_single_embedding(self, text: str) -> Tuple[float, ...]:
        """
        Create embedding for a single text (cached).
        
        Returns tuple instead of list for hashability with lru_cache.
        """
        response = self.llm.create_embedding(text)
        return tuple(response['data'][0]['embedding'])
    
    @lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
    def _cached_embedding(self, text: str) -> Tuple[float, ...]:
        """
        Get cached embedding for text.
        
        LRU cache ensures frequently queried texts (like base lore)
        don't require re-computation.
        """
        return self._create_single_embedding(text)
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents.
        
        Args:
            input: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings: List[List[float]] = []
        
        for text in input:
            try:
                # Use cached embedding
                cached = self._cached_embedding(text)
                embeddings.append(list(cached))
            except Exception as e:
                logger.error(f"Error creating embedding for text '{text[:20]}...': {e}")
                raise
        
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cached_embedding.cache_clear()
        logger.info("Embedding cache cleared.")
    
    def cache_info(self) -> str:
        """Get cache statistics."""
        info = self._cached_embedding.cache_info()
        return f"Hits: {info.hits}, Misses: {info.misses}, Size: {info.currsize}/{info.maxsize}"
