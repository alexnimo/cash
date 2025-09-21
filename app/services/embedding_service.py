"""
Embedding service for generating text embeddings using various providers.
"""
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
import numpy as np
from threading import Lock
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using various providers."""
    
    _instance = None
    _lock = Lock()
    
    def __init__(self, provider: str = "huggingface", model: str = "nomic-ai/nomic-embed-text-v2-moe", **kwargs):
        """Initialize the embedding service.
        
        Args:
            provider: The embedding provider ('huggingface', 'openai', etc.)
            model: The model name/identifier
            **kwargs: Additional configuration options
        """
        self.provider = provider.lower()
        self.model = model
        self.dimension = kwargs.get('dimension', 768)
        self.metric = kwargs.get('metric', 'cosine')
        self.device = kwargs.get('device', 'cpu')
        self._client = None
        self._initialized = False
        
        logger.info(f"Initializing EmbeddingService with provider: {provider}, model: {model}")
        
    @classmethod
    def get_instance(cls, **kwargs) -> 'EmbeddingService':
        """Get or create singleton instance of EmbeddingService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Try to get config from unified config if available
                    try:
                        from app.core.unified_config import ConfigManager
                        config = ConfigManager()
                        embedding_config = config.get('agents', 'embedding') or {}
                        
                        provider = embedding_config.get('provider', 'huggingface')
                        model = embedding_config.get('model', 'nomic-ai/nomic-embed-text-v2-moe')
                        dimension = embedding_config.get('dimension', 768)
                        metric = embedding_config.get('metric', 'cosine')
                        
                        cls._instance = cls(
                            provider=provider,
                            model=model,
                            dimension=dimension,
                            metric=metric,
                            **kwargs
                        )
                    except Exception as e:
                        logger.warning(f"Could not load config, using defaults: {e}")
                        cls._instance = cls(**kwargs)
                        
        return cls._instance
    
    def _initialize_client(self):
        """Initialize the embedding client based on provider."""
        if self._initialized:
            return
            
        try:
            if self.provider == "huggingface":
                self._initialize_huggingface_client()
            elif self.provider == "openai":
                self._initialize_openai_client()
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
                
            self._initialized = True
            logger.info(f"Successfully initialized {self.provider} embedding client")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            # Fall back to a simple implementation
            self._initialize_fallback_client()
            
    def _initialize_huggingface_client(self):
        """Initialize HuggingFace sentence transformers client."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check if model exists locally or needs to be downloaded
            model_path = self.model
            
            logger.info(f"Loading HuggingFace model: {model_path}")
            self._client = SentenceTransformer(model_path, device=self.device, trust_remote_code=True)
            
            # Update dimension based on actual model
            if hasattr(self._client, 'get_sentence_embedding_dimension'):
                self.dimension = self._client.get_sentence_embedding_dimension()
            elif hasattr(self._client, 'encode'):
                # Test with a sample text to get dimension
                sample_embedding = self._client.encode(["test"], show_progress_bar=False)
                if len(sample_embedding) > 0:
                    self.dimension = len(sample_embedding[0])
                    
            logger.info(f"HuggingFace model loaded with dimension: {self.dimension}")
            
        except ImportError:
            logger.error("sentence-transformers package not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {self.model}: {e}")
            raise
    
    def _initialize_openai_client(self):
        """Initialize OpenAI embedding client."""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            self._client = openai.OpenAI(api_key=api_key)
            
            # Set dimension based on model
            if "ada" in self.model.lower():
                self.dimension = 1536
            elif "3-small" in self.model.lower():
                self.dimension = 1536
            elif "3-large" in self.model.lower():
                self.dimension = 3072
                
            logger.info(f"OpenAI client initialized with model: {self.model}, dimension: {self.dimension}")
            
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _initialize_fallback_client(self):
        """Initialize a simple fallback client that returns random embeddings."""
        logger.warning("Using fallback embedding client with random embeddings")
        self._client = "fallback"
        self._initialized = True
    
    def embed_texts(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional options for embedding generation
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if not self._initialized:
            self._initialize_client()
            
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return []
            
        try:
            if self.provider == "huggingface":
                return self._embed_huggingface(texts, **kwargs)
            elif self.provider == "openai":
                return self._embed_openai(texts, **kwargs)
            else:
                return self._embed_fallback(texts, **kwargs)
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return fallback embeddings
            return self._embed_fallback(texts, **kwargs)
    
    def _embed_huggingface(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using HuggingFace sentence transformers."""
        try:
            show_progress = kwargs.get('show_progress_bar', False)
            normalize_embeddings = kwargs.get('normalize_embeddings', True)
            
            embeddings = self._client.encode(
                texts,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize_embeddings,
                convert_to_tensor=False  # Return numpy arrays
            )
            
            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
                
            return embeddings
            
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def _embed_openai(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def _embed_fallback(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate random embeddings as fallback."""
        embeddings = []
        for text in texts:
            # Generate deterministic "embeddings" based on text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to numbers and normalize
            embedding = []
            for i in range(0, min(len(text_hash), self.dimension * 2), 2):
                hex_pair = text_hash[i:i+2]
                value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
                embedding.append(value)
            
            # Pad or truncate to target dimension
            while len(embedding) < self.dimension:
                embedding.append(0.0)
            embedding = embedding[:self.dimension]
            
            # Normalize vector
            magnitude = sum(x**2 for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
                
            embeddings.append(embedding)
            
        logger.warning(f"Generated {len(embeddings)} fallback embeddings")
        return embeddings
    
    async def embed_texts_async(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        # For now, run in thread pool since most embedding libraries are sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts, **kwargs)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self._initialized:
            self._initialize_client()
        return self.dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'provider': self.provider,
            'model': self.model,
            'dimension': self.dimension,
            'metric': self.metric,
            'initialized': self._initialized
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self._client, 'cleanup'):
            self._client.cleanup()
        self._initialized = False
        logger.info("EmbeddingService cleaned up")

# Convenience functions
def get_embedding_service(**kwargs) -> EmbeddingService:
    """Get the singleton embedding service instance."""
    return EmbeddingService.get_instance(**kwargs)

def embed_text(text: str, **kwargs) -> List[float]:
    """Generate embedding for a single text."""
    service = get_embedding_service()
    embeddings = service.embed_texts(text, **kwargs)
    return embeddings[0] if embeddings else []

def embed_texts(texts: List[str], **kwargs) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    service = get_embedding_service()
    return service.embed_texts(texts, **kwargs)
