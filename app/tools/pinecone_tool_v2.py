from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import Document, Node
from llama_index.core import Response
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from pinecone import Pinecone, ServerlessSpec, Index
from typing import List, Dict, Any, Optional
from app.core.settings import get_settings
import numpy as np
import os
import logging
import json
import time
import hashlib
import torch

logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.DEBUG)
logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
logging.getLogger("transformers").setLevel(logging.DEBUG)

class PineconeAdvancedToolSpec(BaseTool):
    """Advanced Pinecone tool for vector operations"""
    def __init__(self):
        super().__init__()
        
        settings = get_settings()
        
        # Ensure agent-specific configurations are present
        if not settings.agents:
            raise ValueError("Agent configuration (settings.agents) not found in settings.")
        if not settings.agents.embedding:
            raise ValueError("Agent embedding configuration (settings.agents.embedding) not found in settings.")
        if not settings.agents.pinecone:
            raise ValueError("Agent Pinecone configuration (settings.agents.pinecone) not found in settings.")

        embedding_model_settings = settings.agents.embedding
        self.agent_pinecone_config = settings.agents.pinecone # Store for use in other methods
        
        # Pinecone API Key: ENV first, then global vector_store config if available
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.warning("PINECONE_API_KEY not found in environment variables, checking global vector_store.config settings...")
            if settings.vector_store and settings.vector_store.config:
                 pinecone_api_key = settings.vector_store.config.get('api_key')
            if not pinecone_api_key:
                raise ValueError("Pinecone API key not found in environment variables or global vector_store.config settings")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        self.index_name = self.agent_pinecone_config.index_name
        if not self.index_name:
            raise ValueError("Pinecone 'index_name' not found in settings.agents.pinecone")
        
        # Embedding Configuration from settings.agents.embedding
        self.dimension = embedding_model_settings.dimension
        if self.dimension is None:
            logger.warning("Embedding dimension not specified in settings.agents.embedding, defaulting to 768 for Nomic.")
            self.dimension = 768

        self.metric = embedding_model_settings.metric
        if not self.metric:
            logger.warning("Embedding 'metric' not found in settings.agents.embedding, defaulting to 'cosine'.")
            self.metric = 'cosine'
        
        model_name_to_load = embedding_model_settings.model
        if not model_name_to_load:
            raise ValueError("Embedding model name (settings.agents.embedding.model) not found.")

        logger.info(f"Initializing embedding model access: {model_name_to_load}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.debug(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")

        try:
            # Use the singleton embedding service instead of creating a new instance
            from app.services.embedding_service import EmbeddingService
            
            # Get the singleton instance
            embedding_service = EmbeddingService.get_instance()
            
            # Get the embedding model (will be initialized if not already done)
            self.embed_model = embedding_service.get_embedding_model(model_name=model_name_to_load)
            
            logger.info(f"Successfully accessed embedding model: {model_name_to_load}")
        except Exception as e:
            logger.error(f"Failed to access embedding model '{model_name_to_load}': {e}", exc_info=True)
            raise
        finally:
            logger.debug("Embedding model access attempt finished.")
        
        self._ensure_index_dimension()
        self.index = self._initialize_index()
        self.pipeline = self._initialize_pipeline()

        logger.info(f"PineconeAdvancedToolSpec initialized successfully with index: {self.index_name}")
        logger.info(f"Embedding model {model_name_to_load} loaded.")

    def get_embedding_model(self) -> BaseEmbedding:
        """Returns the initialized embedding model."""
        if not hasattr(self, 'embed_model') or self.embed_model is None:
            logger.error("Embedding model not initialized before get_embedding_model call.")
            # Potentially re-initialize or raise a more specific error
            # For now, trying to re-initialize if settings are available
            if hasattr(self, 'embedding_model_settings') and self.embedding_model_settings:
                model_name_to_load = self.embedding_model_settings.model
                try:
                    self.embed_model = HuggingFaceEmbedding(model_name=model_name_to_load, trust_remote_code=True)
                    logger.info(f"Re-initialized embedding model: {model_name_to_load}")
                except Exception as e:
                    logger.error(f"Failed to re-initialize embedding model {model_name_to_load}: {e}")
                    raise RuntimeError("Embedding model could not be initialized or re-initialized.") from e
            else:
                 raise AttributeError("Embedding model (self.embed_model) is not available and settings for re-initialization are missing.")
        return self.embed_model

    metadata = ToolMetadata(
        name="pinecone_advanced_tool",
        description="Tool for advanced Pinecone operations: index management, upserting, querying, and processing documents."
    )

    def _initialize_index(self) -> Index:
        """Initialize or get the Pinecone index."""
        try:
            if not self.index_exists(self.index_name):
                logger.info(f"Creating new index {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.agent_pinecone_config.cloud if self.agent_pinecone_config.cloud else 'aws',
                        region=self.agent_pinecone_config.region if self.agent_pinecone_config.region else 'us-west-2'
                    ),
                    deletion_protection='enabled'
                )
                while not self._is_index_ready(self.index_name):
                    time.sleep(1)
            return self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}", exc_info=True)
            raise

    def _initialize_pipeline(self) -> IngestionPipeline:
        """Initialize the ingestion pipeline"""
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        import importlib

        pinecone_vectorstore_module = importlib.import_module('llama_index.vector_stores.pinecone')
        if hasattr(pinecone_vectorstore_module, 'get_pinecone_client'):
            try:
                vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    environment=self.agent_pinecone_config.environment,
                    metadata_filters=True,
                    pinecone_kwargs={"api_key": self.pc.api_key},
                    embedding_model=self.embed_model
                )
                logger.info("Initialized PineconeVectorStore with newer method")
            except Exception as e:
                logger.warning(f"Could not initialize PineconeVectorStore with newer method: {e}")
                vector_store = PineconeVectorStore(
                    pinecone_index=self.index,
                    embedding_model=self.embed_model
                )
                logger.info("Initialized PineconeVectorStore with original method as fallback")
        else:
            vector_store = PineconeVectorStore(
                pinecone_index=self.index,
                embedding_model=self.embed_model
            )
            logger.info("Initialized PineconeVectorStore with original method")
        
        return IngestionPipeline(
            transformations=[
                SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=self.embed_model,
                ),
                self.embed_model, # Add embed_model for embedding nodes
            ],
            vector_store=vector_store,
        )

    def _ensure_index_dimension(self):
        """Ensure index exists with correct dimension"""
        try:
            target_dimension = self.dimension
            
            if not target_dimension:
                logger.error("No dimension specified in embedding config")
                raise ValueError("Target dimension for Pinecone index not specified.")
            
            if not self.index_exists(self.index_name):
                logger.info(f"Creating new index {self.index_name} because it does not exist.")
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=target_dimension,
                        metric=self.metric,
                        spec=ServerlessSpec(
                            cloud=self.agent_pinecone_config.cloud if self.agent_pinecone_config.cloud else 'aws',
                            region=self.agent_pinecone_config.region if self.agent_pinecone_config.region else 'us-west-2'
                        ),
                        deletion_protection='enabled'
                    )
                    while not self._is_index_ready(self.index_name):
                        time.sleep(1)
                    logger.info(f"Created new Pinecone index: {self.index_name} with dimension {target_dimension}")
                except Exception as create_error:
                    if "ALREADY_EXISTS" in str(create_error):
                        logger.info(f"Index {self.index_name} was created by another process, using existing index")
                    else:
                        logger.error(f"Error creating index: {str(create_error)}", exc_info=True)
                        raise
                # After creation or if it was created by another process, re-check description
                # to confirm dimension if possible, or proceed if creation was successful.

            # Index exists or was just created, now check its dimension
            index_description = self.pc.describe_index(self.index_name)
            current_dimension = None
            try:
                current_dimension = getattr(index_description, "dimension", None)
            except (AttributeError, TypeError):
                current_dimension = index_description.get("dimension") if isinstance(index_description, dict) else None
                    
            if current_dimension is None:
                for attr in ["dimension", "dimensions", "vector_dimension", "vector_dimensions"]:
                    current_dimension = getattr(index_description, attr, None)
                    if current_dimension is not None:
                        break
                
            if current_dimension is None:
                logger.warning(f"Could not determine dimension of existing index {self.index_name} after creation/check. Assuming correct if creation did not error.")
                # If creation didn't error, and we can't fetch dimension, we might have to assume it's okay.
                # Or, strict mode: raise ValueError(f"Could not determine dimension of existing index {self.index_name}")
                return # Proceed cautiously
                
            if current_dimension == target_dimension:
                logger.info(f"Using existing index {self.index_name} with correct dimension {target_dimension}")
                return
            else:
                logger.warning(f"Index dimension mismatch. Current: {current_dimension}, Target: {target_dimension}")
                logger.warning("Cannot modify existing index dimensions. Please manually delete the index if you need to change its dimension.")
                raise ValueError(f"Index {self.index_name} exists with incorrect dimension {current_dimension} (expected {target_dimension})")
        except Exception as e:
            logger.error(f"Error ensuring index dimension: {str(e)}", exc_info=True)
            raise

    def _is_index_ready(self, index_name: str) -> bool:
        """Check if the Pinecone index is ready"""
        try:
            description = self.pc.describe_index(index_name)
            try:
                return description.status.ready
            except AttributeError:
                if isinstance(description, dict):
                    return description.get("status", {}).get("ready", False)
                logger.warning(f"Could not determine ready state for index {index_name} via status.ready or dict access, assuming ready if describe_index succeeded.")
                return True # If describe_index works and we can't find specific ready status, assume it might be ready.
        except Exception as e:
            # If describe_index fails with 'not found', it's definitely not ready.
            if "not found" in str(e).lower():
                return False
            logger.warning(f"Error checking if index is ready for '{index_name}': {str(e)}. Assuming not ready.")
            return False

    def index_exists(self, index_name: str) -> bool:
        """Check if a Pinecone index exists"""
        try:
            self.pc.describe_index(index_name)
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                return False
            logger.warning(f"Error checking if index '{index_name}' exists: {e}. Assuming it does not.")
            return False

    def create_pinecone_index_tool(self, name: str, dimension: int, metric: str) -> str:
        """Tool to create a Pinecone index"""
        try:
            if self.index_exists(name):
                return f"Index {name} already exists."
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=self.agent_pinecone_config.cloud if self.agent_pinecone_config.cloud else 'aws',
                    region=self.agent_pinecone_config.region if self.agent_pinecone_config.region else 'us-west-2'
                ),
                deletion_protection='enabled'
            )
            while not self._is_index_ready(name):
                time.sleep(1)
            return f"Index {name} created successfully."
        except Exception as e:
            return f"Error creating index {name}: {str(e)}"

    def delete_pinecone_index_tool(self, name: str) -> str:
        """Tool to delete a Pinecone index"""
        try:
            if not self.index_exists(name):
                return f"Index {name} does not exist."
            self.pc.delete_index(name)
            return f"Index {name} deleted successfully."
        except Exception as e:
            return f"Error deleting index {name}: {str(e)}"

    def get_pinecone_index_stats_tool(self, name: str) -> Dict[str, Any]:
        """Tool to get stats for a Pinecone index"""
        try:
            if not self.index_exists(name):
                return {"error": f"Index {name} does not exist."}
            index = self.pc.Index(name)
            stats = index.describe_index_stats()
            try:
                return {
                    "dimension": getattr(stats, "dimension", 0),
                    "total_vector_count": getattr(stats, "total_vector_count", 0),
                    "namespaces": getattr(stats, "namespaces", {})
                }
            except AttributeError:
                if isinstance(stats, dict):
                    return {
                        "dimension": stats.get("dimension", 0),
                        "total_vector_count": stats.get("total_vector_count", 0),
                        "namespaces": stats.get("namespaces", {})
                    }
                return {
                    "dimension": 0,
                    "total_vector_count": 0,
                    "namespaces": {},
                    "warning": "Could not parse stats object properly."
                }
        except Exception as e:
            return {"error": f"Error getting stats for index {name}: {str(e)}"}

    async def process_documents(self, documents: List[Document]) -> bool:
        """Process and upsert documents using the ingestion pipeline"""
        try:
            try:
                logger.info(f"Processing {len(documents)} documents using ingestion pipeline")
                nodes = self.pipeline.run(documents=documents)
                logger.info(f"Pipeline successfully processed {len(nodes)} nodes")
                return True
            except Exception as pipeline_error:
                logger.warning(f"Pipeline processing failed: {pipeline_error}. Falling back to direct upsert.", exc_info=True)
                
                nodes_to_upsert = []
                for doc in documents:
                    if isinstance(doc, Document):
                        # If it's a LlamaIndex Document, we might need to parse it first if not already nodes
                        # For simplicity here, assuming documents are either pre-parsed Nodes or basic Documents that can be treated as Nodes
                        nodes_to_upsert.append(Node(text=doc.text, metadata=doc.metadata or {}))
                    elif isinstance(doc, Node):
                        nodes_to_upsert.append(doc)
                    else:
                        logger.warning(f"Skipping unsupported document type: {type(doc)}")
                        continue
                
                vectors = []
                for i, node in enumerate(nodes_to_upsert):
                    text = node.get_content() # Changed from node.text
                    embedding = await self.embed_model._aget_text_embedding(text)
                    
                    node_metadata = node.metadata or {}
                    node_metadata['text'] = text # Ensure text is in metadata for retrieval
                    
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    metadata_str = json.dumps(sorted(node_metadata.items()))
                    metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
                    stable_id = f"{content_hash}-{metadata_hash}"
                    node_metadata['doc_id'] = stable_id # Ensure doc_id is set for potential use
                    
                    vectors.append((stable_id, embedding, node_metadata))
                
                if not vectors:
                    logger.info("No vectors to upsert after fallback processing.")
                    return True # No error, but nothing was upserted

                logger.info(f"Upserting {len(vectors)} vectors to Pinecone directly (fallback)")
                for vid, _, meta in vectors[:3]:
                    logger.info(f"Fallback Vector ID: {vid}, Metadata keys: {list(meta.keys())}")
                
                try:
                    self.index.upsert(vectors=vectors)
                except Exception as v6_error:
                    logger.warning(f"Upsert with Pinecone v6+ SDK failed during fallback: {v6_error}. Trying v5 syntax.", exc_info=True)
                    try:
                        self.index.upsert(vectors=[(vid, vec, meta) for vid, vec, meta in vectors])
                    except Exception as v5_error:
                        logger.error(f"All upsert attempts failed during fallback: {v5_error}", exc_info=True)
                        return False
                logger.info("Fallback direct upsert completed.")
                return True

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            return False

    async def query_similar(self, query_text: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """Query similar documents"""
        try:
            query_embedding = await self.embed_model._aget_text_embedding(query_text)
            
            try:
                logger.info(f"Querying Pinecone with top_k={top_k}")
                
                # Direct query approach
                query_response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter=filter,
                    include_metadata=True
                )
                
                # Process the response
                formatted_results = []
                if hasattr(query_response, 'matches'):
                    for match in query_response.matches:
                        try:
                            formatted_results.append({
                                'score': getattr(match, 'score', 0.0),
                                'metadata': getattr(match, 'metadata', {})
                            })
                        except AttributeError:
                            formatted_results.append({
                                'score': match.get('score', 0.0),
                                'metadata': match.get('metadata', {})
                            })
                elif isinstance(query_response, dict) and 'matches' in query_response:
                    for match in query_response['matches']:
                        formatted_results.append({
                            'score': match.get('score', 0.0),
                            'metadata': match.get('metadata', {})
                        })
                else:
                    logger.warning(f"Unexpected query response structure: {type(query_response)}")
                    if isinstance(query_response, list):
                        for item in query_response:
                            if isinstance(item, dict):
                                formatted_results.append({
                                    'score': item.get('score', 0.0),
                                    'metadata': item.get('metadata', {})
                                })
                return formatted_results
            except Exception as e:
                logger.error(f"Error with Pinecone query: {e}", exc_info=True)
                return []
        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}", exc_info=True)
            return []

    def __call__(self, action: str, **kwargs: Any) -> Any:
        """Allows calling specific tool methods by action name."""
        if action == "create_index":
            return self.create_pinecone_index_tool(kwargs['name'], kwargs['dimension'], kwargs['metric'])
        elif action == "delete_index":
            return self.delete_pinecone_index_tool(kwargs['name'])
        elif action == "get_index_stats":
            return self.get_pinecone_index_stats_tool(kwargs['name'])
        elif action == "process_documents":
            # This is async, direct call might not be suitable here without await
            # Consider how to handle async methods if called synchronously.
            # For now, returning a message or raising error.
            # asyncio.run(self.process_documents(kwargs['documents']))
            return "process_documents is async; call via await self.process_documents(...)"
        elif action == "query_similar":
            # asyncio.run(self.query_similar(kwargs['query_text'], kwargs.get('top_k', 5), kwargs.get('filter')))
            return "query_similar is async; call via await self.query_similar(...)"
        else:
            return f"Action '{action}' not recognized."
