from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import Document, Node
from llama_index.core import Response
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec, Index
from typing import List, Dict, Any, Optional
from app.config.agent_config import AGENT_CONFIG
import numpy as np
import os
import logging
import json
import time
import hashlib

logger = logging.getLogger(__name__)

class PineconeAdvancedToolSpec(BaseTool):
    """Advanced Pinecone tool for vector operations"""
    def __init__(self):
        super().__init__()
        
        # Get configurations
        self.pinecone_config = AGENT_CONFIG['pinecone']
        self.embedding_config = AGENT_CONFIG['embedding']
        
        # Try to get API key from environment first, then config
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.warning("PINECONE_API_KEY not found in environment variables, checking config...")
            if 'api_key' not in self.pinecone_config:
                raise ValueError("Pinecone API key not found in environment variables or config")
            api_key = self.pinecone_config['api_key']
        
        self.pc = Pinecone(api_key=api_key)
        
        # Set properties
        self.index_name = self.pinecone_config['index_name']
        self.dimension = self.embedding_config.get('dimension', 768)
        self.metric = self.embedding_config['metric']
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_config['model'],
            trust_remote_code=True
        )
        
        # Ensure index has correct dimension
        self._ensure_index_dimension()
        
        self.index = self._initialize_index()
        self.pipeline = self._initialize_pipeline()

    def _validate_env_vars(self):
        """Validate required environment variables"""
        required_vars = ["PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _initialize_index(self) -> Index:
        """Initialize or get existing Pinecone index"""
        try:
            # Check if index exists
            if not self.index_exists(self.index_name):
                logger.info(f"Creating new index {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.pinecone_config['cloud'],
                        region=self.pinecone_config['region']
                    ),
                    deletion_protection='enabled'
                )
            
            return self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            raise

    def _initialize_pipeline(self):
        """Initialize ingestion pipeline"""
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        import importlib

        # Check if llama_index.vector_stores.pinecone has get_pinecone_client
        pinecone_vectorstore_module = importlib.import_module('llama_index.vector_stores.pinecone')
        if hasattr(pinecone_vectorstore_module, 'get_pinecone_client'):
            # For newer versions of LlamaIndex that need a custom init for Pinecone
            try:
                # Initialize vector store
                vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    environment=self.pinecone_config.get('environment', None),
                    metadata_filters=True,
                    pinecone_kwargs={"api_key": self.pc.api_key},
                    embedding_model=self.embed_model
                )
                logger.info("Initialized PineconeVectorStore with newer method")
            except Exception as e:
                logger.warning(f"Could not initialize PineconeVectorStore with newer method: {e}")
                # Try the original approach
                vector_store = PineconeVectorStore(
                    pinecone_index=self.index,
                    embedding_model=self.embed_model
                )
        else:
            # Initialize vector store with original approach
            vector_store = PineconeVectorStore(
                pinecone_index=self.index,
                embedding_model=self.embed_model
            )
        
        # Initialize pipeline with node parser and vector store
        return IngestionPipeline(
            transformations=[
                SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=self.embed_model
                ),
            ],
            vector_store=vector_store
        )

    def _ensure_index_dimension(self):
        """Ensure index exists with correct dimension"""
        try:
            target_dimension = self.embedding_config.get('dimension')
            
            if not target_dimension:
                logger.error("No dimension specified in embedding config")
                raise ValueError("No dimension specified in embedding config")
            
            # Check if index exists
            try:
                index_description = self.pc.describe_index(self.index_name)
                
                # Handle dimension access based on SDK version
                try:
                    # Try accessing as an attribute (newer SDK)
                    current_dimension = getattr(index_description, "dimension", None)
                except (AttributeError, TypeError):
                    # Try accessing as a dictionary (older SDK)
                    current_dimension = index_description.get("dimension") if isinstance(index_description, dict) else None
                    
                # If we still couldn't get dimension, try alternate attribute names
                if current_dimension is None:
                    # Try other possible attribute names
                    for attr in ["dimension", "dimensions", "vector_dimension", "vector_dimensions"]:
                        current_dimension = getattr(index_description, attr, None)
                        if current_dimension is not None:
                            break
                
                if current_dimension is None:
                    logger.warning(f"Could not determine dimension of existing index {self.index_name}")
                    # Continue with creation to be safe
                    raise ValueError(f"Could not determine dimension of existing index {self.index_name}")
                
                if current_dimension == target_dimension:
                    logger.info(f"Using existing index {self.index_name} with correct dimension {target_dimension}")
                    return
                else:
                    logger.warning(f"Index dimension mismatch. Current: {current_dimension}, Target: {target_dimension}")
                    logger.warning("Cannot modify existing index dimensions. Please manually delete the index if you need to change its dimension.")
                    raise ValueError(f"Index {self.index_name} exists with incorrect dimension {current_dimension} (expected {target_dimension})")
            except Exception as e:
                if "not found" in str(e).lower():
                    # Index doesn't exist, create it
                    logger.info(f"Creating new index {self.index_name}")
                    try:
                        self.pc.create_index(
                            name=self.index_name,
                            dimension=target_dimension,
                            metric=self.metric,
                            spec=ServerlessSpec(
                                cloud=self.pinecone_config.get('cloud', 'aws'),
                                region=self.pinecone_config.get('region', 'us-west-2')
                            ),
                            deletion_protection='enabled'
                        )
                        # Wait for index to be ready
                        while not self._is_index_ready(self.index_name):
                            time.sleep(1)
                        logger.info(f"Created new Pinecone index: {self.index_name} with dimension {target_dimension}")
                    except Exception as create_error:
                        if "ALREADY_EXISTS" in str(create_error):
                            logger.info(f"Index {self.index_name} was created by another process, using existing index")
                            return
                        logger.error(f"Error creating index: {str(create_error)}")
                        raise
                else:
                    logger.error(f"Error checking index: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error ensuring index dimension: {str(e)}")
            raise

    def _is_index_ready(self, index_name: str) -> bool:
        """Check if an index is ready and available"""
        try:
            description = self.pc.describe_index(index_name)
            
            # Try to check ready status in different ways based on SDK version
            # 1. Try status.ready (newer SDK)
            try:
                if hasattr(description, 'status') and hasattr(description.status, 'ready'):
                    return description.status.ready
            except (AttributeError, TypeError):
                pass
                
            # 2. Try direct ready attribute
            if hasattr(description, 'ready'):
                return description.ready
                
            # 3. Try dictionary access (older SDK)
            if isinstance(description, dict):
                if 'status' in description and 'ready' in description['status']:
                    return description['status']['ready']
                if 'ready' in description:
                    return description['ready']
            
            # 4. If we can get the description at all and can't determine ready state,
            # assume it's ready (safer option)
            logger.warning(f"Could not determine ready state for index {index_name}, assuming ready")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking if index is ready: {str(e)}")
            return False

    def index_exists(self, index_name: str) -> bool:
        """Check if a Pinecone index exists"""
        try:
            return index_name in self.pc.list_indexes().names()
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False

    def create_index(self, name: str, dimension: int = 768, metric: str = "cosine") -> bool:
        """Create a new Pinecone index"""
        try:
            if self.index_exists(name):
                logger.warning(f"Index {name} already exists")
                return False
                
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=self.pinecone_config['cloud'],
                    region=self.pinecone_config['region']
                ),
                deletion_protection='enabled'
            )
            logger.info(f"Successfully created index {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False

    def delete_index(self, name: str) -> bool:
        """Delete a Pinecone index"""
        try:
            if not self.index_exists(name):
                logger.warning(f"Index {name} does not exist")
                return False
                
            self.pc.delete_index(name)
            logger.info(f"Successfully deleted index {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False

    def get_index_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a Pinecone index"""
        try:
            if not self.index_exists(name):
                logger.warning(f"Index {name} does not exist")
                return {}
                
            index = self.pc.Index(name)
            stats = index.describe_index_stats()
            
            # Handle different response structures based on Pinecone SDK version
            # New SDK returns an object, old SDK returned a dict
            try:
                # Try to access stats.dimension (newer SDK)
                return {
                    "dimension": getattr(stats, "dimension", 0),
                    "total_vector_count": getattr(stats, "total_vector_count", 0),
                    "namespaces": getattr(stats, "namespaces", {})
                }
            except AttributeError:
                # Try to access as dict (older SDK)
                if isinstance(stats, dict):
                    return {
                        "dimension": stats.get("dimension", 0),
                        "total_vector_count": stats.get("total_vector_count", 0),
                        "namespaces": stats.get("namespaces", {})
                    }
                    
                # If all else fails, return a minimal structure
                return {
                    "dimension": 0,
                    "total_vector_count": 0,
                    "namespaces": {}
                }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}

    async def process_documents(self, documents: List[Document]) -> bool:
        """Process and upsert documents using the ingestion pipeline"""
        try:
            # Try using the pipeline first
            try:
                logger.info(f"Processing {len(documents)} documents using ingestion pipeline")
                # Run pipeline on documents and get nodes
                nodes = self.pipeline.run(documents=documents)
                logger.info(f"Pipeline successfully processed {len(nodes)} nodes")
                return True
            except Exception as pipeline_error:
                # If pipeline fails, use direct upsert as fallback
                logger.warning(f"Pipeline processing failed: {pipeline_error}. Falling back to direct upsert.")
                
                # Process documents directly
                nodes = []
                for doc in documents:
                    nodes.append(doc)
                
                # Get embeddings and metadata for each node
                vectors = []
                
                for i, node in enumerate(nodes):
                    # Get the text content - handle both Document and Node objects
                    text = node.text if hasattr(node, 'text') else str(node)
                    
                    # Get embedding
                    embedding = await self.embed_model._aget_text_embedding(text)
                    
                    # Get metadata - handle both Document and Node objects
                    if hasattr(node, 'metadata') and node.metadata:
                        node_metadata = {
                            'text': text,
                            **node.metadata
                        }
                    else:
                        node_metadata = {'text': text}
                    
                    # Create a stable, unique ID based on content and metadata
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    metadata_str = json.dumps(sorted(node_metadata.items()))
                    metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
                    stable_id = f"{content_hash}_{metadata_hash}"
                    node_metadata['doc_id'] = stable_id
                    
                    # Create vector tuple (id, values, metadata)
                    vectors.append((stable_id, embedding, node_metadata))
                
                # Log vectors being upserted
                logger.info(f"Upserting {len(vectors)} vectors to Pinecone directly")
                for vid, _, meta in vectors[:3]:  # Log only first 3 vectors to avoid excessive logging
                    logger.info(f"Vector ID: {vid}, Metadata keys: {list(meta.keys())}")
                
                try:
                    # Try upsert using the Pinecone SDK v6+ syntax
                    logger.info("Attempting upsert with Pinecone v6+ SDK")
                    self.index.upsert(vectors=vectors)
                except Exception as v6_error:
                    logger.warning(f"Upsert with v6+ SDK failed: {v6_error}. Trying v5 syntax.")
                    try:
                        # Try older Pinecone SDK v5 syntax
                        self.index.upsert(vectors=[(vid, vec, meta) for vid, vec, meta in vectors])
                    except Exception as v5_error:
                        logger.error(f"All upsert attempts failed: {v5_error}")
                        raise
                
                logger.info(f"Successfully processed and stored {len(nodes)} nodes directly")
                return True
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False

    async def query_similar(self, query_text: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """Query similar documents"""
        try:
            # Get query embedding
            query_embedding = await self.embed_model._aget_text_embedding(query_text)
            
            # Try different query approaches for compatibility with different Pinecone SDK versions
            try:
                logger.info(f"Querying Pinecone with top_k={top_k} using v6+ SDK approach")
                # Try newer SDK query approach
                query_response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter=filter,
                    include_metadata=True
                )
                
                # Format results based on response structure
                formatted_results = []
                
                # Check if response has 'matches' attribute (newer SDK)
                if hasattr(query_response, 'matches'):
                    for match in query_response.matches:
                        try:
                            # Try accessing as object properties first
                            formatted_results.append({
                                'score': getattr(match, 'score', 0.0),
                                'metadata': getattr(match, 'metadata', {})
                            })
                        except AttributeError:
                            # Try dict access if object access fails
                            formatted_results.append({
                                'score': match.get('score', 0.0),
                                'metadata': match.get('metadata', {})
                            })
                # Check if response is dict with 'matches' key (older SDK)
                elif isinstance(query_response, dict) and 'matches' in query_response:
                    for match in query_response['matches']:
                        formatted_results.append({
                            'score': match.get('score', 0.0),
                            'metadata': match.get('metadata', {})
                        })
                else:
                    # Handle case where response structure is unknown
                    logger.warning(f"Unexpected query response structure: {type(query_response)}")
                    # Try to extract something useful
                    if isinstance(query_response, list):
                        for item in query_response:
                            if isinstance(item, dict):
                                formatted_results.append({
                                    'score': item.get('score', 0.0),
                                    'metadata': item.get('metadata', {})
                                })
                
                logger.info(f"Query returned {len(formatted_results)} results")
                    
            except Exception as e:
                logger.warning(f"Error with v6+ query approach: {e}. Trying v5 approach...")
                try:
                    # Try older SDK query approach
                    query_response = self.index.query(
                        queries=[query_embedding],  # Array instead of single vector
                        top_k=top_k,
                        include_metadata=True,
                        filter=filter
                    )
                    
                    formatted_results = []
                    # Handle different response structures
                    if isinstance(query_response, dict) and 'results' in query_response:
                        for match in query_response['results'][0]['matches']:
                            formatted_results.append({
                                'score': match.get('score', 0.0),
                                'metadata': match.get('metadata', {})
                            })
                    else:
                        logger.warning(f"Unexpected query response structure from v5 approach")
                        return []
                        
                except Exception as e2:
                    logger.error(f"All query attempts failed: {e2}")
                    return []
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}")
            return []

    def get_embedding_model(self) -> HuggingFaceEmbedding:
        """Get the embedding model"""
        return self.embed_model

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata"""
        return ToolMetadata(
            name="pinecone_advanced_tool",
            description="Advanced Pinecone tool for vector database operations",
            fn_schema={
                "type": "function",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform (upsert, query, delete)",
                    },
                    "data": {
                        "type": "object",
                        "description": "Data for the operation",
                    }
                },
                "required": ["operation", "data"]
            }
        )

    async def __call__(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with given operation and data"""
        try:
            if operation == "upsert":
                return await self.process_documents(data.get("documents", []))
            elif operation == "query":
                return await self.query_similar(data.get("query_text", ""), data.get("top_k", 5), data.get("filter"))
            elif operation == "delete":
                return {"result": "Not implemented"}
            elif operation == "index_exists":
                return {"result": self.index_exists(data.get("index_name", ""))}
            elif operation == "create_index":
                return {"result": self.create_index(data.get("name", ""), data.get("dimension", 768), data.get("metric", "cosine"))}
            elif operation == "delete_index":
                return {"result": self.delete_index(data.get("name", ""))}
            elif operation == "get_index_stats":
                return {"result": self.get_index_stats(data.get("name", ""))}
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in PineconeAdvancedToolSpec: {str(e)}")
            raise
