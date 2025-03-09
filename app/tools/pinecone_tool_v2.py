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

        # Initialize vector store
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
                current_dimension = index_description.dimension
                
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
            return description.status.ready
        except Exception:
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
            return {
                "dimension": stats.dimension,
                "total_vector_count": stats.total_vector_count,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}

    async def process_documents(self, documents: List[Document]) -> bool:
        """Process and upsert documents using the ingestion pipeline"""
        try:
            # Run pipeline on documents and get nodes
            nodes = self.pipeline.run(documents=documents)
            
            # Get embeddings and metadata for each node
            vectors = []
            
            for i, node in enumerate(nodes):
                # Get embedding
                embedding = await self.embed_model._aget_text_embedding(node.text)
                
                # Get metadata
                node_metadata = {
                    'text': node.text,
                    **node.metadata
                }
                
                # Create a stable, unique ID based on content and metadata
                content_hash = hashlib.md5(node.text.encode()).hexdigest()
                metadata_str = json.dumps(sorted(node_metadata.items()))
                metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
                stable_id = f"{content_hash}_{metadata_hash}"
                node_metadata['doc_id'] = stable_id
                
                # Create vector tuple (id, values, metadata)
                vectors.append((stable_id, embedding, node_metadata))
            
            # Log vectors being upserted
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone")
            for vid, _, meta in vectors:
                logger.info(f"Vector ID: {vid}, Metadata: {meta.get('stocks', [])}, Channel: {meta.get('channel_name')}")
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            
            logger.info(f"Successfully processed and stored {len(nodes)} nodes")
            return True
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False

    async def query_similar(self, query_text: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """Query similar documents"""
        try:
            # Get query embedding
            query_embedding = await self.embed_model._aget_text_embedding(query_text)
            
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in query_response.matches:
                formatted_results.append({
                    'score': match.score,
                    'metadata': match.metadata
                })
            
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
