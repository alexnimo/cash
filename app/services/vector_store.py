from abc import ABC, abstractmethod
import os
from pinecone import Pinecone, ServerlessSpec
import lancedb
from typing import List, Dict, Any, Optional
import numpy as np
from app.core.config import get_settings
from app.models.video import TranscriptSegment
from datetime import datetime
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass

class VectorStore(ABC):
    @abstractmethod
    async def store_embeddings(self, video_id: str, embeddings: List[float], metadata: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    async def search(self, query_vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_transcript(self, video_id: str) -> Optional[List[TranscriptSegment]]:
        pass
    
    @abstractmethod
    async def store_transcript(self, video_id: str, segments: List[TranscriptSegment], embeddings: List[List[float]]) -> None:
        pass

class PineconeStore(VectorStore):
    def __init__(self):
        try:
            api_key = settings.vector_store.config.api_key
            if api_key.startswith('${'):
                api_key = os.getenv('PINECONE_API_KEY')
                if not api_key:
                    raise VectorStoreError("Pinecone API key not found in environment")
            
            environment = settings.vector_store.config.environment
            if environment.startswith('${'):
                environment = os.getenv('PINECONE_ENVIRONMENT')
                if not environment:
                    raise VectorStoreError("Pinecone environment not found in environment")
            
            logger.info(f"Initializing Pinecone with environment: {environment}")
            
            self.pc = Pinecone(api_key=api_key.strip())
            
            self.index_name = settings.vector_store.config.index_name
            logger.info(f"Using Pinecone index: {self.index_name}")
            
            existing_indexes = self.pc.list_indexes()
            logger.info(f"Existing indexes: {existing_indexes}")
            
            if self.index_name not in [idx.name for idx in existing_indexes]:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.model.embedding.dimension,
                    metric=settings.vector_store.config.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=environment
                    )
                )
            
            self.index = self.pc.Index(
                name=self.index_name,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=environment
                )
            )
            logger.info("Successfully initialized Pinecone store")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")

    async def store_embeddings(self, video_id: str, embeddings: List[float], metadata: Dict[str, Any]) -> str:
        try:
            vector_id = f"vid_{video_id}_{datetime.now().timestamp()}"
            self.index.upsert(vectors=[(vector_id, embeddings, metadata)])
            return vector_id
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise VectorStoreError(f"Failed to store embeddings: {str(e)}")

    async def search(self, query_vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")

    async def get_transcript(self, video_id: str) -> Optional[List[TranscriptSegment]]:
        try:
            results = await self.search(
                query_vector=[0] * settings.model.embedding.dimension,
                filter={"video_id": video_id, "type": "transcript"},
                top_k=100  # Retrieve all segments
            )
            
            if not results:
                return None
            
            segments = []
            for match in results:
                metadata = match.metadata
                segment = TranscriptSegment(
                    start=metadata["start"],
                    end=metadata["end"],
                    text=metadata["text"]
                )
                segments.append(segment)
            
            segments.sort(key=lambda x: x.start)
            return segments
            
        except Exception as e:
            logger.error(f"Failed to retrieve transcript: {str(e)}")
            raise VectorStoreError(f"Failed to retrieve transcript: {str(e)}")

    async def store_transcript(self, video_id: str, segments: List[TranscriptSegment], embeddings: Optional[List[List[float]]] = None) -> None:
        try:
            vectors = []
            # If no embeddings provided, use zero vectors
            if embeddings is None:
                embeddings = [[0.0] * settings.model.embedding.dimension for _ in segments]
            
            for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
                vector_id = f"transcript_{video_id}_{i}"
                metadata = {
                    "video_id": video_id,
                    "type": "transcript",
                    "start": segment.start_time,  
                    "end": segment.end_time,      
                    "text": segment.text
                }
                vectors.append((vector_id, embedding, metadata))
            
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
        except Exception as e:
            logger.error(f"Failed to store transcript: {str(e)}")
            raise VectorStoreError(f"Failed to store transcript: {str(e)}")

class LanceDBStore(VectorStore):
    def __init__(self):
        try:
            os.makedirs(settings.lancedb_uri, exist_ok=True)
            self.db = lancedb.connect(settings.lancedb_uri)
            self.table_name = "video_embeddings"
            
            if self.table_name not in self.db.table_names():
                logger.info(f"Creating new table: {self.table_name}")
                schema = {
                    "id": "string",
                    "vector": f"float32[{settings.model.embedding.dimension}]",
                    "metadata": "json",
                    "timestamp": "timestamp"
                }
                self.db.create_table(self.table_name, schema=schema)
            self.table = self.db[self.table_name]
            logger.info("Successfully initialized LanceDB store")
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {str(e)}")
            raise VectorStoreError(f"Failed to initialize LanceDB: {str(e)}")
    
    async def store_embeddings(self, video_id: str, embeddings: List[float], metadata: Dict[str, Any]) -> str:
        try:
            vector_id = f"vid_{video_id}_{datetime.now().timestamp()}"
            self.table.add([{
                "id": vector_id,
                "vector": np.array(embeddings, dtype=np.float32),
                "metadata": metadata,
                "timestamp": datetime.now()
            }])
            return vector_id
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise VectorStoreError(f"Failed to store embeddings: {str(e)}")
    
    async def search(self, query_vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            query = self.table.search(query_vector)
            if filter:
                for key, value in filter.items():
                    query = query.where(f"metadata['{key}'] = '{value}'")
            
            results = query.limit(top_k).to_list()
            return [
                {
                    "id": result["id"],
                    "score": result["_distance"],
                    "metadata": result["metadata"]
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")
            
    async def get_transcript(self, video_id: str) -> Optional[List[TranscriptSegment]]:
        try:
            results = self.table.search([0] * settings.model.embedding.dimension) \
                .where(f"metadata['video_id'] = '{video_id}'") \
                .where("metadata['type'] = 'transcript'") \
                .limit(100) \
                .to_list()
            
            if not results:
                return None
                
            segments = []
            for result in results:
                segment_data = result["metadata"]["segment"]
                segments.append(TranscriptSegment(**segment_data))
            
            return sorted(segments, key=lambda x: x.start_time)
            
        except Exception as e:
            logger.error(f"Failed to retrieve transcript: {str(e)}")
            raise VectorStoreError(f"Failed to retrieve transcript: {str(e)}")
            
    async def store_transcript(self, video_id: str, segments: List[TranscriptSegment], embeddings: Optional[List[List[float]]] = None) -> None:
        try:
            records = []
            # If no embeddings provided, use zero vectors
            if embeddings is None:
                embeddings = [[0.0] * settings.model.embedding.dimension for _ in segments]
            
            for segment, embedding in zip(segments, embeddings):
                vector_id = f"vid_{video_id}_transcript_{segment.start_time}"
                records.append({
                    "id": vector_id,
                    "vector": np.array(embedding, dtype=np.float32),
                    "metadata": {
                        "video_id": video_id,
                        "type": "transcript",
                        "segment": segment.dict(),
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now()
                })
            
            self.table.add(records)
        except Exception as e:
            logger.error(f"Failed to store transcript: {str(e)}")
            raise VectorStoreError(f"Failed to store transcript: {str(e)}")

def get_vector_store() -> VectorStore:
    """Get the configured vector store instance"""
    try:
        if settings.vector_store.type == "pinecone":
            return PineconeStore()
        return LanceDBStore()
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise VectorStoreError(f"Failed to initialize vector store: {str(e)}")
