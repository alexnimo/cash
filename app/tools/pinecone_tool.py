from llama_index.core.tools import ToolSpec
import os

# Optional import with graceful fallback
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None

class PineconeToolSpec(ToolSpec):
    def __init__(self):
        super().__init__()
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not available - package not installed")
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
    def query_index(self, index_name: str, query: str, namespace: str):
        index = self.pc.Index(index_name)
        return index.query(vector=query, namespace=namespace)
    
    def upsert_data(self, index_name: str, vectors: list, namespace: str):
        index = self.pc.Index(index_name)
        return index.upsert(vectors=vectors, namespace=namespace)
    
    def create_index(self, index_name: str, dimension: int):
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        return f"Index {index_name} ready"
