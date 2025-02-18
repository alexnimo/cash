import pytest
from unittest.mock import Mock, patch
from app.tools.pinecone_tool_v2 import PineconeAdvancedToolSpec

@pytest.fixture
def mock_pinecone():
    with patch('app.tools.pinecone_tool_v2.Pinecone') as mock:
        yield mock

@pytest.fixture
def mock_embedding():
    with patch('app.tools.pinecone_tool_v2.Embedding') as mock:
        mock.create_default.return_value.get_text_embedding.return_value = [0.1] * 768
        mock.create_default.return_value.get_text_embedding_batch.return_value = [[0.1] * 768]
        yield mock

@pytest.fixture
def pinecone_tool(mock_pinecone, mock_embedding):
    with patch.dict('os.environ', {'PINECONE_API_KEY': 'test-key'}):
        return PineconeAdvancedToolSpec()

def test_create_embeddings(pinecone_tool):
    texts = ["test text"]
    metadata = [{"source": "test"}]
    result = pinecone_tool.create_embeddings(texts, metadata)
    assert len(result) == 1
    assert "values" in result[0]
    assert "metadata" in result[0]

def test_query_index(pinecone_tool, mock_pinecone):
    mock_index = Mock()
    mock_pinecone.return_value.Index.return_value = mock_index
    mock_index.query.return_value = {"matches": []}
    
    result = pinecone_tool.query_index("test-index", "test query")
    assert "matches" in result

def test_delete_vectors(pinecone_tool, mock_pinecone):
    mock_index = Mock()
    mock_pinecone.return_value.Index.return_value = mock_index
    
    pinecone_tool.delete_vectors("test-index", ["test-id"])
    mock_index.delete.assert_called_once_with(ids=["test-id"], namespace="default")

def test_create_index(pinecone_tool, mock_pinecone):
    mock_pinecone.return_value.list_indexes.return_value.names.return_value = []
    mock_pinecone.return_value.describe_index.return_value.status = {"ready": True}
    
    result = pinecone_tool.create_index("test-index")
    assert result is not None
