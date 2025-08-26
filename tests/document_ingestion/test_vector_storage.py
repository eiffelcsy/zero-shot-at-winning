import pytest
from unittest.mock import Mock, patch, MagicMock


class TestVectorStorage:
    """Tests for vector storage component."""
    
    @patch('app.vector_storage.get_embedding_function')
    def test_generate_embeddings(self, mock_embedding_fn):
        """Test creating embeddings for text chunks."""
        from app.vector_storage import VectorStorage
        
        # Mock embedding function
        mock_embedding_fn.return_value = lambda x: [0.1, 0.2, 0.3, 0.4, 0.5]
        
        storage = VectorStorage()
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        embeddings = storage.generate_embeddings(chunks)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 5 for emb in embeddings)
    
    @patch('app.vector_storage.chromadb.Client')
    def test_store_in_chromadb(self, mock_chroma_client):
        """Test saving chunks and embeddings to ChromaDB."""
        from app.vector_storage import VectorStorage
        
        # Mock ChromaDB collection
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        chunks = ["chunk 1", "chunk 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"source": "doc1.pdf"}, {"source": "doc1.pdf"}]
        
        storage.store_chunks(chunks, embeddings, metadatas)
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert len(call_args['documents']) == 2
        assert len(call_args['embeddings']) == 2
        assert len(call_args['metadatas']) == 2
    
    @patch('app.vector_storage.chromadb.Client')
    def test_search_similar(self, mock_chroma_client):
        """Test finding similar chunks by query."""
        from app.vector_storage import VectorStorage
        
        # Mock ChromaDB collection and search results
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['similar chunk 1', 'similar chunk 2']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[{'source': 'doc1.pdf'}, {'source': 'doc2.pdf'}]]
        }
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        results = storage.search_similar("test query", n_results=2)
        
        assert len(results) == 2
        assert all('document' in result for result in results)
        assert all('distance' in result for result in results)
        assert all('metadata' in result for result in results)
    
    @patch('app.vector_storage.get_embedding_function')
    def test_generate_embeddings_empty_input(self, mock_embedding_fn):
        """Test embedding generation with empty input."""
        from app.vector_storage import VectorStorage
        
        storage = VectorStorage()
        embeddings = storage.generate_embeddings([])
        
        assert embeddings == []
        mock_embedding_fn.assert_not_called()
