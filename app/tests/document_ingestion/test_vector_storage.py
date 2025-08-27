import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid


class TestVectorStorage:
    """Tests for vector storage component."""
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_init(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test VectorStorage initialization."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        # Initialize VectorStorage
        storage = VectorStorage(embedding_model="text-embedding-3-small", collection_name="test_collection")
        
        # Assertions
        mock_openai_embeddings.assert_called_once_with(model="text-embedding-3-small")
        mock_get_chroma_client.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"description": "RAG document chunks with embeddings"}
        )
        assert storage.embedding_model == "text-embedding-3-small"
        assert storage.collection_name == "test_collection"
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_generate_embeddings(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test creating embeddings for text chunks."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        embeddings = storage.generate_embeddings(chunks)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 5 for emb in embeddings)
        mock_embeddings_instance.embed_documents.assert_called_once_with(chunks)
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_generate_embeddings_empty_input(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test embedding generation with empty input."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        embeddings = storage.generate_embeddings([])
        
        assert embeddings == []
        mock_embeddings_instance.embed_documents.assert_not_called()
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    @patch('uuid.uuid4')
    def test_store_chunks_with_chunk_objects(self, mock_uuid, mock_openai_embeddings, mock_get_chroma_client, sample_chunks):
        """Test storing ChunkWithMetadata objects."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_uuid.side_effect = [uuid.UUID('12345678-1234-5678-1234-567812345678'), 
                                uuid.UUID('87654321-4321-8765-4321-876543218765'),
                                uuid.UUID('11111111-2222-3333-4444-555555555555')]
        
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]
        ]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        doc_ids = storage.store_chunks(sample_chunks)
        
        # Assertions
        assert len(doc_ids) == 3
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        assert len(call_args['documents']) == 3
        assert len(call_args['embeddings']) == 3
        assert len(call_args['metadatas']) == 3
        assert len(call_args['ids']) == 3
        
        # Check content extraction
        assert call_args['documents'][0] == "This is the first chunk of text."
        assert call_args['documents'][1] == "This is the second chunk of text."
        assert call_args['documents'][2] == "This is the third chunk from another document."
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_store_chunks_with_precomputed_embeddings(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test storing chunks with precomputed embeddings."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        chunks = ["chunk 1", "chunk 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"source": "doc1.pdf"}, {"source": "doc1.pdf"}]
        
        doc_ids = storage.store_chunks(chunks, embeddings=embeddings, metadatas=metadatas)
        
        # Should not generate embeddings when provided
        mock_embeddings_instance.embed_documents.assert_not_called()
        
        # Should store the provided embeddings
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args['embeddings'] == embeddings
        assert call_args['metadatas'] == metadatas
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_search_similar(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test finding similar chunks by query."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.2, 0.3, 0.4, 0.5, 0.6]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['similar chunk 1', 'similar chunk 2']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[{'source': 'doc1.pdf'}, {'source': 'doc2.pdf'}]]
        }
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        results = storage.search_similar("test query", n_results=2)
        
        # Assertions
        assert len(results) == 2
        assert all('document' in result for result in results)
        assert all('distance' in result for result in results)
        assert all('metadata' in result for result in results)
        assert all('rank' in result for result in results)
        
        # Check specific values
        assert results[0]['document'] == 'similar chunk 1'
        assert results[0]['distance'] == 0.1
        assert results[0]['rank'] == 1
        assert results[1]['rank'] == 2
        
        # Check method calls
        mock_embeddings_instance.embed_query.assert_called_once_with("test query")
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.2, 0.3, 0.4, 0.5, 0.6]],
            n_results=2
        )
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_search_similar_with_metadata_filter(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test search with metadata filtering."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.2, 0.3, 0.4, 0.5, 0.6]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['filtered chunk']],
            'distances': [[0.1]],
            'metadatas': [[{'source': 'doc1.pdf'}]]
        }
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        where_filter = {'source': 'doc1.pdf'}
        results = storage.search_similar("test query", n_results=5, where=where_filter)
        
        # Check that where filter was passed to query
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.2, 0.3, 0.4, 0.5, 0.6]],
            n_results=5,
            where=where_filter
        )
        
        assert len(results) == 1
        assert results[0]['document'] == 'filtered chunk'
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_get_collection_stats(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test getting collection statistics."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage(embedding_model="test-model", collection_name="test-collection")
        stats = storage.get_collection_stats()
        
        assert stats['collection_name'] == 'test-collection'
        assert stats['total_documents'] == 42
        assert stats['embedding_model'] == 'test-model'
        mock_collection.count.assert_called_once()
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_delete_by_metadata(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test deleting documents by metadata filter."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': ['id1', 'id2', 'id3']}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        where_filter = {'source': 'doc1.pdf'}
        deleted_count = storage.delete_by_metadata(where_filter)
        
        assert deleted_count == 3
        mock_collection.get.assert_called_once_with(where=where_filter)
        mock_collection.delete.assert_called_once_with(ids=['id1', 'id2', 'id3'])
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_clear_collection(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test clearing all documents from collection."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': ['id1', 'id2', 'id3', 'id4']}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        success = storage.clear_collection()
        
        assert success is True
        mock_collection.get.assert_called_once()
        mock_collection.delete.assert_called_once_with(ids=['id1', 'id2', 'id3', 'id4'])
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_store_chunks_empty_input(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test storing empty chunks list."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        doc_ids = storage.store_chunks([])
        
        assert doc_ids == []
        mock_collection.add.assert_not_called()
        mock_embeddings_instance.embed_documents.assert_not_called()
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_generate_embeddings_error_handling(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test error handling in embedding generation."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.side_effect = Exception("API Error")
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        
        with pytest.raises(ValueError, match="Embedding generation failed"):
            storage.generate_embeddings(["test chunk"])