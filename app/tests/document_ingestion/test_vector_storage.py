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
    def test_store_chunks_with_textchunk_objects(self, mock_uuid, mock_openai_embeddings, mock_get_chroma_client):
        """Test storing TextChunk objects."""
        from rag.ingestion.vector_storage import VectorStorage
        from rag.ingestion.text_chunker import TextChunk
        
        # Setup mocks
        mock_uuid.side_effect = [
            uuid.UUID('12345678-1234-5678-1234-567812345678'), 
            uuid.UUID('87654321-4321-8765-4321-876543218765'),
            uuid.UUID('11111111-2222-3333-4444-555555555555')
        ]
        
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]
        ]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        # Create TextChunk objects
        chunks = [
            TextChunk("This is the first chunk of text."),
            TextChunk("This is the second chunk of text."),
            TextChunk("This is the third chunk from another document.")
        ]
        
        storage = VectorStorage()
        doc_ids = storage.store_chunks(chunks)
        
        # Assertions
        assert len(doc_ids) == 3
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        assert len(call_args['documents']) == 3
        assert len(call_args['embeddings']) == 3
        assert len(call_args['ids']) == 3
        
        # Check content extraction
        assert call_args['documents'][0] == "This is the first chunk of text."
        assert call_args['documents'][1] == "This is the second chunk of text."
        assert call_args['documents'][2] == "This is the third chunk from another document."
        
        # Verify no metadata is stored
        assert 'metadatas' not in call_args
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_store_chunks_with_strings(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test storing string chunks."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        chunks = ["chunk 1", "chunk 2"]
        
        doc_ids = storage.store_chunks(chunks)
        
        # Should generate embeddings automatically
        mock_embeddings_instance.embed_documents.assert_called_once_with(chunks)
        
        # Should store without metadata
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args['documents'] == chunks
        assert 'metadatas' not in call_args
    
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
        
        doc_ids = storage.store_chunks(chunks, embeddings=embeddings)
        
        # Should not generate embeddings when provided
        mock_embeddings_instance.embed_documents.assert_not_called()
        
        # Should store the provided embeddings
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args['embeddings'] == embeddings
    
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
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_store_chunks_batch_processing(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test storing chunks with batch processing."""
        from rag.ingestion.vector_storage import VectorStorage
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        # Create embeddings for a large number of chunks
        large_embeddings = [[0.1, 0.2] for _ in range(500)]
        mock_embeddings_instance.embed_documents.return_value = large_embeddings
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        storage = VectorStorage()
        # Create a large number of chunks to test batching
        chunks = [f"chunk {i}" for i in range(500)]
        
        doc_ids = storage.store_chunks(chunks, batch_size=100)
        
        # Should call add multiple times for batching
        assert mock_collection.add.call_count >= 2
        assert len(doc_ids) == 500
    
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_store_chunks_dimension_mismatch_error(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test error handling when text and embedding dimensions don't match."""
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
        embeddings = [[0.1, 0.2]]  # Only one embedding for two chunks
        
        with pytest.raises(ValueError, match="Texts and embeddings must have the same length"):
            storage.store_chunks(chunks, embeddings=embeddings)