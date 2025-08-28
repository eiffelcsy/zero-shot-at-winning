import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestPipeline:
    """Tests for end-to-end PDF ingestion pipeline."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_process_single_pdf(self, mock_openai_embeddings, mock_get_chroma_client, sample_pdf_upload):
        """Test end-to-end processing of a single uploaded PDF."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Mock external dependencies
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        mock_embeddings = Mock()
        # Make embed_documents return embeddings matching the number of input texts
        def mock_embed_documents(texts):
            return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(len(texts))]
        mock_embeddings.embed_documents.side_effect = mock_embed_documents
        mock_openai_embeddings.return_value = mock_embeddings
        
        # Test pipeline with uploaded file - this will use real components but mocked external dependencies
        pipeline = PDFIngestionPipeline()
        result = pipeline.process_pdf(sample_pdf_upload)
        
        # Verify results (basic integration test)
        assert result["status"] == "success"
        assert "chunks_processed" in result
        assert "filename" in result
        assert "document_ids" in result
        assert "text_length" in result
        assert "chunk_stats" in result
        
        # Verify expected result structure (no metadata fields)
        assert "metadata" not in result  # Should not have metadata
        assert result["filename"] == "sample_document.pdf"
        assert result["chunks_processed"] > 0
        assert isinstance(result["document_ids"], list)
        assert len(result["document_ids"]) == result["chunks_processed"]
        
        # Verify processing details
        assert "processing_details" in result
        details = result["processing_details"]
        assert details["text_extraction_successful"] is True
        assert details["embeddings_generated"] > 0
        assert details["storage_successful"] is True
        
        # Verify external dependencies were called
        mock_get_chroma_client.assert_called()
        mock_openai_embeddings.assert_called()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_process_multiple_pdfs(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test processing a batch of uploaded PDFs."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Mock external dependencies
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        mock_embeddings = Mock()
        # Make embed_documents return embeddings matching the number of input texts
        def mock_embed_documents(texts):
            return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(len(texts))]
        mock_embeddings.embed_documents.side_effect = mock_embed_documents
        mock_openai_embeddings.return_value = mock_embeddings
        
        # Create simple mock upload files
        mock_uploads = []
        for i in range(3):
            mock_upload = Mock()
            mock_upload.filename = f"document_{i+1}.pdf"
            mock_upload.content_type = "application/pdf"
            mock_uploads.append(mock_upload)
        
        # Test batch processing - this will test that the pipeline can handle multiple files
        pipeline = PDFIngestionPipeline()
        results = pipeline.process_batch(mock_uploads)
        
        # Basic verification that batch processing works
        assert len(results) == 3
        assert all("status" in result for result in results)
        assert all("filename" in result for result in results)
        assert all("batch_summary" in result for result in results)
        
        # Check batch summary
        for result in results:
            batch_summary = result["batch_summary"]
            assert batch_summary["total_files"] == 3
            assert "successful" in batch_summary
            assert "failed" in batch_summary
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    def test_process_invalid_pdf(self, invalid_file_upload):
        """Test pipeline handling of invalid uploaded file."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Test with invalid file - this should fail at the PDF processing stage
        pipeline = PDFIngestionPipeline()
        result = pipeline.process_pdf(invalid_file_upload)
        
        # Verify error handling - the real PDF processor will reject the invalid file
        assert result["status"] == "error"
        assert "error" in result
        assert result["chunks_processed"] == 0
        assert "filename" in result
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_pipeline_stats(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test getting pipeline statistics."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Mock external dependencies
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        mock_embeddings = Mock()
        mock_openai_embeddings.return_value = mock_embeddings
        
        pipeline = PDFIngestionPipeline(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="test-model",
            collection_name="test-collection",
            batch_size=200
        )
        
        stats = pipeline.get_pipeline_stats()
        
        assert stats["status"] == "operational"
        assert "pipeline_config" in stats
        assert "storage_stats" in stats
        
        config = stats["pipeline_config"]
        assert config["chunk_size"] == 500
        assert config["chunk_overlap"] == 100
        assert config["embedding_model"] == "test-model"
        assert config["collection_name"] == "test-collection"
        assert config["batch_size"] == 200
        
        storage_stats = stats["storage_stats"]
        assert storage_stats["total_documents"] == 10
        assert storage_stats["collection_name"] == "test-collection"
        assert storage_stats["embedding_model"] == "test-model"
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_clear_collection(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test clearing the collection."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Mock external dependencies
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': ['id1', 'id2']}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        mock_embeddings = Mock()
        mock_openai_embeddings.return_value = mock_embeddings
        
        pipeline = PDFIngestionPipeline(collection_name="test-collection")
        result = pipeline.clear_collection()
        
        assert result["status"] == "success"
        assert "message" in result
        assert result["collection_name"] == "test-collection"
        mock_collection.get.assert_called_once()
        mock_collection.delete.assert_called_once()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_process_empty_pdf_result(self, mock_openai_embeddings, mock_get_chroma_client, empty_pdf_upload):
        """Test processing a PDF that results in no chunks."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Mock external dependencies
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        mock_embeddings = Mock()
        mock_openai_embeddings.return_value = mock_embeddings
        
        pipeline = PDFIngestionPipeline()
        result = pipeline.process_pdf(empty_pdf_upload)
        
        # Should get an error for empty PDF
        assert result["status"] == "error"
        assert result["chunks_processed"] == 0
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    @patch('rag.ingestion.vector_storage.get_chroma_client')
    @patch('rag.ingestion.vector_storage.OpenAIEmbeddings')
    def test_pipeline_initialization_parameters(self, mock_openai_embeddings, mock_get_chroma_client):
        """Test pipeline initialization with custom parameters."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        # Mock external dependencies
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_chroma_client.return_value = mock_client
        
        mock_embeddings = Mock()
        mock_openai_embeddings.return_value = mock_embeddings
        
        pipeline = PDFIngestionPipeline(
            chunk_size=800,
            chunk_overlap=150,
            embedding_model="custom-model",
            collection_name="custom-collection",
            batch_size=400
        )
        
        # Verify that components were initialized with correct parameters
        assert pipeline.text_chunker.chunk_size == 800
        assert pipeline.text_chunker.overlap == 150
        assert pipeline.vector_storage.embedding_model == "custom-model"
        assert pipeline.vector_storage.collection_name == "custom-collection"
        assert pipeline.batch_size == 400
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    def test_process_batch_empty_list(self):
        """Test processing an empty batch of PDFs."""
        from rag.ingestion.pipeline import PDFIngestionPipeline
        
        pipeline = PDFIngestionPipeline()
        results = pipeline.process_batch([])
        
        assert results == []