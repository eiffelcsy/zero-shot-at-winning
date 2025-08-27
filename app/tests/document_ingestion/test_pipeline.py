import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from rag.ingestion.text_chunker import ChunkWithMetadata


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
    

