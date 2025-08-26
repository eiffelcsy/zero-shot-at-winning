import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestPipeline:
    """Tests for end-to-end PDF ingestion pipeline."""
    
    @patch('app.pipeline.VectorStorage')
    @patch('app.pipeline.TextChunker')
    @patch('app.pipeline.PDFProcessor')
    def test_process_single_pdf(self, mock_pdf_processor, mock_text_chunker, mock_vector_storage, sample_pdf_path):
        """Test end-to-end processing of a single PDF."""
        from app.pipeline import PDFIngestionPipeline
        
        # Mock components
        mock_processor_instance = Mock()
        mock_processor_instance.load_pdf.return_value = "Sample PDF content"
        mock_processor_instance.extract_metadata.return_value = {"filename": "test.pdf"}
        mock_pdf_processor.return_value = mock_processor_instance
        
        mock_chunker_instance = Mock()
        mock_chunker_instance.chunk_text.return_value = ["chunk1", "chunk2"]
        mock_text_chunker.return_value = mock_chunker_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.generate_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_vector_storage.return_value = mock_storage_instance
        
        # Test pipeline
        pipeline = PDFIngestionPipeline()
        result = pipeline.process_pdf(sample_pdf_path)
        
        assert result["status"] == "success"
        assert result["chunks_processed"] == 2
        assert "filename" in result
        mock_storage_instance.store_chunks.assert_called_once()
    
    @patch('app.pipeline.VectorStorage')
    @patch('app.pipeline.TextChunker')
    @patch('app.pipeline.PDFProcessor')
    def test_process_multiple_pdfs(self, mock_pdf_processor, mock_text_chunker, mock_vector_storage):
        """Test processing a batch of PDFs."""
        from app.pipeline import PDFIngestionPipeline
        
        # Mock components
        mock_processor_instance = Mock()
        mock_processor_instance.load_pdf.return_value = "Sample content"
        mock_processor_instance.extract_metadata.return_value = {"filename": "test.pdf"}
        mock_pdf_processor.return_value = mock_processor_instance
        
        mock_chunker_instance = Mock()
        mock_chunker_instance.chunk_text.return_value = ["chunk1", "chunk2"]
        mock_text_chunker.return_value = mock_chunker_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.generate_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_vector_storage.return_value = mock_storage_instance
        
        # Test batch processing
        pipeline = PDFIngestionPipeline()
        pdf_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        results = pipeline.process_batch(pdf_paths)
        
        assert len(results) == 3
        assert all(result["status"] == "success" for result in results)
        assert mock_storage_instance.store_chunks.call_count == 3
    
    @patch('app.pipeline.PDFProcessor')
    def test_process_invalid_pdf(self, mock_pdf_processor):
        """Test pipeline handling of invalid PDF."""
        from app.pipeline import PDFIngestionPipeline
        
        # Mock processor to raise error
        mock_processor_instance = Mock()
        mock_processor_instance.load_pdf.side_effect = ValueError("Invalid file format")
        mock_pdf_processor.return_value = mock_processor_instance
        
        pipeline = PDFIngestionPipeline()
        result = pipeline.process_pdf("invalid.txt")
        
        assert result["status"] == "error"
        assert "Invalid file format" in result["error"]
    
    @patch('app.pipeline.VectorStorage')
    @patch('app.pipeline.TextChunker')
    @patch('app.pipeline.PDFProcessor')
    def test_pipeline_search_functionality(self, mock_pdf_processor, mock_text_chunker, mock_vector_storage):
        """Test pipeline search after ingestion."""
        from app.pipeline import PDFIngestionPipeline
        
        # Mock storage search
        mock_storage_instance = Mock()
        mock_storage_instance.search_similar.return_value = [
            {"document": "relevant chunk", "distance": 0.1, "metadata": {"source": "doc1.pdf"}}
        ]
        mock_vector_storage.return_value = mock_storage_instance
        
        pipeline = PDFIngestionPipeline()
        results = pipeline.search("test query")
        
        assert len(results) == 1
        assert results[0]["document"] == "relevant chunk"
        mock_storage_instance.search_similar.assert_called_once_with("test query")
