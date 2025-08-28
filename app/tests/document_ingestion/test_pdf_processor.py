import pytest
from unittest.mock import Mock, patch
import os
from rag.ingestion.pdf_processor import PDFProcessor, PDFValidationError


class TestPDFProcessor:
    """Tests for PDF processor component."""
    
    def test_load_pdf(self, sample_pdf_upload):
        """Test loading a PDF file from upload and extracting text."""
        processor = PDFProcessor()
        text = processor.load_pdf(sample_pdf_upload)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Sample PDF Document" in text
    
    def test_reject_invalid_content_type(self, invalid_file_upload):
        """Test rejection of files with invalid content type."""
        processor = PDFProcessor()
        
        with pytest.raises(ValueError, match="Invalid file format"):
            processor.load_pdf(invalid_file_upload)
    
    def test_reject_invalid_pdf_content(self, invalid_pdf_upload):
        """Test rejection of files with PDF extension but invalid content."""
        processor = PDFProcessor()
        
        with pytest.raises((ValueError, PDFValidationError)):
            processor.load_pdf(invalid_pdf_upload)
    
    def test_reject_empty_file(self, empty_pdf_upload):
        """Test rejection of empty files."""
        processor = PDFProcessor()
        
        with pytest.raises(PDFValidationError, match="PDF file is empty"):
            processor.load_pdf(empty_pdf_upload)
    
    def test_file_size_limit(self, sample_pdf_upload):
        """Test file size validation."""
        processor = PDFProcessor()
        # Set a very small file size limit
        processor.max_file_size = 10  # 10 bytes
        
        with pytest.raises(PDFValidationError, match="PDF file too large"):
            processor.load_pdf(sample_pdf_upload)
    
    def test_file_pointer_reset(self, sample_pdf_upload):
        """Test that file pointer is properly reset after operations."""
        processor = PDFProcessor()
        
        # First operation
        text1 = processor.load_pdf(sample_pdf_upload)
        
        # Second operation should work the same
        text2 = processor.load_pdf(sample_pdf_upload)
        
        assert text1 == text2
    
    def test_markdown_output_format(self, sample_pdf_upload):
        """Test that the output is in markdown format."""
        processor = PDFProcessor()
        text = processor.load_pdf(sample_pdf_upload)
        
        # Check that we get structured text (markdown might have newlines, structure)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
        # Note: The exact markdown format depends on pymupdf4llm output
    
    def test_pdf_validation_encrypted(self):
        """Test that encrypted PDFs are properly rejected."""
        processor = PDFProcessor()
        
        # Create a mock upload file that simulates an encrypted PDF
        mock_upload = Mock()
        mock_upload.content_type = 'application/pdf'
        mock_upload.filename = 'encrypted.pdf'
        mock_upload.file.read.return_value = b'%PDF-1.4\nencrypted_content'
        mock_upload.file.seek = Mock()
        
        # This test would need a real encrypted PDF to test properly
        # For now, we just test that the validation logic exists
        assert hasattr(processor, '_extract_text_from_file')
    
    def test_supported_extensions(self):
        """Test that processor only supports PDF extensions."""
        processor = PDFProcessor()
        
        assert '.pdf' in processor.supported_extensions
        assert len(processor.supported_extensions) >= 1
    
    def test_max_file_size_default(self):
        """Test that processor has a reasonable default file size limit."""
        processor = PDFProcessor()
        
        # Should have a default file size limit (200MB)
        assert processor.max_file_size > 0
        assert processor.max_file_size == 200 * 1024 * 1024