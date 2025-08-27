import pytest
from unittest.mock import Mock, patch
from datetime import datetime
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
    
    def test_extract_basic_metadata(self, sample_pdf_upload):
        """Test extraction of basic metadata from uploaded PDF."""
        processor = PDFProcessor()
        metadata = processor.extract_metadata(sample_pdf_upload)
        
        assert "filename" in metadata
        assert "date_processed" in metadata
        assert "content_type" in metadata
        assert metadata["filename"] == "sample_document.pdf"
        assert metadata["content_type"] == "application/pdf"
        assert isinstance(metadata["date_processed"], datetime)
    
    def test_extract_pdf_specific_metadata(self, sample_pdf_upload):
        """Test extraction of PDF-specific metadata."""
        processor = PDFProcessor()
        metadata = processor.extract_metadata(sample_pdf_upload)
        
        assert "page_count" in metadata
        assert "is_encrypted" in metadata
        assert metadata["page_count"] == 1
        assert metadata["is_encrypted"] is False
    
    def test_validate_upload_success(self, sample_pdf_upload):
        """Test successful validation of uploaded PDF."""
        processor = PDFProcessor()
        result = processor.validate_upload(sample_pdf_upload)
        
        assert "is_valid" in result
        assert "validation_timestamp" in result
        assert "file_metadata" in result
        assert result["is_valid"] is True
        assert isinstance(result["validation_timestamp"], datetime)
    
    def test_validate_upload_failure(self, invalid_file_upload):
        """Test validation failure for invalid uploads."""
        processor = PDFProcessor()
        
        with pytest.raises((ValueError, PDFValidationError)):
            processor.validate_upload(invalid_file_upload)
    
    def test_process_upload_complete_workflow(self, sample_pdf_upload):
        """Test complete processing workflow for uploaded PDF."""
        processor = PDFProcessor()
        result = processor.process_upload(sample_pdf_upload)
        
        assert "extracted_text" in result
        assert "metadata" in result
        assert "validation_result" in result
        
        # Check extracted text
        assert isinstance(result["extracted_text"], str)
        assert len(result["extracted_text"]) > 0
        assert "Sample PDF Document" in result["extracted_text"]
        
        # Check metadata
        assert result["metadata"]["filename"] == "sample_document.pdf"
        assert result["metadata"]["source_type"] == "upload"
        
        # Check validation result
        assert result["validation_result"]["is_valid"] is True
        assert result["validation_result"]["processing_status"] == "success"
        assert "text_length" in result["validation_result"]
    
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
