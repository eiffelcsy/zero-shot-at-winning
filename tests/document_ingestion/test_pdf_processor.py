import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import os


class TestPDFProcessor:
    """Tests for PDF processor component."""
    
    def test_load_pdf(self, sample_pdf_path):
        """Test loading a PDF file and extracting text."""
        from app.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        text = processor.load_pdf(sample_pdf_path)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Sample PDF Document" in text
    
    def test_reject_invalid_file(self, invalid_file_path):
        """Test rejection of non-PDF files."""
        from app.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        with pytest.raises(ValueError, match="Invalid file format"):
            processor.load_pdf(invalid_file_path)
    
    def test_extract_basic_metadata(self, sample_pdf_path):
        """Test extraction of basic metadata from PDF."""
        from app.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        metadata = processor.extract_metadata(sample_pdf_path)
        
        assert "filename" in metadata
        assert "date_processed" in metadata
        assert metadata["filename"] == os.path.basename(sample_pdf_path)
        assert isinstance(metadata["date_processed"], datetime)
    
    def test_load_nonexistent_file(self):
        """Test handling of nonexistent files."""
        from app.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load_pdf("nonexistent.pdf")
