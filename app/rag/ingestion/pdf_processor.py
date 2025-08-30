import os
import logging
import tempfile
from pathlib import Path
from typing import BinaryIO
import pymupdf4llm
import pymupdf
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Raised when PDF validation fails."""
    pass


class PDFProcessor:

    def __init__(self):
        """Initialize the PDF processor."""
        self.supported_extensions = {'.pdf'}
        self.max_file_size = 200 * 1024 * 1024
    
    def load_pdf(self, upload_file: UploadFile) -> str:
        """
        Load a PDF file and extract its text content from an uploaded file.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Extracted text content as string (markdown format)
            
        Raises:
            ValueError: If file format is invalid
            PDFValidationError: If PDF is corrupted or invalid
        """
        if upload_file.content_type != 'application/pdf':
            raise ValueError(f"Invalid file format. Expected PDF, got: {upload_file.content_type}")
        
        if upload_file.filename:
            file_ext = Path(upload_file.filename).suffix.lower()
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Invalid file format. Expected PDF, got: {file_ext}")
        
        try:
            file_content = upload_file.file.read()
            upload_file.file.seek(0)
            
            if len(file_content) == 0:
                raise PDFValidationError("PDF file is empty")
            
            if len(file_content) > self.max_file_size:
                raise PDFValidationError(f"PDF file too large: {len(file_content)} bytes (max: {self.max_file_size})")
            
            if not file_content.startswith(b'%PDF-'):
                raise ValueError("Invalid file format. File does not appear to be a valid PDF.")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, 'rb') as file:
                    return self._extract_text_from_file(file, upload_file.filename or "uploaded_file.pdf")
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            if isinstance(e, (ValueError, PDFValidationError)):
                raise
            raise PDFValidationError(f"Failed to process uploaded PDF: {e}")
    

    def _extract_text_from_file(self, file: BinaryIO, source_name: str) -> str:
        """Extract text from an open file object using pymupdf4llm."""
        try:
            file.seek(0)
            pdf_data = file.read()
            file.seek(0)
            
            pdf_doc = pymupdf.open(stream=pdf_data, filetype="pdf")
            
            if len(pdf_doc) == 0:
                pdf_doc.close()
                raise PDFValidationError("PDF contains no pages")
            
            if pdf_doc.needs_pass:
                pdf_doc.close()
                raise PDFValidationError("PDF is password-protected and cannot be processed")
            
            pdf_doc.close()
            
        except Exception as e:
            if isinstance(e, PDFValidationError):
                raise
            if "password-protected" in str(e) or "encrypted" in str(e) or "needs_pass" in str(e):
                raise PDFValidationError("PDF is password-protected and cannot be processed")
            elif "No pages" in str(e) or "pages" in str(e).lower():
                raise PDFValidationError("PDF contains no pages")
            else:
                raise PDFValidationError(f"PDF validation failed: {e}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name
        
        try:
            markdown_text = pymupdf4llm.to_markdown(temp_path)
            
            if not markdown_text or not markdown_text.strip():
                raise PDFValidationError("No extractable text found in PDF - may be image-based or corrupted")
            
            logger.info(f"Successfully extracted {len(markdown_text)} characters (markdown) from {source_name}")
            return markdown_text
            
        except Exception as e:
            if isinstance(e, PDFValidationError):
                raise
            raise PDFValidationError(f"Failed to extract text using pymupdf4llm: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)