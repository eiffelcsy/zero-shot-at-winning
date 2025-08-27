import os
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, BinaryIO
import pypdf
from pypdf.errors import PdfReadError
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Raised when PDF validation fails."""
    pass


class PDFProcessor:

    def __init__(self):
        """Initialize the PDF processor."""
        self.supported_extensions = {'.pdf'}
        self.max_file_size = 200 * 1024 * 1024  # 200MB default limit
    
    def load_pdf(self, upload_file: UploadFile) -> str:
        """
        Load a PDF file and extract its text content from an uploaded file.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Extracted text content as string
            
        Raises:
            ValueError: If file format is invalid
            PDFValidationError: If PDF is corrupted or invalid
        """
        return self._load_pdf_from_upload(upload_file)
    

    def _load_pdf_from_upload(self, upload_file: UploadFile) -> str:
        """Load PDF from FastAPI UploadFile."""
        # Validate content type
        if upload_file.content_type != 'application/pdf':
            raise ValueError(f"Invalid file format. Expected PDF, got: {upload_file.content_type}")
        
        # Validate filename extension
        if upload_file.filename:
            file_ext = Path(upload_file.filename).suffix.lower()
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Invalid file format. Expected PDF, got: {file_ext}")
        
        try:
            # Read file content
            file_content = upload_file.file.read()
            upload_file.file.seek(0)  # Reset file pointer for potential future reads
            
            # Validate file size
            if len(file_content) == 0:
                raise PDFValidationError("PDF file is empty")
            
            if len(file_content) > self.max_file_size:
                raise PDFValidationError(f"PDF file too large: {len(file_content)} bytes (max: {self.max_file_size})")
            
            # Validate PDF header
            if not file_content.startswith(b'%PDF-'):
                raise ValueError("Invalid file format. File does not appear to be a valid PDF.")
            
            # Create temporary file for pypdf processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                # Process using temporary file
                with open(temp_path, 'rb') as file:
                    return self._extract_text_from_file(file, upload_file.filename or "uploaded_file.pdf")
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except PdfReadError as e:
            raise PDFValidationError(f"PDF is corrupted or invalid: {e}")
        except Exception as e:
            raise PDFValidationError(f"Failed to process uploaded PDF: {e}")
    
    def _extract_text_from_file(self, file: BinaryIO, source_name: str) -> str:
        """Extract text from an open file object."""
        pdf_reader = pypdf.PdfReader(file)
        
        # Check if PDF is text-extractable
        if len(pdf_reader.pages) == 0:
            raise PDFValidationError("PDF contains no pages")
        
        # Check if encrypted
        if pdf_reader.is_encrypted:
            raise PDFValidationError("PDF is password-protected and cannot be processed")
        
        # Extract text from all pages
        text_content = []
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_content.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        # Join all text content
        full_text = '\n\n'.join(text_content)
        
        # Validate that we extracted meaningful text
        if not full_text.strip():
            raise PDFValidationError("No extractable text found in PDF - may be image-based or corrupted")
        
        logger.info(f"Successfully extracted {len(full_text)} characters from {source_name}")
        return full_text
    
    def extract_metadata(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Extract metadata from an uploaded PDF file.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Dictionary containing metadata
            
        Raises:
            PDFValidationError: If PDF is corrupted or invalid
        """
        return self._extract_metadata_from_upload(upload_file)
    

    def _extract_metadata_from_upload(self, upload_file: UploadFile) -> Dict[str, Any]:
        """Extract metadata from uploaded file."""
        # Read file content to get size
        file_content = upload_file.file.read()
        upload_file.file.seek(0)  # Reset file pointer
        
        metadata = {
            'filename': upload_file.filename or 'uploaded_file.pdf',
            'date_processed': datetime.now().isoformat(),
            'content_type': upload_file.content_type or 'application/pdf'
        }
        
        # Extract PDF-specific metadata using temporary file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, 'rb') as file:
                    self._extract_pdf_metadata(file, metadata)
                    logger.info(f"Extracted metadata from upload {upload_file.filename}: {metadata['page_count']} pages")
            finally:
                os.unlink(temp_path)
                
        except PdfReadError as e:
            raise PDFValidationError(f"PDF is corrupted or invalid: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata from upload: {e}")
            # Continue with basic file metadata
        
        return metadata
    
    def _extract_pdf_metadata(self, file: BinaryIO, metadata: Dict[str, Any]) -> None:
        """Extract PDF-specific metadata from an open file."""
        pdf_reader = pypdf.PdfReader(file)
        
        # Basic PDF properties
        metadata.update({
            'page_count': len(pdf_reader.pages),
            'is_encrypted': pdf_reader.is_encrypted,
        })
        
        # Document information if available
        if pdf_reader.metadata:
            pdf_info = pdf_reader.metadata
            # Convert PyPDF TextStringObject types to regular strings for ChromaDB compatibility
            metadata.update({
                'title': str(pdf_info.get('/Title', '')) if pdf_info.get('/Title') else '',
                'author': str(pdf_info.get('/Author', '')) if pdf_info.get('/Author') else '',
                'subject': str(pdf_info.get('/Subject', '')) if pdf_info.get('/Subject') else '',
                'creator': str(pdf_info.get('/Creator', '')) if pdf_info.get('/Creator') else '',
                'producer': str(pdf_info.get('/Producer', '')) if pdf_info.get('/Producer') else '',
            })
    

    
    def validate_upload(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Perform complete validation of an uploaded PDF file.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Dictionary containing validation results and basic metadata
            
        Raises:
            ValueError: If file format is invalid
            PDFValidationError: If PDF is corrupted or invalid
        """
        # Get basic metadata (includes validation)
        metadata = self.extract_metadata(upload_file)
        
        # Add validation status
        validation_result = {
            'is_valid': True,
            'validation_timestamp': datetime.now().isoformat(),  # Convert to string for ChromaDB compatibility
            'file_metadata': metadata,
        }
        
        logger.info(f"PDF validation successful: {upload_file.filename}")
        return validation_result
    
    def process_upload(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Complete processing workflow for uploaded PDF files.
        
        This is a convenience method that combines validation, text extraction,
        and metadata extraction for use in FastAPI endpoints.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Dictionary containing:
            - extracted_text: The PDF text content
            - metadata: File and PDF metadata
            - validation_result: Validation status and timestamp
            
        Raises:
            ValueError: If file format is invalid
            PDFValidationError: If PDF is corrupted or invalid
        """
        try:
            # Extract text (includes validation)
            extracted_text = self.load_pdf(upload_file)
            
            # Get metadata
            metadata = self.extract_metadata(upload_file)
            
            # Create comprehensive result
            result = {
                'extracted_text': extracted_text,
                'metadata': metadata,
                'validation_result': {
                    'is_valid': True,
                    'validation_timestamp': datetime.now().isoformat(),  # Convert to string for ChromaDB compatibility
                    'text_length': len(extracted_text),
                    'processing_status': 'success'
                }
            }
            
            logger.info(f"Successfully processed upload: {upload_file.filename}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process upload {upload_file.filename}: {e}")
            raise