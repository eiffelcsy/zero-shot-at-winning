import pytest
import tempfile
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


@pytest.fixture
def sample_pdf_path():
    """Create a simple PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        # Create a simple PDF with some text
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(100, 750, "Sample PDF Document")
        p.drawString(100, 730, "This is a test document for the ingestion pipeline.")
        p.drawString(100, 710, "It contains multiple lines of text to test chunking.")
        p.drawString(100, 690, "Each line represents different content sections.")
        p.showPage()
        p.save()
        
        # Write to temp file
        buffer.seek(0)
        tmp_file.write(buffer.getvalue())
        tmp_file.flush()
        
        yield tmp_file.name
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


@pytest.fixture
def invalid_file_path():
    """Create a non-PDF file for testing error cases."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(b"This is not a PDF file")
        tmp_file.flush()
        
        yield tmp_file.name
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


@pytest.fixture
def sample_text():
    """Sample text for chunking tests."""
    return """This is a long document that needs to be chunked into smaller pieces. 
    The chunking algorithm should split this text while maintaining context and overlap.
    Each chunk should have a reasonable size and preserve readability.
    The original source document information should be maintained throughout the process.
    This text is long enough to create multiple chunks for testing purposes."""


@pytest.fixture
def mock_embeddings():
    """Mock embedding vectors for testing."""
    return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(3)]
