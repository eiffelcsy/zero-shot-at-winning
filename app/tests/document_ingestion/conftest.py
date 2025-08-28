import pytest
import tempfile
import os
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from unittest.mock import Mock


class MockUploadFile:
    """Mock UploadFile object for testing."""
    
    def __init__(self, content: bytes, filename: str, content_type: str = "application/pdf"):
        self.file = BytesIO(content)
        self.filename = filename
        self.content_type = content_type
        self.size = len(content)


@pytest.fixture
def sample_pdf_upload():
    """Create a mock UploadFile with a simple PDF for testing."""
    # Create a simple PDF with some text using a temporary file approach
    # to ensure we get a completely valid PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        # Create PDF content
        p = canvas.Canvas(tmp_file.name, pagesize=letter)
        p.drawString(100, 750, "Sample PDF Document")
        p.drawString(100, 730, "This is a test document for the ingestion pipeline.")
        p.drawString(100, 710, "It contains multiple lines of text to test chunking.")
        p.drawString(100, 690, "Each line represents different content sections.")
        p.showPage()
        p.save()
        
        # Read the generated PDF file
        with open(tmp_file.name, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Clean up temp file
        os.unlink(tmp_file.name)
    
    # Create mock UploadFile
    return MockUploadFile(
        content=pdf_content,
        filename="sample_document.pdf",
        content_type="application/pdf"
    )


@pytest.fixture
def invalid_file_upload():
    """Create a mock UploadFile with non-PDF content for testing error cases."""
    content = b"This is not a PDF file"
    return MockUploadFile(
        content=content,
        filename="invalid_file.txt",
        content_type="text/plain"
    )


@pytest.fixture
def invalid_pdf_upload():
    """Create a mock UploadFile with PDF extension but invalid content."""
    content = b"This is not a PDF file but has .pdf extension"
    return MockUploadFile(
        content=content,
        filename="invalid_file.pdf",
        content_type="application/pdf"
    )


@pytest.fixture
def empty_pdf_upload():
    """Create a mock UploadFile with empty content."""
    return MockUploadFile(
        content=b"",
        filename="empty_file.pdf",
        content_type="application/pdf"
    )


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


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings client."""
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
    mock_embeddings.embed_query.return_value = [0.2, 0.3, 0.4, 0.5, 0.6]
    return mock_embeddings


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client and collection."""
    mock_collection = Mock()
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        'documents': [['test document 1', 'test document 2']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[{'source': 'test1.pdf'}, {'source': 'test2.pdf'}]],
        'ids': [['id1', 'id2']]
    }
    mock_collection.count.return_value = 2
    mock_collection.get.return_value = {'ids': ['id1', 'id2']}
    mock_collection.delete.return_value = None
    
    mock_client = Mock()
    mock_client.get_or_create_collection.return_value = mock_collection
    
    return mock_client


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    from rag.ingestion.text_chunker import TextChunk
    return [
        TextChunk("This is the first chunk of text."),
        TextChunk("This is the second chunk of text."),
        TextChunk("This is the third chunk from another document.")
    ]


@pytest.fixture
def sample_pdf_path():
    """Sample PDF file path for testing."""
    return "test_document.pdf"