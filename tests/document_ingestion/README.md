# PDF Ingestion Pipeline Tests

This directory contains test-driven development tests for the PDF ingestion pipeline.

## Test Structure

### 1. PDF Processor Tests (`test_pdf_processor.py`)
- `test_load_pdf()` - Loads PDF and extracts text
- `test_reject_invalid_file()` - Rejects non-PDF files  
- `test_extract_basic_metadata()` - Gets filename and date

### 2. Text Chunker Tests (`test_text_chunker.py`)
- `test_chunk_text()` - Splits text into fixed-size chunks with overlap
- `test_chunk_preserves_source()` - Chunks remember source document

### 3. Vector Storage Tests (`test_vector_storage.py`)
- `test_generate_embeddings()` - Creates embeddings for text chunks
- `test_store_in_chromadb()` - Saves chunks and embeddings to database
- `test_search_similar()` - Finds similar chunks by query

### 4. Pipeline Tests (`test_pipeline.py`)
- `test_process_single_pdf()` - End-to-end test processing one PDF
- `test_process_multiple_pdfs()` - Processes a batch of PDFs

## Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_pdf_processor.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_pdf_processor.py::TestPDFProcessor::test_load_pdf
```

## Implementation Guide

The tests define the interfaces that need to be implemented:

1. **PDFProcessor** (`app/pdf_processor.py`)
   - `load_pdf(file_path: str) -> str`
   - `extract_metadata(file_path: str) -> dict`

2. **TextChunker** (`app/text_chunker.py`)
   - `chunk_text(text: str, source_id: str = None) -> list`

3. **VectorStorage** (`app/vector_storage.py`)
   - `generate_embeddings(chunks: list) -> list`
   - `store_chunks(chunks: list, embeddings: list, metadatas: list)`
   - `search_similar(query: str, n_results: int = 5) -> list`

4. **PDFIngestionPipeline** (`app/pipeline.py`)
   - `process_pdf(file_path: str) -> dict`
   - `process_batch(file_paths: list) -> list`
   - `search(query: str) -> list`

## Next Steps

1. Run the tests (they will fail initially)
2. Implement the classes and methods to make tests pass
3. Start with the simplest components first (PDFProcessor)
4. Use mocks for external dependencies (ChromaDB, embeddings)
5. Focus on making tests pass, then refactor for quality
