import pytest


class TestTextChunker:
    """Tests for text chunker component."""
    
    def test_chunk_text(self, sample_text):
        """Test splitting text into fixed-size chunks with overlap."""
        from app.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + some tolerance
        assert len(chunks[0]) > 0
    
    def test_chunk_preserves_source(self, sample_text):
        """Test that chunks remember their source document."""
        from app.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        source_id = "test_document.pdf"
        chunks = chunker.chunk_text(sample_text, source_id=source_id)
        
        assert all(hasattr(chunk, 'source_id') or 
                  (isinstance(chunk, dict) and 'source_id' in chunk) 
                  for chunk in chunks)
    
    def test_empty_text_chunking(self):
        """Test chunking empty or whitespace-only text."""
        from app.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        
        empty_chunks = chunker.chunk_text("")
        whitespace_chunks = chunker.chunk_text("   \n\t  ")
        
        assert empty_chunks == []
        assert whitespace_chunks == []
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        from app.text_chunker import TextChunker
        
        text = "A" * 200  # Simple repeated text
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        # Check that there's some overlap between consecutive chunks
        if len(chunks) > 1:
            assert chunks[0][-10:] in chunks[1] or chunks[1][:10] in chunks[0]
