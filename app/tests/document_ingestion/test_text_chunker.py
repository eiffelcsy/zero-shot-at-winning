import pytest


class TestTextChunker:
    """Tests for text chunker component."""
    
    def test_chunk_text(self, sample_text):
        """Test splitting text into fixed-size chunks with overlap."""
        from rag.ingestion.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + some tolerance
        assert len(chunks[0]) > 0
    
    def test_chunk_returns_textchunk_objects(self, sample_text):
        """Test that chunks are TextChunk objects."""
        from rag.ingestion.text_chunker import TextChunker, TextChunk
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(hasattr(chunk, 'content') for chunk in chunks)
    
    def test_empty_text_chunking(self):
        """Test chunking empty or whitespace-only text."""
        from rag.ingestion.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        
        empty_chunks = chunker.chunk_text("")
        whitespace_chunks = chunker.chunk_text("   \n\t  ")
        
        assert empty_chunks == []
        assert whitespace_chunks == []
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        from rag.ingestion.text_chunker import TextChunker
        
        text = "A" * 200  # Simple repeated text
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        # Check that there's some overlap between consecutive chunks
        if len(chunks) > 1:
            chunk1_content = chunks[0].content
            chunk2_content = chunks[1].content
            assert chunk1_content[-10:] in chunk2_content or chunk2_content[:10] in chunk1_content
    
    def test_textchunk_string_operations(self, sample_text):
        """Test that TextChunk objects support string operations."""
        from rag.ingestion.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        if chunks:
            chunk = chunks[0]
            
            # Test string conversion
            assert str(chunk) == chunk.content
            
            # Test length
            assert len(chunk) == len(chunk.content)
            
            # Test subscripting
            assert chunk[0] == chunk.content[0]
            assert chunk[:5] == chunk.content[:5]
            
            # Test containment
            if len(chunk.content) > 5:
                substring = chunk.content[:5]
                assert substring in chunk
    
    def test_chunk_stats(self, sample_text):
        """Test chunk statistics calculation."""
        from rag.ingestion.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        stats = chunker.get_chunk_stats(chunks)
        
        assert isinstance(stats, dict)
        assert 'total_chunks' in stats
        assert 'average_chunk_size' in stats
        assert 'min_chunk_size' in stats
        assert 'max_chunk_size' in stats
        assert 'total_characters' in stats
        
        assert stats['total_chunks'] == len(chunks)
        assert stats['total_characters'] == sum(len(chunk) for chunk in chunks)
        
        if chunks:
            assert stats['min_chunk_size'] > 0
            assert stats['max_chunk_size'] >= stats['min_chunk_size']
            assert stats['average_chunk_size'] > 0
    
    def test_chunk_stats_empty_input(self):
        """Test chunk statistics with empty input."""
        from rag.ingestion.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, overlap=20)
        stats = chunker.get_chunk_stats([])
        
        assert stats['total_chunks'] == 0
        assert stats['average_chunk_size'] == 0
        assert stats['min_chunk_size'] == 0
        assert stats['max_chunk_size'] == 0
        assert stats['total_characters'] == 0
    
    def test_chunker_initialization(self):
        """Test TextChunker initialization with custom parameters."""
        from rag.ingestion.text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=500, overlap=100)
        
        assert chunker.chunk_size == 500
        assert chunker.overlap == 100
        assert hasattr(chunker, 'splitter')
    
    def test_long_text_chunking(self):
        """Test chunking very long text."""
        from rag.ingestion.text_chunker import TextChunker
        
        # Create a long text
        long_text = "This is a test sentence. " * 100  # ~2500 characters
        
        chunker = TextChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk_text(long_text)
        
        assert len(chunks) > 5  # Should create multiple chunks
        assert all(len(chunk) <= 250 for chunk in chunks)  # Reasonable size limit
        assert all(len(chunk.content.strip()) > 0 for chunk in chunks)  # No empty chunks