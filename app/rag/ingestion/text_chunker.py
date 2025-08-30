import logging
from typing import List, Optional, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunk:
    """A text chunk with content and index."""
    
    def __init__(self, content: str, chunk_index: int = 0):
        self.content = content
        self.chunk_index = chunk_index
    
    def __str__(self) -> str:
        return self.content
    
    def __len__(self) -> int:
        return len(self.content)
    
    def __getitem__(self, key):
        """Make the object subscriptable for string operations."""
        return self.content[key]
    
    def __contains__(self, item):
        """Support 'in' operator for string containment checks."""
        return item in self.content


class TextChunker:
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"Initialized TextChunker with chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.debug("Empty or whitespace-only text provided, returning empty list")
            return []
        
        try:
            raw_chunks = self.splitter.split_text(text)
            
            chunks = []
            chunk_index = 0
            for chunk_content in raw_chunks:
                if chunk_content.strip():
                    chunk = TextChunk(content=chunk_content, chunk_index=chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
            
            logger.info(f"Successfully chunked text into {len(chunks)} chunks")
            logger.debug(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise ValueError(f"Text chunking failed: {e}")
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunked text.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }