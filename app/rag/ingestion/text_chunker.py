import logging
from typing import List, Optional, Union, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkWithMetadata:
    """A text chunk with associated metadata."""
    
    def __init__(self, content: str, source_id: Optional[str] = None, chunk_index: int = 0):
        self.content = content
        self.source_id = source_id
        self.chunk_index = chunk_index
        self.metadata = {
            'source_id': source_id,
            'chunk_index': chunk_index,
            'chunk_length': len(content)
        }
    
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
    
    def chunk_text(self, text: str, source_id: Optional[str] = None) -> List[Union[ChunkWithMetadata, Dict[str, Any]]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: The text to be chunked
            source_id: Optional identifier for the source document
            
        Returns:
            List of text chunks with metadata
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            logger.debug("Empty or whitespace-only text provided, returning empty list")
            return []
        
        try:
            # Use LangChain's text splitter to create chunks
            raw_chunks = self.splitter.split_text(text)
            
            # Create chunks with metadata
            chunks = []
            for i, chunk_content in enumerate(raw_chunks):
                if chunk_content.strip():  # Only include non-empty chunks
                    chunk_with_metadata = ChunkWithMetadata(
                        content=chunk_content,
                        source_id=source_id,
                        chunk_index=i
                    )
                    chunks.append(chunk_with_metadata)
            
            logger.info(f"Successfully chunked text into {len(chunks)} chunks")
            logger.debug(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise ValueError(f"Text chunking failed: {e}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents from the PDF processor.
        
        Args:
            documents: List of documents with 'extracted_text' and 'metadata' keys
            
        Returns:
            List of chunked documents with metadata
        """
        chunked_documents = []
        
        for doc in documents:
            try:
                text = doc.get('extracted_text', '')
                metadata = doc.get('metadata', {})
                source_id = metadata.get('filename', 'unknown_document')
                
                # Chunk the text
                chunks = self.chunk_text(text, source_id=source_id)
                
                # Create document entries for each chunk
                for chunk in chunks:
                    chunked_doc = {
                        'content': chunk.content,
                        'metadata': {
                            **metadata,  # Include original document metadata
                            'chunk_index': chunk.chunk_index,
                            'chunk_length': len(chunk.content),
                            'source_id': source_id,
                            'original_document_length': len(text)
                        }
                    }
                    chunked_documents.append(chunked_doc)
                    
                logger.info(f"Chunked document {source_id} into {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('metadata', {}).get('filename', 'unknown')}: {e}")
                continue
        
        return chunked_documents
    
    def get_chunk_stats(self, chunks: List[Union[ChunkWithMetadata, Dict[str, Any]]]) -> Dict[str, Any]:
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
