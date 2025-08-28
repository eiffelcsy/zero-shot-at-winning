import logging
import uuid
from typing import List, Optional, Union, Dict, Any
from langchain_openai import OpenAIEmbeddings
from chroma.chroma_connection import get_chroma_client
from rag.ingestion.text_chunker import TextChunk

logger = logging.getLogger(__name__)


class VectorStorage:
    
    def __init__(self, embedding_model: str = "text-embedding-3-large", collection_name: str = "regulation_kb"):
        """
        Initialize the VectorStorage with OpenAI embeddings and ChromaDB.
        
        Args:
            embedding_model: OpenAI embedding model to use
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize ChromaDB client and collection
        self.client = get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document chunks with embeddings"}
        )
        
        logger.info(f"Initialized VectorStorage with model {embedding_model} and collection {collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI embedding model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.debug("Empty text list provided, returning empty embeddings")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise ValueError(f"Embedding generation failed: {e}")
    
    def store_chunks(
        self, 
        chunks: List[Union[TextChunk, str]], 
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 300
    ) -> List[str]:
        """
        Store text chunks with their embeddings in ChromaDB in batches.
        
        Args:
            chunks: List of text chunks (can be TextChunk objects or strings)
            embeddings: Optional pre-computed embeddings. If None, will generate them
            metadata: Optional metadata dictionary containing regulation_name, geo_jurisdiction, etc.
            batch_size: Number of records to send to ChromaDB per batch (default: 300)
            
        Returns:
            List of document IDs that were stored
        """
        if not chunks:
            logger.debug("No chunks provided, skipping storage")
            return []
        
        try:
            # Extract text content and prepare metadata for chunks
            texts = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                chunk_text = ""
                chunk_index = i  # Default chunk index
                
                if isinstance(chunk, TextChunk):
                    chunk_text = chunk.content
                    chunk_index = chunk.chunk_index
                elif isinstance(chunk, str):
                    chunk_text = chunk
                else:
                    chunk_text = str(chunk)
                
                texts.append(chunk_text)
                
                # Prepare metadata for this chunk
                chunk_meta = {
                    'chunk_index': chunk_index
                }
                
                # Add document-level metadata if provided
                if metadata:
                    chunk_meta.update(metadata)
                    logger.debug(f"Added metadata to chunk {chunk_index}: {chunk_meta}")
                else:
                    logger.debug(f"No metadata provided for chunk {chunk_index}")
                
                chunk_metadata.append(chunk_meta)
            
            # Generate embeddings if not provided
            if embeddings is None:
                embeddings = self.generate_embeddings(texts)
            
            # Validate dimensions match
            if len(texts) != len(embeddings):
                raise ValueError("Texts and embeddings must have the same length")
            
            # Generate unique IDs for each chunk
            document_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Store in ChromaDB using batching
            all_stored_ids = []
            total_chunks = len(texts)
            
            if total_chunks <= batch_size:
                # No need for batching
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    ids=document_ids,
                    metadatas=chunk_metadata
                )
                all_stored_ids = document_ids
                logger.info(f"Stored {total_chunks} chunks in single batch")
            else:
                # Process in batches
                num_batches = (total_chunks + batch_size - 1) // batch_size  # Ceiling division
                logger.info(f"Storing {total_chunks} chunks in {num_batches} batches of {batch_size}")
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_chunks)
                    
                    batch_texts = texts[start_idx:end_idx]
                    batch_embeddings = embeddings[start_idx:end_idx]
                    batch_ids = document_ids[start_idx:end_idx]
                    batch_metadata = chunk_metadata[start_idx:end_idx]
                    
                    try:
                        self.collection.add(
                            documents=batch_texts,
                            embeddings=batch_embeddings,
                            ids=batch_ids,
                            metadatas=batch_metadata
                        )
                        all_stored_ids.extend(batch_ids)
                        logger.info(f"Successfully stored batch {batch_idx + 1}/{num_batches} "
                                  f"({len(batch_texts)} chunks)")
                        
                    except Exception as batch_error:
                        logger.error(f"Failed to store batch {batch_idx + 1}/{num_batches}: {batch_error}")
                        # Continue with next batch instead of failing completely
                        continue
            
            logger.info(f"Successfully stored {len(all_stored_ids)} chunks total")
            return all_stored_ids
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise ValueError(f"Chunk storage failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'embedding_model': self.embedding_model
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'total_documents': 0,
                'embedding_model': self.embedding_model,
                'error': str(e)
            }
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                deleted_count = len(all_docs['ids'])
                logger.info(f"Cleared {deleted_count} documents from collection")
            else:
                logger.info("Collection was already empty")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False