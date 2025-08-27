import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from langchain_openai import OpenAIEmbeddings
from chroma.chroma_connection import get_chroma_client
from rag.ingestion.text_chunker import ChunkWithMetadata

logger = logging.getLogger(__name__)


class VectorStorage:
    
    def __init__(self, embedding_model: str = "text-embedding-3-large", collection_name: str = "rag_collection"):
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
        chunks: List[Union[ChunkWithMetadata, Dict[str, Any], str]], 
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Store text chunks with their embeddings in ChromaDB.
        
        Args:
            chunks: List of text chunks (can be ChunkWithMetadata objects, dicts, or strings)
            embeddings: Optional pre-computed embeddings. If None, will generate them
            metadatas: Optional metadata for each chunk. If None, will extract from chunks
            
        Returns:
            List of document IDs that were stored
        """
        if not chunks:
            logger.debug("No chunks provided, skipping storage")
            return []
        
        try:
            # Extract text content from chunks
            texts = []
            extracted_metadatas = []
            
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, ChunkWithMetadata):
                    texts.append(chunk.content)
                    extracted_metadatas.append(chunk.metadata)
                elif isinstance(chunk, dict):
                    texts.append(chunk.get('content', str(chunk)))
                    extracted_metadatas.append(chunk.get('metadata', {}))
                elif isinstance(chunk, str):
                    texts.append(chunk)
                    extracted_metadatas.append({'chunk_index': i})
                else:
                    texts.append(str(chunk))
                    extracted_metadatas.append({'chunk_index': i})
            
            # Use provided metadatas or extracted ones
            final_metadatas = metadatas if metadatas is not None else extracted_metadatas
            
            # Generate embeddings if not provided
            if embeddings is None:
                logger.info("No embeddings provided, generating them")
                embeddings = self.generate_embeddings(texts)
            
            # Validate dimensions match
            if len(texts) != len(embeddings) or len(texts) != len(final_metadatas):
                raise ValueError("Texts, embeddings, and metadatas must have the same length")
            
            # Generate unique IDs for each chunk
            document_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Store in ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=final_metadatas,
                ids=document_ids
            )
            
            logger.info(f"Successfully stored {len(texts)} chunks in ChromaDB")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise ValueError(f"Chunk storage failed: {e}")
    
    def search_similar(
        self, 
        query: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of dictionaries containing document, distance, and metadata
        """
        try:
            logger.info(f"Searching for similar chunks with query: '{query[:50]}...'")
            
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in ChromaDB
            search_kwargs = {
                'query_embeddings': [query_embedding],
                'n_results': n_results
            }
            
            if where:
                search_kwargs['where'] = where
            
            results = self.collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, distance, metadata) in enumerate(zip(
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )):
                    formatted_results.append({
                        'document': doc,
                        'distance': distance,
                        'metadata': metadata,
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise ValueError(f"Similarity search failed: {e}")
    
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
    
    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """
        Delete documents from the collection based on metadata filter.
        
        Args:
            where: Metadata filter to select documents for deletion
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get documents matching the filter
            results = self.collection.get(where=where)
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                logger.info(f"Deleted {deleted_count} documents matching filter: {where}")
                return deleted_count
            else:
                logger.info(f"No documents found matching filter: {where}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise ValueError(f"Document deletion failed: {e}")
    
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
