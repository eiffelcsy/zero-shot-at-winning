import logging
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from rag.tools.base_tool import BaseTool
from rag.retrieval.query_processor import QueryProcessor
from rag.retrieval.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class RetrievalTool(BaseTool):
    """
    A tool that provides document retrieval capabilities for agents.
    
    This tool combines query enhancement and vector-based document retrieval
    to find relevant documents. Returns raw retrieval results for downstream
    processing and synthesis.
    """
    
    def __init__(self, query_processor: QueryProcessor, retriever: RAGRetriever, 
                 embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the RetrievalTool.
        
        Args:
            query_processor: QueryProcessor instance for query enhancement
            retriever: RAGRetriever instance for document retrieval
            embedding_model: Optional OpenAI embedding model for generating query embeddings
            
        Raises:
            ValueError: If required dependencies are None
        """
        super().__init__("retrieval_tool")
        
        if query_processor is None:
            raise ValueError("query_processor is required")
        if retriever is None:
            raise ValueError("retriever is required")
            
        self.query_processor = query_processor
        self.retriever = retriever
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
    
    async def run(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the retrieval workflow: enhance query -> retrieve documents.
        
        Args:
            query: The input query string
            **kwargs: Additional parameters (e.g., n_results, metadata_filter)
            
        Returns:
            List of raw retrieval result dictionaries from ChromaDB
        """
        try:
            # Handle empty or None queries
            if not query or not query.strip():
                return []
            
            # Step 1: Enhance the query
            enhanced_query = await self._enhance_query(query)
            
            # Step 2: Retrieve documents
            raw_results = await self._retrieve_documents(enhanced_query, **kwargs)
            
            # Return raw results for downstream synthesis
            return raw_results
            
        except Exception as e:
            logger.error(f"Error in retrieval tool run: {e}")
            # Return empty results on error to maintain stability
            return []
    
    async def _enhance_query(self, query: str) -> str:
        """
        Enhance the query using the query processor.
        
        Args:
            query: Original query string
            
        Returns:
            Enhanced query string
        """
        try:
            return await self.query_processor.enhance_query(query)
        except Exception as e:
            logger.warning(f"Query enhancement failed, using original query: {e}")
            return query
    
    async def _retrieve_documents(self, enhanced_query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using the enhanced query.
        
        Args:
            enhanced_query: The enhanced query string
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of raw retrieval results
        """
        try:
            n_results = kwargs.get('n_results', 5)
            metadata_filter = kwargs.get('metadata_filter')
            
            # Generate embedding for the enhanced query
            query_embedding = await self._generate_query_embedding(enhanced_query)
            
            # Retrieve documents using the embedding
            if metadata_filter:
                return self.retriever.retrieve_with_metadata_filter(
                    query_embedding=query_embedding,
                    metadata_filter=metadata_filter,
                    n_results=n_results
                )
            else:
                return self.retriever.retrieve(
                    query_embedding=query_embedding,
                    n_results=n_results
                )
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for the query string.
        
        Args:
            query: The query string to embed
            
        Returns:
            List of floats representing the query embedding
        """
        try:
            return self.embeddings.embed_query(query)
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    

    
    def validate_inputs(self, query: str = None, **kwargs) -> bool:
        """
        Validate tool inputs.
        
        Args:
            query: Query string to validate
            **kwargs: Additional parameters
            
        Returns:
            True if inputs are valid
        """
        # For MVP, we just check that query is provided and is a string
        if query is None:
            return False
        
        if not isinstance(query, str):
            return False
            
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this tool.
        
        Returns:
            Dictionary containing tool metadata
        """
        base_info = super().get_info()
        base_info.update({
            "description": "Retrieves relevant documents for downstream synthesis",
            "capabilities": [
                "query_enhancement",
                "vector_search"
            ],
            "input_format": "text query string",
            "output_format": "list of raw document results"
        })
        return base_info
