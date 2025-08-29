import logging
from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from rag.retrieval.query_processor import QueryProcessor
from rag.retrieval.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class RetrievalToolInput(BaseModel):
    """Input schema for the RetrievalTool."""
    query: str = Field(description="The search query string to retrieve relevant documents")
    n_results: Optional[int] = Field(default=5, description="Number of results to retrieve")


class RetrievalTool(BaseTool):
    """
    A tool that provides document retrieval capabilities for agents.
    
    This tool combines query enhancement and vector-based document retrieval
    to find relevant documents. Returns raw retrieval results for downstream
    processing and synthesis.
    """
    
    name: str = "retrieval_tool"
    description: str = "Retrieves relevant documents for downstream synthesis. Combines query enhancement and vector-based document retrieval."
    args_schema: Type[BaseModel] = RetrievalToolInput
    
    query_processor: QueryProcessor = Field(description="QueryProcessor instance for query enhancement")
    retriever: RAGRetriever = Field(description="RAGRetriever instance for document retrieval")
    embeddings: Any = Field(default=None, description="OpenAI embeddings instance")
    
    def __init__(self, query_processor: QueryProcessor, retriever: RAGRetriever, 
                 embedding_model: str = "text-embedding-3-large", **kwargs):
        """
        Initialize the RetrievalTool.
        
        Args:
            query_processor: QueryProcessor instance for query enhancement
            retriever: RAGRetriever instance for document retrieval
            embedding_model: Optional OpenAI embedding model for generating query embeddings
            **kwargs: Additional arguments passed to BaseTool
            
        Raises:
            ValueError: If required dependencies are None
        """
        if query_processor is None:
            raise ValueError("query_processor is required")
        if retriever is None:
            raise ValueError("retriever is required")
        
        # Pass the required fields through the parent constructor
        super().__init__(
            query_processor=query_processor,
            retriever=retriever,
            embeddings=OpenAIEmbeddings(model=embedding_model),
            **kwargs
        )
    
    def _run(
        self,
        query: str,
        n_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous execution of the retrieval workflow.
        
        Args:
            query: The input query string
            n_results: Number of results to retrieve
            run_manager: Optional callback manager for tool runs
            
        Returns:
            List of raw retrieval result dictionaries from ChromaDB
        """
        # Since the original implementation was async, we'll raise an error for sync calls
        raise NotImplementedError("RetrievalTool only supports async execution. Use async invocation.")
    
    async def _arun(
        self,
        query: str,
        n_results_per_query: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the retrieval workflow: enhance query -> retrieve documents.
        
        Args:
            query: The input query string
            n_results: Number of results to retrieve
            run_manager: Optional callback manager for async tool runs
            
        Returns:
            Dictionary containing raw_results (list), enhanced_queries (list), and enhanced_query (str)
        """
        try:
            # Handle empty or None queries
            if not query or not query.strip():
                return {
                    "raw_results": [], 
                    "enhanced_queries": [],
                    "enhanced_query": ""
                }
            
            # Step 1: Enhance the query
            enhanced_queries = await self.query_processor.expand_query(query)
            
            # Step 2: Retrieve documents
            kwargs = {"n_results_per_query": n_results_per_query}
                
            raw_results = []
            for enhanced_query in enhanced_queries:
                res = await self._retrieve_documents(enhanced_query, **kwargs)
                raw_results.extend(res)
            
            # Return raw results and enhanced queries for downstream synthesis
            return {
                "raw_results": raw_results, 
                "enhanced_queries": enhanced_queries,
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval tool run: {e}")
            # Return empty results on error to maintain stability
            return {
                "raw_results": [], 
                "enhanced_queries": [],
            }
    
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
            n_results_per_query = kwargs.get('n_results_per_query', 5)
            
            # Generate embedding for the enhanced query
            query_embedding = await self._generate_query_embedding(enhanced_query)
            
            # Retrieve documents using the embedding
            return self.retriever.retrieve(
                query_embedding=query_embedding,
                n_results=n_results_per_query
            )
            
        except Exception as e:
            logger.info(f"Document retrieval failed: {e}")
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
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [
                "query_enhancement",
                "vector_search"
            ],
            "input_format": "text query string with optional parameters",
            "output_format": "list of raw document results",
            "args_schema": self.args_schema.schema() if self.args_schema else None
        }
