from typing import List, Dict, Any, Optional
from chromadb.api.models.Collection import Collection

class RAGRetriever:
    """
    RAG (Retrieval-Augmented Generation) Retriever for fetching relevant documents
    from a ChromaDB vector database collection.
    """
    
    def __init__(self, collection: Collection):
        """
        Initialize the RAG retriever with a ChromaDB collection.
        
        Args:
            collection: ChromaDB Collection instance
        """
        self.collection = collection
    
    def retrieve(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        include: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on query embedding.
        
        Args:
            query_embedding: Vector embedding of the query
            n_results: Number of results to retrieve
            include: List of fields to include in results (metadatas, documents, distances)
        
        Returns:
            List of dictionaries containing retrieved documents with metadata and distances
        """
        if include is None:
            include = ['metadatas', 'documents', 'distances']
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )
        
        return self._format_results(results)
    

    def _format_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format ChromaDB query results into a standardized format.
        
        Args:
            raw_results: Raw results from ChromaDB query
        
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        if not raw_results['ids'] or not raw_results['ids'][0]:
            return []
        
        ids = raw_results['ids'][0]
        documents = raw_results.get('documents', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        
        for i, doc_id in enumerate(ids):
            result = {
                'id': doc_id,
                'document': documents[i] if i < len(documents) else None,
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'distance': distances[i] if i < len(distances) else None
            }
            formatted_results.append(result)
        
        return formatted_results
