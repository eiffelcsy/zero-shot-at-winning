import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any


class TestRAGRetriever:
    """Tests for RAGRetriever component."""
    
    def setup_method(self):
        """Setup test fixtures for each test method."""
        self.mock_collection = Mock()
        
        # Sample ChromaDB query response
        self.sample_query_response = {
            'ids': [['doc_1', 'doc_2']],
            'documents': [['COPPA requires parental consent for children under 13.', 
                          'California SB-976 prohibits algorithmic recommendations for minors.']],
            'metadatas': [[
                {'source': 'COPPA_regulations.pdf', 'regulation': 'US_COPPA', 'jurisdiction': 'US'},
                {'source': 'CA_SB976.pdf', 'regulation': 'CA_SB976', 'jurisdiction': 'CA'}
            ]],
            'distances': [[0.1, 0.3]]
        }
        
        # Sample query embedding
        self.sample_query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def test_init_with_collection(self):
        """Test RAGRetriever initialization with ChromaDB collection."""
        from rag.retrieval.retriever import RAGRetriever
        
        # Initialize retriever with collection
        retriever = RAGRetriever(collection=self.mock_collection)
        
        # Assertions
        assert retriever.collection == self.mock_collection
    
    def test_retrieve_basic(self):
        """Test basic retrieve functionality."""
        from rag.retrieval.retriever import RAGRetriever
        
        # Setup mock collection response
        self.mock_collection.query.return_value = self.sample_query_response
        
        # Initialize retriever
        retriever = RAGRetriever(collection=self.mock_collection)
        
        # Perform retrieval
        results = retriever.retrieve(
            query_embedding=self.sample_query_embedding,
            n_results=2
        )
        
        # Assertions
        assert len(results) == 2
        assert results[0]['id'] == 'doc_1'
        assert results[0]['document'] == 'COPPA requires parental consent for children under 13.'
        assert results[0]['metadata']['regulation'] == 'US_COPPA'
        assert results[0]['distance'] == 0.1
        
        assert results[1]['id'] == 'doc_2'
        assert results[1]['document'] == 'California SB-976 prohibits algorithmic recommendations for minors.'
        assert results[1]['metadata']['regulation'] == 'CA_SB976'
        assert results[1]['distance'] == 0.3
        
        # Verify collection.query was called correctly
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[self.sample_query_embedding],
            n_results=2,
            include=['metadatas', 'documents', 'distances']
        )
    
    def test_retrieve_with_metadata_filter(self):
        """Test retrieve with metadata filtering."""
        from rag.retrieval.retriever import RAGRetriever
        
        # Setup mock collection response with filtered results
        filtered_response = {
            'ids': [['doc_1']],
            'documents': [['COPPA requires parental consent for children under 13.']],
            'metadatas': [[{'source': 'COPPA_regulations.pdf', 'regulation': 'US_COPPA', 'jurisdiction': 'US'}]],
            'distances': [[0.15]]
        }
        self.mock_collection.query.return_value = filtered_response
        
        # Initialize retriever
        retriever = RAGRetriever(collection=self.mock_collection)
        
        # Search with metadata filter
        metadata_filter = {'jurisdiction': 'US'}
        results = retriever.retrieve_with_metadata_filter(
            query_embedding=self.sample_query_embedding,
            metadata_filter=metadata_filter,
            n_results=5
        )
        
        # Assertions
        assert len(results) == 1
        assert results[0]['id'] == 'doc_1'
        assert results[0]['metadata']['jurisdiction'] == 'US'
        assert results[0]['metadata']['regulation'] == 'US_COPPA'
        
        # Verify collection.query was called with filter
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[self.sample_query_embedding],
            n_results=5,
            where=metadata_filter,
            include=['metadatas', 'documents', 'distances']
        )
    
    def test_retrieve_empty_results(self):
        """Test retrieve with no results found."""
        from rag.retrieval.retriever import RAGRetriever
        
        # Setup mock collection with empty response
        empty_response = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        self.mock_collection.query.return_value = empty_response
        
        # Initialize retriever
        retriever = RAGRetriever(collection=self.mock_collection)
        
        # Perform retrieval
        results = retriever.retrieve(
            query_embedding=self.sample_query_embedding,
            n_results=5
        )
        
        # Assertions
        assert results == []
    
    def test_retrieve_error_handling(self):
        """Test error handling when ChromaDB query fails."""
        from rag.retrieval.retriever import RAGRetriever
        
        # Setup mock collection to raise exception
        self.mock_collection.query.side_effect = Exception("ChromaDB connection failed")
        
        # Initialize retriever
        retriever = RAGRetriever(collection=self.mock_collection)
        
        # Should raise exception when query fails
        with pytest.raises(Exception, match="ChromaDB connection failed"):
            retriever.retrieve(
                query_embedding=self.sample_query_embedding,
                n_results=5
            )
    
    def test_retrieve_with_custom_include_params(self):
        """Test retrieve with custom include parameters."""
        from rag.retrieval.retriever import RAGRetriever
        
        # Setup mock collection response
        self.mock_collection.query.return_value = self.sample_query_response
        
        # Initialize retriever
        retriever = RAGRetriever(collection=self.mock_collection)
        
        # Perform retrieval with custom include params
        results = retriever.retrieve(
            query_embedding=self.sample_query_embedding,
            n_results=2,
            include=['documents', 'metadatas']  # No distances
        )
        
        # Verify collection.query was called with custom include
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[self.sample_query_embedding],
            n_results=2,
            include=['documents', 'metadatas']
        )
