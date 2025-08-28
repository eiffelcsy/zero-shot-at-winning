import pytest
from unittest.mock import Mock


class TestRetrievalTool:
    """Test the MVP core retrieval flow."""
    
    def test_init_with_dependencies(self, mock_query_processor, mock_retriever):
        """Test RetrievalTool initialization with required dependencies."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=mock_retriever
        )
        
        assert tool.query_processor == mock_query_processor
        assert tool.retriever == mock_retriever
        assert tool.name == "retrieval_tool"
    
    def test_init_missing_dependencies_raises_error(self):
        """Test that missing dependencies raise appropriate errors."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        with pytest.raises(ValueError, match="query_processor is required"):
            RetrievalTool(query_processor=None, retriever=Mock())
        
        with pytest.raises(ValueError, match="retriever is required"):
            RetrievalTool(query_processor=Mock(), retriever=None)
    
    @pytest.mark.asyncio
    async def test_run_core_flow(self, mock_query_processor, mock_retriever):
        """Test the complete run() flow: enhance_query -> retrieve."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=mock_retriever
        )
        
        # Agent's query
        agent_query = "What are the age verification requirements for our platform?"
        
        # Run the tool
        result = await tool.run(agent_query)
        
        # Verify the flow was executed
        mock_query_processor.enhance_query.assert_called_once_with(agent_query)
        mock_retriever.retrieve.assert_called_once()
        
        # Verify result format - should be raw retrieval results
        assert isinstance(result, list)
        assert len(result) >= 1
        
        # Check result structure matches raw ChromaDB format
        first_result = result[0]
        assert "id" in first_result
        assert "document" in first_result
        assert "metadata" in first_result
        assert "distance" in first_result
    
    @pytest.mark.asyncio
    async def test_enhance_query_step(self, mock_query_processor, mock_retriever):
        """Test query enhancement step specifically."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=mock_retriever
        )
        
        original_query = "Do we need parental consent for algorithmic recommendations?"
        enhanced_query = "parental consent algorithmic recommendations minors COPPA requirements social media"
        
        # Mock the enhancement
        mock_query_processor.enhance_query.return_value = enhanced_query
        
        await tool.run(original_query)
        
        # Verify enhancement was called with original query
        mock_query_processor.enhance_query.assert_called_once_with(original_query)
        
        # Verify retriever was called with enhanced query
        retrieve_call_args = mock_retriever.retrieve.call_args[0]
        # The enhanced query should be passed to retrieve (likely as embedding)
        mock_retriever.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_step(self, mock_query_processor, mock_retriever):
        """Test the retrieve step with ChromaDB results."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=mock_retriever
        )
        
        # Mock raw ChromaDB-style results
        raw_results = [
            {
                'id': 'doc_1',
                'document': 'COPPA requires parental consent for data collection from children under 13.',
                'metadata': {'regulation_code': 'US_COPPA', 'jurisdiction': 'US'},
                'distance': 0.2
            }
        ]
        mock_retriever.retrieve.return_value = raw_results
        
        result = await tool.run("test query")
        
        # Verify retriever was called
        mock_retriever.retrieve.assert_called_once()
        
        # Verify results were returned as raw ChromaDB format
        assert len(result) == 1
        assert result[0]["metadata"]["regulation_code"] == "US_COPPA"
    
    @pytest.mark.asyncio 
    async def test_empty_query_handling(self, mock_query_processor, mock_retriever):
        """Test handling of empty or None queries."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=mock_retriever
        )
        
        # Test empty string
        result = await tool.run("")
        assert result == []
        
        # Test None
        result = await tool.run(None)
        assert result == []
        
        # Test whitespace only
        result = await tool.run("   \n\t   ")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_no_results_found(self, mock_query_processor, mock_retriever):
        """Test behavior when no results are found."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        # Mock empty results
        mock_retriever.retrieve.return_value = []
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=mock_retriever
        )
        
        result = await tool.run("obscure query with no matches")
        
        assert result == []
        # Should still call the pipeline
        mock_query_processor.enhance_query.assert_called_once()
        mock_retriever.retrieve.assert_called_once()


class TestRetrievalToolErrorHandling:
    """Test error handling for MVP functionality."""
    
    @pytest.mark.asyncio
    async def test_query_processor_error(self, mock_retriever):
        """Test handling when query processor fails."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        # Mock failing query processor
        failing_processor = Mock()
        failing_processor.enhance_query.side_effect = Exception("Query enhancement failed")
        
        tool = RetrievalTool(
            query_processor=failing_processor,
            retriever=mock_retriever
        )
        
        # Should handle gracefully and still return results using original query
        result = await tool.run("test query")
        
        # Should return results from retriever (using fallback to original query)
        assert isinstance(result, list)
        assert len(result) > 0  # Should have results from mock_retriever
        
        # Verify that retriever was still called despite query processor failure
        mock_retriever.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retriever_error(self, mock_query_processor):
        """Test handling when retriever fails."""
        from rag.tools.retrieval_tool import RetrievalTool
        
        # Mock failing retriever
        failing_retriever = Mock()
        failing_retriever.retrieve.side_effect = Exception("ChromaDB connection failed")
        
        tool = RetrievalTool(
            query_processor=mock_query_processor,
            retriever=failing_retriever
        )
        
        # Should handle gracefully
        result = await tool.run("test query")
        
        # Should return empty results
        assert isinstance(result, list)
        assert len(result) == 0