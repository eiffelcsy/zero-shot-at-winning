import pytest
from unittest.mock import AsyncMock
from typing import List


class TestQueryProcessor:
    """Tests for QueryProcessor component."""

    def setup_method(self):
        """Setup test fixtures for each test method."""
        self.mock_llm = AsyncMock()
        
        # Sample queries relevant to geo-compliance assessment
        self.compliance_queries = {
            'privacy_basic': "What are the COPPA requirements for data collection?",
            'age_verification': "Age verification requirements for social media",
            'geo_restrictions': "California SB-976 geographic restrictions",
            'consent_mechanisms': "Parental consent requirements for minors",
            'data_processing': "GDPR data processing restrictions for children",
            'short_regulatory': "COPPA compliance",
            'technical_feature': "automated content recommendation age verification"
        }

    def test_init_with_llm(self, mock_llm):
        """Test QueryProcessor can be initialized with an LLM."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        assert processor.llm == mock_llm
    
    def test_init_without_llm(self):
        """Test QueryProcessor can be initialized without an LLM (for basic functionality)."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor()
        assert processor.llm is None
    
    @pytest.mark.asyncio
    async def test_expand_query_basic_compliance_terms(self, mock_llm):
        """Test expand_query adds relevant compliance and regulatory terms."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock LLM response with compliance-related expansion terms
        mock_llm.apredict.return_value = "data collection requirements, privacy protection, consent mechanisms, age verification, parental approval, regulatory compliance"
        
        original_query = "COPPA requirements for children's data"
        expanded_query = await processor.expand_query(original_query)
        
        # Should contain original query and expansion terms
        assert "COPPA requirements for children's data" in expanded_query
        assert "data collection requirements" in expanded_query
        assert "privacy protection" in expanded_query
        assert "consent mechanisms" in expanded_query
        assert "age verification" in expanded_query
        
        # Verify LLM was called with appropriate prompt
        mock_llm.apredict.assert_called_once()
        call_args = mock_llm.apredict.call_args[0][0]
        assert "COPPA requirements for children's data" in call_args
        assert "compliance" in call_args.lower()
    
    @pytest.mark.asyncio 
    async def test_expand_query_geographic_jurisdictions(self, mock_llm):
        """Test expand_query includes relevant geographic and jurisdictional terms."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock response focusing on geographic compliance
        mock_llm.apredict.return_value = "California laws, state regulations, federal requirements, jurisdiction-specific rules, geographic restrictions, regional compliance"
        
        original_query = "California SB-976 social media restrictions"
        expanded_query = await processor.expand_query(original_query)
        
        # Should expand with jurisdictional context
        assert "California laws" in expanded_query
        assert "state regulations" in expanded_query
        assert "jurisdiction-specific rules" in expanded_query
        assert "geographic restrictions" in expanded_query
    
    @pytest.mark.asyncio
    async def test_expand_query_technical_features(self, mock_llm):
        """Test expand_query handles technical feature assessment queries."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock response for technical features
        mock_llm.apredict.return_value = "algorithmic systems, automated decision making, recommendation engines, content filtering, user profiling, behavioral targeting"
        
        original_query = "automated content recommendation compliance"
        expanded_query = await processor.expand_query(original_query)
        
        # Should include technical synonyms and related concepts
        assert "algorithmic systems" in expanded_query
        assert "recommendation engines" in expanded_query
        assert "automated decision making" in expanded_query
    
    @pytest.mark.asyncio
    async def test_expand_query_empty_input(self, mock_llm):
        """Test expand_query handles empty or None input gracefully."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Test empty string
        result = await processor.expand_query("")
        assert result == ""
        
        # Test None
        result = await processor.expand_query(None)
        assert result == ""
        
        # LLM should not be called for empty inputs
        mock_llm.apredict.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_expand_query_without_llm(self):
        """Test expand_query returns original query when no LLM is provided."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor()  # No LLM
        
        original_query = "GDPR compliance requirements"
        result = await processor.expand_query(original_query)
        
        # Should return original query unchanged
        assert result == original_query
    
    @pytest.mark.asyncio
    async def test_expand_query_llm_error_handling(self, mock_llm):
        """Test expand_query handles LLM errors gracefully."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock LLM to raise an exception
        mock_llm.apredict.side_effect = Exception("API rate limit exceeded")
        
        original_query = "COPPA requirements"
        result = await processor.expand_query(original_query)
        
        # Should fallback to original query on error
        assert result == original_query