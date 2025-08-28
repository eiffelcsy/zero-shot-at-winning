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
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_basic_variations(self, mock_llm):
        """Test generate_multiple_queries creates diverse query variations."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock LLM response with query variations
        mock_llm.apredict.return_value = """1. What are the COPPA requirements for collecting children's personal data?
2. How does COPPA regulate data collection from minors under 13?
3. What parental consent requirements apply under COPPA?
4. Which data collection practices are prohibited by COPPA?
5. What are the compliance obligations for children's privacy under COPPA?"""
        
        original_query = "COPPA data collection requirements"
        variations = await processor.generate_multiple_queries(original_query)
        
        # Should return a list of distinct queries
        assert isinstance(variations, list)
        assert len(variations) >= 3  # Should generate multiple variations
        assert len(variations) <= 10  # But not too many
        
        # Each variation should be different and relevant
        variation_texts = [v.lower() for v in variations]
        assert any("personal data" in v for v in variation_texts)
        assert any("minors under 13" in v for v in variation_texts)
        assert any("parental consent" in v for v in variation_texts)
        
        # Variations should be distinct
        assert len(set(variations)) == len(variations)  # No duplicates
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_different_perspectives(self, mock_llm):
        """Test generate_multiple_queries provides different compliance perspectives."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock response with different compliance perspectives
        mock_llm.apredict.return_value = """1. What are the technical implementation requirements for age verification?
2. How do platforms ensure COPPA compliance in recommendation systems?
3. What legal liabilities exist for non-compliance with age restrictions?
4. Which user interface elements must include parental consent flows?
5. How should data processing workflows incorporate COPPA requirements?"""
        
        original_query = "age verification compliance implementation"
        variations = await processor.generate_multiple_queries(original_query)
        
        # Should cover different aspects: technical, legal, implementation
        variation_text = " ".join(variations).lower()
        assert "technical" in variation_text or "implementation" in variation_text
        assert "legal" in variation_text or "compliance" in variation_text
        assert "user interface" in variation_text or "workflow" in variation_text
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_geographic_variations(self, mock_llm):
        """Test generate_multiple_queries includes geographic/jurisdictional variations."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock response with geographic focus
        mock_llm.apredict.return_value = """1. How do California SB-976 restrictions apply to social media platforms?
2. What are the federal requirements versus California state laws for social media?
3. Which geographic regions have similar social media restriction laws?
4. How do multi-state companies comply with varying social media regulations?
5. What are the enforcement mechanisms for California's social media laws?"""
        
        original_query = "California social media restrictions"
        variations = await processor.generate_multiple_queries(original_query)
        
        # Should include jurisdictional perspectives
        variation_text = " ".join(variations).lower()
        assert "california" in variation_text
        assert ("federal" in variation_text or "state" in variation_text or 
                "multi-state" in variation_text or "regions" in variation_text)
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_with_count_parameter(self, mock_llm):
        """Test generate_multiple_queries respects the count parameter."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock response with numbered variations
        mock_llm.apredict.return_value = """1. What are GDPR requirements for children?
2. How does GDPR apply to minors' data?
3. What parental consent is required under GDPR?"""
        
        original_query = "GDPR children requirements"
        
        # Test with specific count
        variations = await processor.generate_multiple_queries(original_query, count=3)
        
        assert len(variations) == 3
        
        # Verify the LLM prompt requested the correct count
        mock_llm.apredict.assert_called_once()
        call_args = mock_llm.apredict.call_args[0][0]
        assert "3" in call_args
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_empty_input(self, mock_llm):
        """Test generate_multiple_queries handles empty input gracefully."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Test empty string
        result = await processor.generate_multiple_queries("")
        assert result == []
        
        # Test None
        result = await processor.generate_multiple_queries(None)
        assert result == []
        
        # LLM should not be called for empty inputs
        mock_llm.apredict.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_without_llm(self):
        """Test generate_multiple_queries returns original query when no LLM available."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor()  # No LLM
        
        original_query = "privacy compliance requirements"
        result = await processor.generate_multiple_queries(original_query)
        
        # Should return list containing only the original query
        assert result == [original_query]
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_llm_error_handling(self, mock_llm):
        """Test generate_multiple_queries handles LLM errors gracefully."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock LLM to raise an exception
        mock_llm.apredict.side_effect = Exception("API timeout")
        
        original_query = "data protection requirements"
        result = await processor.generate_multiple_queries(original_query)
        
        # Should fallback to original query on error
        assert result == [original_query]
    
    @pytest.mark.asyncio
    async def test_generate_multiple_queries_filters_invalid_responses(self, mock_llm):
        """Test generate_multiple_queries filters out empty or invalid variations."""
        from rag.retrieval.query_processor import QueryProcessor
        
        processor = QueryProcessor(llm=mock_llm)
        
        # Mock response with some empty/invalid lines
        mock_llm.apredict.return_value = """1. What are the COPPA requirements?
2. 
3. How do platforms comply with children's privacy laws?
4. 
5. What parental consent mechanisms are required?
6. """
        
        original_query = "children's privacy compliance"
        variations = await processor.generate_multiple_queries(original_query)
        
        # Should filter out empty variations
        assert len(variations) == 3
        for variation in variations:
            assert variation.strip() != ""
            assert len(variation.strip()) > 5  # Should be meaningful queries
