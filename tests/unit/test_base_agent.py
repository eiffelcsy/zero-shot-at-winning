import pytest
import asyncio
from unittest.mock import AsyncMock

from app.agents.base import BaseComplianceAgent

class DummyAgent(BaseComplianceAgent):  # ✅ RENAMED from TestAgent
    """Test implementation of BaseComplianceAgent"""
    async def process(self, input_data):
        result = await self.safe_llm_call("Test prompt")
        return self._parse_llm_output(result)

class TestBaseComplianceAgent:
    
    @pytest.mark.asyncio
    async def test_basic_agent_processing(self, mock_llm):
        """Test basic agent processing functionality"""
        agent = DummyAgent("DummyAgent", mock_llm)  # ✅ FIXED: Pass mock_llm as positional arg
        
        result = await agent.process({"test": "data"})
        
        assert result["risk_level"] == "HIGH"
        assert result["compliance_required"] == True
        assert "reasoning" in result
        
        # Verify LLM was called
        mock_llm.apredict.assert_called_once()
    
    def test_confidence_calculation_high(self):
        """Test confidence calculation with high-confidence indicators"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        confidence = agent.calculate_confidence(
            reasoning="This clearly violates GDPR regulation specifically documented in Article 6",
            evidence=["GDPR Article 6", "Data processing", "Legal requirement"],
            context={"has_legal_keywords": True, "geographic_specificity": True}
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # Should be high confidence
    
    def test_confidence_calculation_low(self):
        """Test confidence calculation with low-confidence indicators"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        confidence = agent.calculate_confidence(
            reasoning="This is unclear and possibly might need review",
            evidence=[],
            context=None
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.5  # Should be low confidence
    
    @pytest.mark.parametrize("reasoning,evidence,expected_range", [
        ("clearly documented regulation", ["GDPR"], (0.6, 0.8)),
        ("likely suggests compliance", ["some evidence"], (0.3, 0.6)), 
        ("unclear and ambiguous", [], (0.25, 0.35))
    ])
    def test_confidence_calculation_ranges(self, reasoning, evidence, expected_range):
        """Test confidence calculation across different scenarios"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        confidence = agent.calculate_confidence(reasoning, evidence)
        
        assert expected_range[0] <= confidence <= expected_range[1]
    
    def test_json_parsing_valid_json(self):
        """Test JSON parsing with valid JSON"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        result = agent._parse_llm_output('{"test": "success", "confidence": 0.8}')
        
        assert result["test"] == "success"
        assert result["confidence"] == 0.8
    
    def test_json_parsing_markdown_wrapped(self):
        """Test JSON parsing with markdown-wrapped JSON"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        markdown_json = """
        Here's the analysis:
        ```
        {"risk_level": "HIGH", "compliance_required": true}
        ```
        Hope this helps!
        """
        
        result = agent._parse_llm_output(markdown_json)
        
        assert result["risk_level"] == "HIGH"
        assert result["compliance_required"] == True
    
    def test_json_parsing_fallback(self):
        """Test JSON parsing fallback for invalid JSON"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        result = agent._parse_llm_output("This is not JSON at all")
        
        assert "error" in result
        assert result["confidence"] == 0.1
        assert result["needs_human_review"] == True
    
    def test_agent_configuration(self):
        """Test agent configuration methods"""
        agent = DummyAgent("TestAgent", AsyncMock())  # ✅ FIXED: Pass AsyncMock as positional arg
        
        config = agent.get_config()
        
        assert config["name"] == "TestAgent"
        assert "created_at" in config
        assert "confidence_threshold" in config
        assert config["confidence_threshold"] == 0.7
    
    @pytest.mark.asyncio
    async def test_safe_llm_call_success(self, mock_llm):
        """Test successful LLM call"""
        agent = DummyAgent("TestAgent", mock_llm)  # ✅ FIXED: Pass mock_llm as positional arg
        
        result = await agent.safe_llm_call("Test prompt")
        
        assert isinstance(result, str)
        mock_llm.apredict.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_safe_llm_call_retry_on_failure(self):
        """Test LLM call retries on failure"""
        failing_llm = AsyncMock()
        failing_llm.apredict.side_effect = [
            Exception("Connection error"),
            Exception("Another error"), 
            '{"success": true}'  # Success on 3rd attempt
        ]
        
        agent = DummyAgent("TestAgent", failing_llm)  # ✅ FIXED: Pass failing_llm as positional arg
        
        result = await agent.safe_llm_call("Test prompt", max_retries=3)
        
        assert result == '{"success": true}'
        assert failing_llm.apredict.call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_llm):
        """Test agent as async context manager"""
        async with DummyAgent("TestAgent", mock_llm) as agent:  # ✅ FIXED: Pass mock_llm as positional arg
            assert agent.name == "TestAgent"
            result = await agent.process({"test": "data"})
            assert "risk_level" in result
    
    def test_agent_string_representation(self, mock_llm):
        """Test agent string representation"""
        agent = DummyAgent("TestAgent", mock_llm)  # ✅ FIXED: Pass mock_llm as positional arg
        
        repr_str = repr(agent)
        
        assert "TestAgent" in repr_str or "DummyAgent" in repr_str
        assert "name=" in repr_str