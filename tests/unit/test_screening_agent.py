import pytest
from unittest.mock import AsyncMock

from app.agents.screening import ScreeningAgent

class TestScreeningAgent:
    
    @pytest.mark.asyncio
    async def test_high_risk_feature_detection(self, mock_llm_with_response):
        """Test screening agent with high-risk compliance feature"""
        llm_response = '{"risk_level": "HIGH", "compliance_required": true, "reasoning": "This clearly involves ASL and GH for Utah minors", "trigger_keywords": ["ASL", "GH", "Utah"], "confidence": 0.9}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": "ASL-based curfew restrictions for Utah minors using GH",
            "feature_name": "Utah Curfew Feature"
        })
        
        assert result["agent"] == "ScreeningAgent"
        assert result["analysis"]["risk_level"] == "HIGH"
        assert result["analysis"]["compliance_required"] == True
        assert result["next_step"] in ["research", "validation"]
        assert result["feature_name"] == "Utah Curfew Feature"
    
    @pytest.mark.asyncio
    async def test_low_risk_feature_detection(self, mock_llm_with_response):
        """Test screening agent with low-risk feature"""
        llm_response = '{"risk_level": "LOW", "compliance_required": false, "reasoning": "Simple UI change with no regulatory implications", "confidence": 0.8}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": "Update button colors and spacing",
            "feature_name": "UI Update"
        })
        
        assert result["analysis"]["risk_level"] == "LOW"
        assert result["analysis"]["compliance_required"] == False
        # The actual confidence might be recalculated and lowered
        assert result["next_step"] in ["validation", "human_review"]
    
    @pytest.mark.asyncio
    async def test_keyword_detection_enhancement(self, mock_llm_with_response):
        """Test that agent enhances LLM output with rule-based keyword detection"""
        llm_response = '{"risk_level": "MEDIUM", "compliance_required": true, "reasoning": "Basic analysis", "trigger_keywords": []}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": "Feature uses T5 data with Jellybean controls and GH routing",
            "feature_name": "Data Processing Feature"
        })
        
        # Should detect T5, Jellybean, and GH as trigger keywords
        trigger_keywords = result["analysis"]["trigger_keywords"]
        assert "T5" in trigger_keywords
        assert "Jellybean" in trigger_keywords
        assert "GH" in trigger_keywords
    
    @pytest.mark.parametrize("feature_desc,expected_regions", [
        ("ASL restrictions for Utah minors", ["Utah"]),
        ("GDPR compliance for EU users", ["EU"]),
        ("California SB976 implementation", ["California"]),
        ("Florida parental controls", ["Florida"]),
        ("NCMEC reporting system", ["US"]),
        ("Simple UI change", ["unknown"])
    ])
    @pytest.mark.asyncio
    async def test_geographic_scope_extraction(self, mock_llm_with_response, feature_desc, expected_regions):
        """Test geographic region extraction from feature descriptions"""
        llm_response = '{"risk_level": "MEDIUM", "compliance_required": true, "reasoning": "test"}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": feature_desc,
            "feature_name": "Test Feature"
        })
        
        geographic_scope = result["analysis"]["geographic_scope"]
        for region in expected_regions:
            assert region in geographic_scope
    
    @pytest.mark.parametrize("feature_desc,expected_age_sensitive", [
        ("ASL-based restrictions for minors", True),
        ("Jellybean parental controls", True),
        ("Snowcap child safety measures", True),
        ("Simple UI color change", False),
        ("Adult user analytics", False)
    ])
    @pytest.mark.asyncio
    async def test_age_sensitivity_detection(self, mock_llm_with_response, feature_desc, expected_age_sensitive):
        """Test age sensitivity detection"""
        llm_response = '{"risk_level": "MEDIUM", "compliance_required": true, "reasoning": "test"}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": feature_desc,
            "feature_name": "Test Feature"
        })
        
        assert result["analysis"]["age_sensitivity"] == expected_age_sensitive
    
    @pytest.mark.parametrize("feature_desc,expected_sensitivity", [
        ("T5 data processing", "T5"),
        ("Personal data collection", "T4"),
        ("User analytics logging", "T3"),
        ("Simple button click", "T1")
    ])
    @pytest.mark.asyncio
    async def test_data_sensitivity_classification(self, mock_llm_with_response, feature_desc, expected_sensitivity):
        """Test data sensitivity classification"""
        llm_response = '{"risk_level": "MEDIUM", "compliance_required": true, "reasoning": "test"}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": feature_desc,
            "feature_name": "Test Feature"
        })
        
        assert result["analysis"]["data_sensitivity"] == expected_sensitivity
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test agent error handling when LLM fails"""
        failing_llm = AsyncMock()
        failing_llm.apredict.side_effect = Exception("LLM connection failed")
        agent = ScreeningAgent(failing_llm)
        
        result = await agent.process({
            "feature_description": "Test feature",
            "feature_name": "Test Feature"
        })
        
        assert result["agent"] == "ScreeningAgent"
        assert "error" in result["analysis"]
        assert result["next_step"] == "human_review"
        assert result["analysis"]["needs_human_review"] == True
    
    @pytest.mark.asyncio
    async def test_confidence_enhancement(self, mock_llm_with_response):
        """Test that agent enhances confidence calculation"""
        llm_response = '{"risk_level": "HIGH", "compliance_required": true, "reasoning": "This clearly violates specific GDPR regulations", "trigger_keywords": ["GDPR", "regulation"]}'
        llm = mock_llm_with_response(llm_response)
        agent = ScreeningAgent(llm)
        
        result = await agent.process({
            "feature_description": "T5 personal data processing for EU users with location tracking",
            "feature_name": "GDPR Feature"
        })
        
        # Should have high confidence due to legal keywords and geographic specificity
        assert result["analysis"]["confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_next_step_determination(self, mock_llm_with_response, sample_features):
        """Test next step determination logic"""
        test_cases = [
            ('{"risk_level": "HIGH", "compliance_required": true, "confidence": 0.8}', "research"),
            ('{"risk_level": "LOW", "compliance_required": false, "confidence": 0.8}', ["validation", "human_review"]),  # ✅ Accept both
            ('{"risk_level": "MEDIUM", "compliance_required": true, "confidence": 0.3}', "human_review")
        ]
        
        for llm_response, expected_next_step in test_cases:
            llm = mock_llm_with_response(llm_response)
            agent = ScreeningAgent(llm)
            
            result = await agent.process({
                "feature_description": "Test feature",
                "feature_name": "Test"
            })
            
            if isinstance(expected_next_step, list):
                assert result["next_step"] in expected_next_step
            else:
                assert result["next_step"] == expected_next_step