#!/usr/bin/env python3
"""
Unit tests for ResearchAgent with mocked ChromaDB tools
"""

import unittest
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from datetime import datetime
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.agents.research import ResearchAgent, ResearchOutput

class MockJurisdictionChecker:
    """Mock ChromaDB jurisdiction checker"""
    def __init__(self, client):
        self.client = client
        
    def check_applicable_jurisdictions(self, geographic_scope, trigger_keywords):
        """Return mock jurisdiction results"""
        if "US" in geographic_scope or "California" in geographic_scope:
            return [
                {"regulation_code": "COPPA", "jurisdiction": "US", "relevance_score": 0.9},
                {"regulation_code": "CA_SB976", "jurisdiction": "California", "relevance_score": 0.85}
            ]
        elif "EU" in geographic_scope:
            return [
                {"regulation_code": "GDPR", "jurisdiction": "EU", "relevance_score": 0.95}
            ]
        return []

class MockRegulationSearcher:
    """Mock ChromaDB regulation searcher"""
    def __init__(self, client):
        self.client = client
        
    def search_by_compliance_patterns(self, patterns, geographic_scope):
        """Return mock regulation search results"""
        results = []
        
        if "age_restrictions" in patterns:
            results.extend([
                {
                    "reg": "COPPA",
                    "jurisdiction": "US",
                    "name": "Children's Online Privacy Protection Act",
                    "section": "Section 1304",
                    "url": "https://www.coppa.gov",
                    "excerpt": "Requires parental consent for data collection from children under 13",
                    "score": 9.2
                },
                {
                    "reg": "CA_SB976", 
                    "jurisdiction": "California",
                    "name": "California Age-Appropriate Design Code",
                    "section": "Default Settings",
                    "url": "https://leginfo.legislature.ca.gov",
                    "excerpt": "Social media platforms must disable personalized feeds by default for users under 18",
                    "score": 8.8
                }
            ])
        
        if "data_protection" in patterns:
            results.append({
                "reg": "GDPR",
                "jurisdiction": "EU", 
                "name": "General Data Protection Regulation",
                "section": "Article 8",
                "url": "https://gdpr-info.eu",
                "excerpt": "Processing of personal data of children requires parental consent",
                "score": 9.5
            })
            
        return results

class MockRiskCalculator:
    """Mock ChromaDB risk calculator"""
    def __init__(self, client):
        self.client = client
        
    def calculate_compliance_risk(self, feature_description, geographic_scope, 
                                age_sensitivity, data_sensitivity, trigger_keywords):
        """Return mock risk assessment"""
        risk_score = 0.3  # Base risk
        
        if age_sensitivity:
            risk_score += 0.4
        if data_sensitivity in ["T5", "T4"]:
            risk_score += 0.3
        if any(geo in ["US", "California", "EU"] for geo in geographic_scope):
            risk_score += 0.2
            
        risk_score = min(risk_score, 1.0)
        
        return {
            "overall_risk_score": risk_score,
            "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW",
            "compliance_required": risk_score > 0.6,
            "confidence": 0.85,
            "risk_factors": [
                {"factor": "age_sensitivity", "score": 0.4 if age_sensitivity else 0.0},
                {"factor": "data_sensitivity", "score": 0.3 if data_sensitivity in ["T5", "T4"] else 0.0}
            ]
        }

class TestResearchAgent(unittest.TestCase):
    
    @patch('chromadb.HttpClient')
    @patch('app.agents.research.JurisdictionChecker', MockJurisdictionChecker)
    @patch('app.agents.research.RegulationSearcher', MockRegulationSearcher) 
    @patch('app.agents.research.RiskCalculator', MockRiskCalculator)
    def setUp(self, mock_chroma):
        """Setup test fixtures with mocked ChromaDB"""
        mock_chroma.return_value = MagicMock()
        self.agent = ResearchAgent()
    
    def test_init_with_memory_overlay(self):
        """Test agent initialization with memory overlay"""
        with patch('chromadb.HttpClient'), \
             patch('app.agents.research.JurisdictionChecker', MockJurisdictionChecker), \
             patch('app.agents.research.RegulationSearcher', MockRegulationSearcher), \
             patch('app.agents.research.RiskCalculator', MockRiskCalculator):
            
            memory = "TEST RESEARCH MEMORY"
            agent = ResearchAgent(memory_overlay=memory)
            
            self.assertEqual(agent.memory_overlay, memory)
            self.assertEqual(agent.name, "ResearchAgent")
    
    def test_extract_compliance_patterns(self):
        """Test compliance pattern extraction from screening analysis"""
        screening_analysis = {
            "age_sensitivity": True,
            "data_sensitivity": "T5", 
            "compliance_required": True,
            "geographic_scope": ["US", "California"]
        }
        
        patterns = self.agent._extract_compliance_patterns(screening_analysis)
        
        expected_patterns = [
            "age_restrictions",
            "data_protection", 
            "content_governance",
            "platform_responsibilities",
            "geographic_enforcement"
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns)
    
    def test_extract_compliance_patterns_minimal(self):
        """Test pattern extraction with minimal compliance requirements"""
        screening_analysis = {
            "age_sensitivity": False,
            "data_sensitivity": "T2",
            "compliance_required": False, 
            "geographic_scope": ["unknown"]
        }
        
        patterns = self.agent._extract_compliance_patterns(screening_analysis)
        
        # Should not have age_restrictions or data_protection
        self.assertNotIn("age_restrictions", patterns)
        self.assertNotIn("data_protection", patterns)
        self.assertNotIn("content_governance", patterns)
        # Should not have geographic_enforcement for unknown scope
        self.assertNotIn("geographic_enforcement", patterns)
    
    def test_format_candidates(self):
        """Test formatting of jurisdiction results into candidates"""
        jurisdiction_results = [
            {"regulation_code": "COPPA", "jurisdiction": "US", "relevance_score": 0.9},
            {"regulation_code": "GDPR", "jurisdiction": "EU", "relevance_score": 0.85}
        ]
        
        candidates = self.agent._format_candidates(jurisdiction_results)
        
        self.assertEqual(len(candidates), 2)
        
        coppa_candidate = next(c for c in candidates if c["reg"] == "COPPA")
        self.assertEqual(coppa_candidate["why"], "Applicable to US")
        self.assertEqual(coppa_candidate["score"], 0.9)
        
        gdpr_candidate = next(c for c in candidates if c["reg"] == "GDPR")
        self.assertEqual(gdpr_candidate["why"], "Applicable to EU") 
        self.assertEqual(gdpr_candidate["score"], 0.85)
    
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    async def test_process_success_high_risk(self, mock_llm):
        """Test successful processing of high-risk research"""
        # Mock LLM response
        mock_llm.return_value = {
            "agent": "ResearchAgent",
            "candidates": [
                {"reg": "COPPA", "why": "Applies to US minors data collection", "score": 0.9}
            ],
            "evidence": [
                {
                    "reg": "COPPA",
                    "jurisdiction": "US", 
                    "name": "Children's Online Privacy Protection Act",
                    "section": "Section 1304",
                    "url": "https://www.coppa.gov",
                    "excerpt": "Requires parental consent for children under 13",
                    "score": 9.2
                }
            ],
            "query_used": "age_restrictions data_protection US California minors T5",
            "confidence_score": 0.88
        }
        
        # Test state with high-risk screening
        state = {
            "feature_name": "Teen Location Sharing",
            "feature_description": "Location sharing feature for users under 18",
            "screening_analysis": {
                "agent": "ScreeningAgent",
                "risk_level": "HIGH",
                "compliance_required": True,
                "age_sensitivity": True,
                "data_sensitivity": "T5",
                "geographic_scope": ["US", "California"],
                "trigger_keywords": ["minors", "location", "T5"]
            }
        }
        
        # Execute
        result = await self.agent.process(state)
        
        # Assertions
        self.assertTrue(result["research_completed"])
        self.assertEqual(result["next_step"], "validation")
        
        analysis = result["research_analysis"]
        self.assertEqual(analysis["agent"], "ResearchAgent")
        self.assertGreater(len(analysis["candidates"]), 0)
        self.assertGreater(len(analysis["evidence"]), 0)
        self.assertGreater(analysis["confidence_score"], 0.0)
        
        # Check risk assessment
        risk_assessment = result["research_risk_assessment"]
        self.assertEqual(risk_assessment["risk_level"], "HIGH")
        self.assertTrue(risk_assessment["compliance_required"])
    
    async def test_process_missing_screening(self):
        """Test error handling when screening analysis is missing"""
        state = {
            "feature_name": "Test Feature",
            "feature_description": "Test description"
            # Missing screening_analysis
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["research_completed"])
        self.assertEqual(result["next_step"], "validation")
        self.assertIn("research_error", result)
        self.assertIn("Missing screening analysis", result["research_error"])
    
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    async def test_process_llm_failure(self, mock_llm):
        """Test error handling when LLM call fails"""
        mock_llm.side_effect = Exception("LLM API Error")
        
        state = {
            "feature_description": "Test feature",
            "screening_analysis": {
                "risk_level": "MEDIUM",
                "geographic_scope": ["US"],
                "trigger_keywords": ["test"],
                "age_sensitivity": False,
                "data_sensitivity": "T3"
            }
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["research_completed"])
        analysis = result["research_analysis"] 
        self.assertEqual(analysis["confidence_score"], 0.0)
        self.assertIn("LLM API Error", analysis["error"])

class AsyncTestCase(unittest.TestCase):
    """Base class for async test methods"""
    
    def run_async(self, coro):
        """Helper to run async tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Sync versions of async tests for unittest compatibility
class TestResearchAgentSync(AsyncTestCase):
    
    @patch('chromadb.HttpClient')
    @patch('app.agents.research.JurisdictionChecker', MockJurisdictionChecker)
    @patch('app.agents.research.RegulationSearcher', MockRegulationSearcher)
    @patch('app.agents.research.RiskCalculator', MockRiskCalculator)
    def setUp(self, mock_chroma):
        mock_chroma.return_value = MagicMock()
        self.agent = ResearchAgent()
    
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    def test_process_success_sync(self, mock_llm):
        """Sync version of successful process test"""
        mock_llm.return_value = {
            "agent": "ResearchAgent",
            "candidates": [{"reg": "COPPA", "why": "US minors", "score": 0.9}],
            "evidence": [{
                "reg": "COPPA",
                "jurisdiction": "US",
                "name": "COPPA",
                "section": "Section 1304", 
                "url": "https://coppa.gov",
                "excerpt": "Parental consent required",
                "score": 9.0
            }],
            "query_used": "age_restrictions US",
            "confidence_score": 0.85
        }
        
        state = {
            "feature_description": "Test feature for minors",
            "screening_analysis": {
                "risk_level": "HIGH",
                "age_sensitivity": True,
                "data_sensitivity": "T5",
                "geographic_scope": ["US"],
                "trigger_keywords": ["minors"],
                "compliance_required": True
            }
        }
        
        result = self.run_async(self.agent.process(state))
        
        self.assertTrue(result["research_completed"])
        self.assertGreater(len(result["research_evidence"]), 0)
        self.assertGreater(len(result["research_candidates"]), 0)

if __name__ == '__main__':
    # Run with: python -m pytest test_research_agent.py -v
    unittest.main()
