#!/usr/bin/env python3
"""
Integration tests for Screening -> Research agent workflow
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.agents.screening import ScreeningAgent
from app.agents.research import ResearchAgent
from app.agents.state import create_initial_state

class TestWorkflowIntegration(unittest.TestCase):
    
    @patch('chromadb.HttpClient')
    @patch('app.agents.research.JurisdictionChecker')
    @patch('app.agents.research.RegulationSearcher') 
    @patch('app.agents.research.RiskCalculator')
    def setUp(self, mock_risk, mock_reg, mock_jur, mock_chroma):
        """Setup agents with mocked dependencies"""
        # Setup mocks
        mock_chroma.return_value = MagicMock()
        mock_jur.return_value.check_applicable_jurisdictions.return_value = []
        mock_reg.return_value.search_by_compliance_patterns.return_value = []
        mock_risk.return_value.calculate_compliance_risk.return_value = {
            "overall_risk_score": 0.5,
            "risk_level": "MEDIUM",
            "compliance_required": True,
            "confidence": 0.8
        }
        
        self.screening_agent = ScreeningAgent()
        self.research_agent = ResearchAgent()
    
    def run_async(self, coro):
        """Helper to run async tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @patch('app.agents.screening.ScreeningAgent.safe_llm_call')
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    def test_full_workflow_high_risk(self, mock_research_llm, mock_screening_llm):
        """Test complete workflow for high-risk feature"""
        
        # Mock screening LLM response
        mock_screening_llm.return_value = {
            "agent": "ScreeningAgent",
            "risk_level": "HIGH",
            "compliance_required": True,
            "confidence": 0.9,
            "trigger_keywords": ["minors", "T5", "ASL"],
            "reasoning": "Feature involves T5 data processing for minors with ASL",
            "needs_research": True,
            "geographic_scope": ["US", "California"],
            "age_sensitivity": True,
            "data_sensitivity": "T5"
        }
        
        # Mock research LLM response  
        mock_research_llm.return_value = {
            "agent": "ResearchAgent",
            "candidates": [
                {"reg": "COPPA", "why": "US minors data protection", "score": 0.95},
                {"reg": "CA_SB976", "why": "California minors social media", "score": 0.88}
            ],
            "evidence": [
                {
                    "reg": "COPPA",
                    "jurisdiction": "US",
                    "name": "Children's Online Privacy Protection Act", 
                    "section": "Section 1304",
                    "url": "https://www.coppa.gov",
                    "excerpt": "Requires parental consent for data collection from children under 13",
                    "score": 9.5
                }
            ],
            "query_used": "age_restrictions data_protection US California minors T5 ASL",
            "confidence_score": 0.92
        }
        
        # Initial state
        initial_state = create_initial_state(
            feature_name="Teen Social Location Sharing",
            feature_description="Feature allowing users under 18 to share location data with friends using ASL verification and T5 data processing through GH routing",
            context_documents="Uses Jellybean parental controls and Snowcap child safety framework"
        )
        
        # Run screening
        screening_result = self.run_async(self.screening_agent.process(initial_state))
        
        # Verify screening worked
        self.assertTrue(screening_result["screening_completed"])
        self.assertEqual(screening_result["next_step"], "research")
        
        # Update state with screening results
        updated_state = initial_state.copy()
        updated_state.update(screening_result)
        
        # Run research
        research_result = self.run_async(self.research_agent.process(updated_state))
        
        # Verify research worked
        self.assertTrue(research_result["research_completed"])
        self.assertEqual(research_result["next_step"], "validation")
        
        # Verify data flow between agents
        screening_analysis = screening_result["screening_analysis"]
        research_analysis = research_result["research_analysis"]
        
        # Check that research used screening data
        self.assertEqual(screening_analysis["risk_level"], "HIGH")
        self.assertTrue(screening_analysis["age_sensitivity"])
        self.assertEqual(screening_analysis["data_sensitivity"], "T5")
        
        # Check research findings
        self.assertGreater(len(research_analysis["candidates"]), 0)
        self.assertGreater(len(research_analysis["evidence"]), 0)
        self.assertGreater(research_analysis["confidence_score"], 0.8)
        
        # Verify workflow continuity
        self.assertIn("COPPA", [c["reg"] for c in research_analysis["candidates"]])
        
    @patch('app.agents.screening.ScreeningAgent.safe_llm_call') 
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    def test_workflow_skip_research(self, mock_research_llm, mock_screening_llm):
        """Test workflow that skips research for low-risk features"""
        
        # Mock screening response for low-risk feature
        mock_screening_llm.return_value = {
            "agent": "ScreeningAgent",
            "risk_level": "LOW", 
            "compliance_required": False,
            "confidence": 0.95,
            "trigger_keywords": ["PF", "recommendation"],
            "reasoning": "Standard PF recommendation algorithm with minimal compliance impact",
            "needs_research": False,
            "geographic_scope": ["unknown"],
            "age_sensitivity": False,
            "data_sensitivity": "T2"
        }
        
        initial_state = create_initial_state(
            feature_name="Basic Content Recommendations",
            feature_description="Standard PF algorithm for content recommendations using T2 analytics data"
        )
        
        # Run screening
        screening_result = self.run_async(self.screening_agent.process(initial_state))
        
        # Verify screening skips to validation
        self.assertTrue(screening_result["screening_completed"])
        self.assertEqual(screening_result["next_step"], "validation")
        
        analysis = screening_result["screening_analysis"]
        self.assertEqual(analysis["risk_level"], "LOW")
        self.assertFalse(analysis["needs_research"])
        self.assertFalse(analysis["age_sensitivity"])
    
    def test_memory_overlay_propagation(self):
        """Test that memory overlay is properly propagated"""
        memory_overlay = """
        EXAMPLE:
        Feature: "Location sharing for teens"
        Expected: {"risk_level": "HIGH", "age_sensitivity": true}
        """
        
        screening_agent = ScreeningAgent(memory_overlay=memory_overlay)
        research_agent = ResearchAgent(memory_overlay=memory_overlay)
        
        self.assertEqual(screening_agent.memory_overlay, memory_overlay)
        self.assertEqual(research_agent.memory_overlay, memory_overlay)
        
        # Test memory update
        new_memory = "UPDATED MEMORY"
        screening_agent.update_memory(new_memory)
        research_agent.update_memory(new_memory)
        
        self.assertEqual(screening_agent.memory_overlay, new_memory)
        self.assertEqual(research_agent.memory_overlay, new_memory)

if __name__ == '__main__':
    # Run with: python -m pytest test_integration.py -v
    unittest.main()
