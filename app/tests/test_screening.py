"""
Unit tests for ScreeningAgent
"""

import unittest
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.agents.screening import ScreeningAgent, ScreeningOutput
from app.agents.state import create_initial_state

class TestScreeningAgent(unittest.TestCase):
    
    def setUp(self):
        """Setup test fixtures"""
        self.agent = ScreeningAgent()
        
    def test_init_with_memory_overlay(self):
        """Test agent initialization with memory overlay"""
        memory = "TEST MEMORY OVERLAY"
        agent = ScreeningAgent(memory_overlay=memory)
        
        self.assertEqual(agent.memory_overlay, memory)
        self.assertEqual(agent.name, "ScreeningAgent")
    
    def test_update_memory(self):
        """Test memory overlay update functionality"""
        initial_memory = "INITIAL MEMORY"
        new_memory = "NEW MEMORY"
        
        agent = ScreeningAgent(memory_overlay=initial_memory)
        self.assertEqual(agent.memory_overlay, initial_memory)
        
        agent.update_memory(new_memory)
        self.assertEqual(agent.memory_overlay, new_memory)
    
    @patch('app.agents.screening.ScreeningAgent.safe_llm_call')
    async def test_process_success_high_risk(self, mock_llm):
        """Test successful processing of high-risk feature"""
        # Mock LLM response
        mock_llm.return_value = {
            "agent": "ScreeningAgent",
            "risk_level": "HIGH",
            "compliance_required": True,
            "confidence": 0.9,
            "trigger_keywords": ["personal data", "minors", "T5"],
            "reasoning": "Feature involves T5 data processing for minors requiring compliance",
            "needs_research": True,
            "geographic_scope": ["US", "California"],
            "age_sensitivity": True,
            "data_sensitivity": "T5"
        }
        
        # Test state
        state = {
            "feature_name": "Teen Location Sharing",
            "feature_description": "Feature allows users under 18 to share location with ASL verification and T5 data processing",
            "context_documents": "Uses Jellybean parental controls and Snowcap framework"
        }
        
        # Execute
        result = await self.agent.process(state)
        
        # Assertions
        self.assertTrue(result["screening_completed"])
        self.assertEqual(result["next_step"], "research")
        
        analysis = result["screening_analysis"]
        self.assertEqual(analysis["agent"], "ScreeningAgent")
        self.assertEqual(analysis["risk_level"], "HIGH")
        self.assertTrue(analysis["age_sensitivity"])
        self.assertEqual(analysis["data_sensitivity"], "T5")
        self.assertTrue(analysis["needs_research"])
        
        # Check metadata
        self.assertIn("session_metadata", analysis)
        self.assertEqual(analysis["session_metadata"]["feature_name"], "Teen Location Sharing")
    
    @patch('app.agents.screening.ScreeningAgent.safe_llm_call')
    async def test_process_success_low_risk(self, mock_llm):
        """Test successful processing of low-risk feature"""
        mock_llm.return_value = {
            "agent": "ScreeningAgent", 
            "risk_level": "LOW",
            "compliance_required": False,
            "confidence": 0.85,
            "trigger_keywords": ["recommendation", "PF"],
            "reasoning": "Standard PF algorithm with minimal compliance impact",
            "needs_research": False,
            "geographic_scope": ["unknown"],
            "age_sensitivity": False,
            "data_sensitivity": "T2"
        }
        
        state = {
            "feature_name": "Basic Recommendations",
            "feature_description": "Standard PF algorithm for content recommendations using T2 data",
            "context_documents": ""
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["screening_completed"])
        self.assertEqual(result["next_step"], "validation")  # Skip research
        
        analysis = result["screening_analysis"]
        self.assertEqual(analysis["risk_level"], "LOW")
        self.assertFalse(analysis["age_sensitivity"])
        self.assertFalse(analysis["needs_research"])
    
    async def test_process_missing_description(self):
        """Test error handling for missing feature description"""
        state = {
            "feature_name": "Test Feature",
            "context_documents": "Some context"
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["screening_completed"])
        self.assertEqual(result["next_step"], "validation")
        
        analysis = result["screening_analysis"]
        self.assertEqual(analysis["risk_level"], "ERROR")
        self.assertIn("error", analysis)
        self.assertIn("Missing feature description", analysis["reasoning"])
    
    @patch('app.agents.screening.ScreeningAgent.safe_llm_call')
    async def test_process_llm_failure(self, mock_llm):
        """Test error handling when LLM call fails"""
        mock_llm.side_effect = Exception("LLM API Error")
        
        state = {
            "feature_name": "Test Feature",
            "feature_description": "Test description",
            "context_documents": ""
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["screening_completed"])
        analysis = result["screening_analysis"]
        self.assertEqual(analysis["risk_level"], "ERROR")
        self.assertIn("LLM API Error", analysis["reasoning"])
    
    def test_format_context_documents_string(self):
        """Test context document formatting - string input"""
        context = "This is a string context document"
        formatted = self.agent._format_context_documents(context)
        self.assertEqual(formatted, context)
    
    def test_format_context_documents_list(self):
        """Test context document formatting - list input"""
        context = [
            "First document content",
            {"title": "Second Doc", "content": "Second document content"}
        ]
        formatted = self.agent._format_context_documents(context)
        
        self.assertIn("**Document 1**:", formatted)
        self.assertIn("First document content", formatted)
        self.assertIn("**Second Doc**:", formatted)
        self.assertIn("Second document content", formatted)
    
    def test_format_context_documents_dict(self):
        """Test context document formatting - dict input"""
        context = {
            "Requirements": "Must comply with COPPA",
            "Technical": "Uses ASL and GH systems"
        }
        formatted = self.agent._format_context_documents(context)
        
        self.assertIn("**Requirements**:", formatted)
        self.assertIn("Must comply with COPPA", formatted)
        self.assertIn("**Technical**:", formatted)
    
    def test_format_context_documents_empty(self):
        """Test context document formatting - empty input"""
        formatted = self.agent._format_context_documents(None)
        self.assertEqual(formatted, "No additional context documents provided.")
        
        formatted = self.agent._format_context_documents("")
        self.assertEqual(formatted, "No additional context documents provided.")
    
    def test_enhance_result_validation(self):
        """Test result enhancement and validation"""
        # Mock LLM result with invalid data
        llm_result = {
            "risk_level": "INVALID",  # Invalid risk level
            "confidence": 2.0,        # Invalid confidence (>1.0)
            "geographic_scope": "global",  # Should be list
            "trigger_keywords": "not a list",  # Should be list
            "data_sensitivity": "INVALID"  # Invalid sensitivity
        }
        
        state = {"feature_name": "Test"}
        enhanced = self.agent._enhance_result(llm_result, state)
        
        # Check validation fixes
        self.assertEqual(enhanced["agent"], "ScreeningAgent")
        self.assertEqual(enhanced["risk_level"], "MEDIUM")  # Default for invalid
        self.assertEqual(enhanced["confidence"], 0.5)      # Fixed invalid confidence
        self.assertEqual(enhanced["geographic_scope"], ["global"])  # Converted to list
        self.assertEqual(enhanced["trigger_keywords"], [])  # Fixed to empty list
        self.assertEqual(enhanced["data_sensitivity"], "none")  # Default for invalid
        self.assertIn("session_metadata", enhanced)

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

# Convert async tests to sync for unittest
class TestScreeningAgentSync(AsyncTestCase):
    
    def setUp(self):
        self.agent = ScreeningAgent()
    
    @patch('app.agents.screening.ScreeningAgent.safe_llm_call')
    def test_process_high_risk_sync(self, mock_llm):
        """Sync version of high-risk test"""
        mock_llm.return_value = {
            "agent": "ScreeningAgent",
            "risk_level": "HIGH", 
            "compliance_required": True,
            "confidence": 0.9,
            "trigger_keywords": ["T5", "minors"],
            "reasoning": "High risk analysis",
            "needs_research": True,
            "geographic_scope": ["US"],
            "age_sensitivity": True,
            "data_sensitivity": "T5"
        }
        
        state = {
            "feature_name": "Test",
            "feature_description": "Test ASL feature with T5 data for minors",
            "context_documents": ""
        }
        
        result = self.run_async(self.agent.process(state))
        
        self.assertTrue(result["screening_completed"])
        self.assertEqual(result["screening_analysis"]["risk_level"], "HIGH")

if __name__ == '__main__':
    # Run with: python -m pytest test_screening_agent.py -v
    unittest.main()
