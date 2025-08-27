"""Shared pytest fixtures for all tests"""
import pytest
import sys
import os
from unittest.mock import AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_llm():
    """Mock LLM that returns configurable responses"""
    llm = AsyncMock()
    llm.apredict.return_value = '{"risk_level": "HIGH", "compliance_required": true, "reasoning": "Test reasoning", "confidence": 0.85}'
    return llm

@pytest.fixture
def mock_llm_with_response():
    """Factory fixture for creating mock LLMs with custom responses"""
    def _create_mock(response):
        llm = AsyncMock()
        llm.apredict.return_value = response
        return llm
    return _create_mock

@pytest.fixture
def sample_features():
    """Sample feature descriptions for testing"""
    return [
        {
            "name": "Utah Curfew Feature",
            "description": "ASL-based curfew restrictions for Utah minors using GH"
        },
        {
            "name": "GDPR Compliance Feature", 
            "description": "T5 data processing with retention controls for EU users"
        },
        {
            "name": "Simple UI Feature",
            "description": "Update button colors and spacing"
        },
        {
            "name": "California Teens Feature",
            "description": "PF default toggle with NR enforcement for California teens"
        }
    ]