"""
Multi-agent compliance analysis system
"""

import logging

# Configure logging for all agents
logging.getLogger("agent").setLevel(logging.INFO)

# Import base class and implemented agents
from .base import BaseComplianceAgent
from .screening import ScreeningAgent

# Will add more as you implement them:
# from .research import ResearchAgent
# from .validation import ValidationAgent
# from .orchestrator import ComplianceOrchestrator

__all__ = [
    "BaseComplianceAgent",
    "ScreeningAgent",
]

# Package constants
CONFIDENCE_THRESHOLD = 0.7
SUPPORTED_REGULATIONS = [
    "GDPR",
    "CCPA", 
    "COPPA",
    "Utah Social Media Regulation Act",
    "Florida Online Protections for Minors"
]