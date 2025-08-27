import logging

logging.getLogger("agent").setLevel(logging.INFO)

from .base import BaseComplianceAgent
from .screening import ScreeningAgent
from .orchestrator import ComplianceOrchestrator

# When research/validation agents are ready:
# from .research import ResearchAgent
# from .validation import ValidationAgent
# from .learning import LearningAgent

# Public APIs
__all__ = [
    "BaseComplianceAgent",
    "ScreeningAgent",
    "ComplianceOrchestrator",
]

# Package constants (edit later)
CONFIDENCE_THRESHOLD = 0.7
SUPPORTED_REGULATIONS = [
    "GDPR",
    "CCPA", 
    "COPPA",
    "Utah Social Media Regulation Act",
    "Florida Online Protections for Minors"
]