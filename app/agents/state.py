from typing_extensions import TypedDict
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime

class ComplianceState(TypedDict):
    # Input fields
    feature_name: str
    feature_description: str
    session_id: Optional[str]
    
    # Screening agent outputs
    screening_analysis: Optional[Dict[str, Any]]
    screening_completed: Optional[bool]
    screening_timestamp: Optional[str]
    
    # Research agent outputs  
    research_evidence: Optional[List[Dict]]
    research_completed: Optional[bool]
    
    # Validation agent outputs
    final_decision: Optional[Dict[str, Any]]
    validation_completed: Optional[bool]
    
    # Flow control
    next_step: Optional[str]  # "research", "validation", or "complete"
    
    # Metadata
    workflow_started: Optional[str]
    workflow_completed: Optional[str]
    confidence_score: Optional[float]

class AgentMessage(BaseModel):
    agent_name: str
    analysis_result: Dict
    confidence: float
    next_agent: Optional[str]
    metadata: Dict