from typing_extensions import TypedDict
from typing import Dict, Any, Optional, List
from datetime import datetime

class ComplianceState(TypedDict):
    # Input fields
    feature_name: str
    feature_description: str
    session_id: Optional[str]
    
    # Screening agent outputs
    screening_result: Optional[Dict[str, Any]]
    screening_completed: Optional[bool]
    screening_timestamp: Optional[str]
    screening_error: Optional[str]
    
    # Research agent outputs (for later)
    research_evidence: Optional[List[Dict]]
    research_completed: Optional[bool]
    
    # Validation agent outputs (for later)
    final_decision: Optional[Dict[str, Any]]
    validation_completed: Optional[bool]
    
    # Flow control
    next_step: Optional[str]  # "research", "validation", or "complete"
    
    # Metadata
    workflow_started: Optional[str]
    workflow_completed: Optional[str]
    confidence_score: Optional[float]