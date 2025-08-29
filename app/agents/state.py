from typing_extensions import TypedDict
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime


class ComplianceState(TypedDict):
    # Input fields
    feature_name: str
    feature_description: str
    context_documents: Optional[Any]
    session_id: Optional[str]
    
    # Screening agent outputs
    screening_analysis: Optional[Dict[str, Any]]
    screening_completed: Optional[bool]
    screening_timestamp: Optional[str]
    
    # Research agent outputs - updated to match research.py naming
    research_regulations: Optional[List[Dict[str, Any]]]
    research_queries: Optional[List[str]]
    research_confidence: Optional[float]
    research_retrieved_documents: Optional[List[Dict[str, Any]]]
    research_analysis: Optional[Dict[str, Any]]
    research_completed: Optional[bool]
    research_timestamp: Optional[str]
    research_error: Optional[str]
    
    # Enhanced research outputs with ChromaDB
    research_risk_assessment: Optional[Dict[str, Any]]
    research_jurisdictions: Optional[List[Dict]]
    research_compliance_patterns: Optional[List[str]]
    
    # Validation agent outputs (for future implementation)
    validation_analysis: Optional[Dict[str, Any]]
    final_decision: Optional[Dict[str, Any]]
    validation_completed: Optional[bool]
    validation_timestamp: Optional[str]
    validation_error: Optional[str]
    
    # Flow control
    next_step: Optional[str]  # "research", "validation", or "complete"
    
    # Metadata
    workflow_started: Optional[str]
    workflow_completed: Optional[str]
    confidence_score: Optional[float]  # Overall workflow confidence
    
    # Error handling
    workflow_errors: Optional[List[str]]  # Track any errors during workflow
    workflow_status: Optional[str]  # "running", "completed", "failed"


class AgentMessage(BaseModel):
    """Standard message format for agent communications"""
    agent_name: str
    analysis_result: Dict[str, Any]
    confidence: float
    next_agent: Optional[str]
    metadata: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)


class ScreeningResult(BaseModel):
    """Typed model for screening agent results"""
    agent: str
    risk_level: str  # LOW, MEDIUM, HIGH
    compliance_required: bool
    confidence: float
    trigger_keywords: List[str]
    reasoning: str
    needs_research: bool
    geographic_scope: List[str]
    age_sensitivity: bool
    data_sensitivity: str  # T5, T4, T3, T2, T1, none


class ResearchResult(BaseModel):
    """Typed model for research agent results - updated to match research.py naming"""
    agent: str
    regulations: List[Dict[str, Any]]  # Changed from candidates
    evidence: List[Dict[str, Any]]
    query_used: str
    confidence_score: float


class ValidationResult(BaseModel):
    """Typed model for validation agent results (future use)"""
    agent: str
    validation_status: str  # CONFIRMED, DISPUTED, UNCLEAR
    final_risk_level: str  # LOW, MEDIUM, HIGH
    final_compliance_required: bool
    confidence: float
    conflicts_found: List[str]
    reasoning: str
    recommendations: Optional[List[str]] = None


class WorkflowMetadata(BaseModel):
    """Metadata tracking for the entire workflow"""
    session_id: str
    feature_name: str
    started_at: str
    completed_at: Optional[str] = None
    total_agents_run: int = 0
    agents_completed: List[str] = []
    agents_failed: List[str] = []
    overall_confidence: Optional[float] = None
    status: str = "initialized"  # initialized, running, completed, failed


# Helper functions for state management
def create_initial_state(
    feature_name: str, 
    feature_description: str, 
    context_documents: Optional[Any] = None,
    session_id: Optional[str] = None
) -> ComplianceState:
    """Create initial state for compliance workflow"""
    return ComplianceState(
        # Input fields
        feature_name=feature_name,
        feature_description=feature_description,
        context_documents=context_documents,
        session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        
        # Initialize all other fields as None
        screening_analysis=None,
        screening_completed=None,
        screening_timestamp=None,
        
        # Updated research field names to match research.py
        research_regulations=None,
        research_queries=None,
        research_confidence=None,
        research_retrieved_documents=None,
        research_analysis=None,
        research_completed=None,
        research_timestamp=None,
        research_error=None,
        
        research_risk_assessment=None,
        research_jurisdictions=None,
        research_compliance_patterns=None,
        
        validation_analysis=None,
        final_decision=None,
        validation_completed=None,
        validation_timestamp=None,
        validation_error=None,
        
        next_step="screening",  # Start with screening
        workflow_started=datetime.now().isoformat(),
        workflow_completed=None,
        confidence_score=None,
        workflow_errors=None,
        workflow_status="initialized"
    )


def update_state(current_state: ComplianceState, updates: Dict[str, Any]) -> ComplianceState:
    """Helper function to update state with new values"""
    updated_state = current_state.copy()
    updated_state.update(updates)
    return updated_state


def is_workflow_complete(state: ComplianceState) -> bool:
    """Check if the workflow is complete"""
    return (
        state.get("next_step") == "complete" or 
        state.get("workflow_status") == "completed" or
        state.get("workflow_completed") is not None
    )


def get_workflow_summary(state: ComplianceState) -> Dict[str, Any]:
    """Generate a summary of the workflow execution"""
    summary = {
        "session_id": state.get("session_id"),
        "feature_name": state.get("feature_name"),
        "workflow_status": state.get("workflow_status", "unknown"),
        "started_at": state.get("workflow_started"),
        "completed_at": state.get("workflow_completed"),
        "agents_completed": []
    }
    
    # Track which agents completed
    if state.get("screening_completed"):
        summary["agents_completed"].append("screening")
    if state.get("research_completed"):
        summary["agents_completed"].append("research")
    if state.get("validation_completed"):
        summary["agents_completed"].append("validation")
    
    # Add final results if available
    if state.get("screening_analysis"):
        summary["final_risk_level"] = state["screening_analysis"].get("risk_level")
        summary["compliance_required"] = state["screening_analysis"].get("compliance_required")
    
    # Updated to use new field names
    if state.get("research_analysis"):
        summary["regulations_found"] = len(state["research_analysis"].get("regulations", []))
        summary["evidence_pieces"] = len(state["research_analysis"].get("evidence", []))
    
    summary["overall_confidence"] = state.get("confidence_score")
    summary["errors"] = state.get("workflow_errors", [])
    
    return summary


# Constants for workflow control
class WorkflowSteps:
    SCREENING = "screening"
    RESEARCH = "research" 
    VALIDATION = "validation"
    COMPLETE = "complete"
    FAILED = "failed"


class WorkflowStatus:
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"