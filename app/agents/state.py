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
    
    # Research agent outputs  
    research_analysis: Optional[Dict[str, Any]]
    research_completed: Optional[bool]
    research_timestamp: Optional[str]
    
    # Validation agent outputs
    validation_analysis: Optional[Dict[str, Any]]
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
        
        research_analysis=None,
        research_completed=None,
        research_timestamp=None,
        
        validation_analysis=None,
        validation_completed=None,
        validation_timestamp=None,
        
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
        summary["evidence_pieces"] = len(state["research_analysis"].get("regulations", []))
    
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