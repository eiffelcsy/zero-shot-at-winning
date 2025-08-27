from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from .screening import LangGraphScreeningAgent
from .state import ComplianceState
import uuid
from datetime import datetime

class ComplianceWorkflow:
    def __init__(self):
        self.screening_agent = LangGraphScreeningAgent()
        # TODO: Add research and validation agents
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        # Create the graph
        workflow = StateGraph(ComplianceState)
        
        # Add nodes
        workflow.add_node("screening", self._screening_node)
        # TODO: Add research and validation nodes
        
        # Define flow
        workflow.add_edge(START, "screening")
        
        # Conditional edges based on screening result
        workflow.add_conditional_edges(
            "screening",
            self._route_after_screening,
            {
                "research": "research",  # Will add later
                "validation": "validation",  # Will add later
                "complete": END
            }
        )
        
        return workflow.compile()
    
    async def _screening_node(self, state: ComplianceState) -> ComplianceState:
        """Screening node for LangGraph"""
        # Add session metadata
        if not state.get("session_id"):
            state["session_id"] = f"compliance_{uuid.uuid4().hex[:8]}"
        
        if not state.get("workflow_started"):
            state["workflow_started"] = datetime.now().isoformat()
        
        # Run screening agent
        screening_updates = await self.screening_agent.process(state)
        
        # Merge updates into state
        return {**state, **screening_updates}
    
    def _route_after_screening(self, state: ComplianceState) -> str:
        """Determine next step after screening"""
        screening_result = state.get("screening_result", {})
        
        if screening_result.get("error"):
            return "complete"  # End on error
        
        if screening_result.get("needs_research", True):
            return "research"
        else:
            return "validation"
    
    async def analyze_feature(self, feature_name: str, feature_description: str) -> Dict[str, Any]:
        """Main entry point"""
        initial_state = ComplianceState(
            feature_name=feature_name,
            feature_description=feature_description
        )
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Extract final result
        return {
            "session_id": final_state.get("session_id"),
            "feature_name": feature_name,
            "screening_result": final_state.get("screening_result"),
            "workflow_completed": datetime.now().isoformat()
        }