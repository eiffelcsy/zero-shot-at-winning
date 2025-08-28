from langgraph.graph import StateGraph, START, END
from .state import ComplianceState
from .screening import ScreeningAgent
from .research import ResearchAgent
from .validation import ValidationAgent
from .learning import LearningAgent
from typing import Dict, Any
import uuid
from datetime import datetime

class ComplianceOrchestrator:
    """LangGraph-powered multi-agent orchestrator with direct agent integration"""
    
    def __init__(self, kb_dir: str = "data/kb"):
        # Initialize agents directly
        self.screening_agent = ScreeningAgent()
        self.research_agent = ResearchAgent(kb_dir=kb_dir)
        self.validation_agent = ValidationAgent()
        self.learning_agent = LearningAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ComplianceState)
        
        # Add nodes - agents handle LangGraph state directly
        workflow.add_node("screening", self.screening_agent.process)
        workflow.add_node("research", self.research_agent.process)
        workflow.add_node("validation", self.validation_agent.process)
        workflow.add_node("learning", self.learning_agent.process)
        
        # Define flow
        workflow.add_edge(START, "screening")
        
        # Conditional routing after screening
        workflow.add_conditional_edges(
            "screening",
            self._route_after_screening,
            {
                "research": "research",
                "validation": "validation",
                "end": END
            }
        )
        
        workflow.add_edge("research", "validation")
        workflow.add_edge("validation", 'learning')
        workflow.add_edge("learning", END)
        
        return workflow.compile()
    
    def _route_after_screening(self, state: ComplianceState) -> str:
        """Determine next step after screening"""
        screening_analysis = state.get("screening_analysis", {})
        
        # End on error
        if screening_analysis.get("error"):
            return "end"
        
        # Route based on research need
        if screening_analysis.get("needs_research", True):
            return "research"
        else:
            return "validation"
    
    async def analyze_feature(self, feature_name: str, feature_description: str) -> Dict[str, Any]:
        """Main entry point for feature analysis"""
        
        # Create initial state
        initial_state = ComplianceState(
            feature_name=feature_name,
            feature_description=feature_description,
            session_id=f"compliance_{uuid.uuid4().hex[:8]}",
            workflow_started=datetime.now().isoformat()
        )
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Extract and format final result
        final_decision = final_state.get("final_decision", {})
        screening_analysis = final_state.get("screening_analysis", {})
        
        return {
            "session_id": final_state.get("session_id"),
            "feature_name": feature_name,
            "needs_geo_logic": final_decision.get("needs_geo_logic", "UNKNOWN"),
            "reasoning": final_decision.get("reasoning", ""),
            "related_regulations": final_decision.get("related_regulations", []),
            "confidence_score": final_decision.get("confidence", 0.0),
            "risk_level": screening_analysis.get("risk_level", "UNKNOWN"),
            "workflow_completed": final_state.get("workflow_completed"),
            "agents_used": ["screening", "research", "validation"],
            "evidence_sources": len(final_state.get("research_evidence", []))
        }