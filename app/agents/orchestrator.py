from langgraph.graph import StateGraph, START, END
from .state import ComplianceState, create_initial_state
from .screening import ScreeningAgent
from .research import ResearchAgent
from .validation import ValidationAgent
from .learning import LearningAgent
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

class ComplianceOrchestrator:
    """LangGraph-powered multi-agent orchestrator with memory overlay support"""
    
    def __init__(self, 
                chroma_host: str = "localhost", 
                chroma_port: int = 8001, 
                memory_overlay: str = ""):
        # Store configuration
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.memory_overlay = memory_overlay
        
        # Initialize agents with memory overlay support
        self.screening_agent = ScreeningAgent(memory_overlay=memory_overlay)
        self.research_agent = ResearchAgent(
            chroma_host=chroma_host, 
            chroma_port=chroma_port,
            memory_overlay=memory_overlay
        )
        self.validation_agent = ValidationAgent(memory_overlay=memory_overlay)
        self.learning_agent = LearningAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def update_agent_memory(self, new_memory_overlay: str):
        """Update memory overlay for all agents and rebuild workflow"""
        self.memory_overlay = new_memory_overlay
        
        # Update each agent's memory
        self.screening_agent.update_memory(new_memory_overlay)
        self.research_agent.update_memory(new_memory_overlay)
        self.validation_agent.update_memory(new_memory_overlay)
        
        # Rebuild the workflow with updated agents
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
    
    async def analyze_feature(self, 
                            feature_name: str, 
                            feature_description: str, 
                            context_documents: Optional[Any] = None) -> Dict[str, Any]:
        """Main entry point for feature analysis"""
        
        # Create initial state using helper function
        initial_state = create_initial_state(
            feature_name=feature_name,
            feature_description=feature_description,
            context_documents=context_documents,
            session_id=f"compliance_{uuid.uuid4().hex[:8]}"
        )
        
        try:
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Extract and format final result
            final_decision = final_state.get("final_decision", {})
            screening_analysis = final_state.get("screening_analysis", {})
            research_analysis = final_state.get("research_analysis", {})
            
            return {
                "session_id": final_state.get("session_id"),
                "feature_name": feature_name,
                "needs_geo_logic": final_decision.get("needs_geo_logic", "UNKNOWN"),
                "reasoning": final_decision.get("reasoning", screening_analysis.get("reasoning", "")),
                "related_regulations": final_decision.get("related_regulations", []),
                "confidence_score": final_decision.get("confidence", screening_analysis.get("confidence", 0.0)),
                "risk_level": screening_analysis.get("risk_level", "UNKNOWN"),
                "workflow_completed": final_state.get("workflow_completed"),
                "agents_completed": [
                    agent for agent in ["screening", "research", "validation", "learning"]
                    if final_state.get(f"{agent}_completed", False)
                ],
                "evidence_sources": len(final_state.get("research_evidence", [])),
                "research_confidence": final_state.get("research_confidence", 0.0),
                "applicable_jurisdictions": [
                    jur.get("jurisdiction", "") for jur in 
                    final_state.get("research_jurisdictions", [])
                ]
            }
            
        except Exception as e:
            # Return error response
            return {
                "session_id": initial_state.get("session_id"),
                "feature_name": feature_name,
                "error": f"Workflow failed: {str(e)}",
                "workflow_completed": False,
                "agents_completed": [],
                "risk_level": "ERROR",
                "confidence_score": 0.0,
                "evidence_sources": 0
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow configuration status"""
        return {
            "chroma_host": self.chroma_host,
            "chroma_port": self.chroma_port,
            "has_memory_overlay": bool(self.memory_overlay),
            "memory_overlay_length": len(self.memory_overlay),
            "agents_initialized": {
                "screening": bool(self.screening_agent),
                "research": bool(self.research_agent),
                "validation": bool(self.validation_agent),
                "learning": bool(self.learning_agent)
            }
        }