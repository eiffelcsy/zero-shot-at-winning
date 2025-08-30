from langgraph.graph import StateGraph, START, END
from .state import ComplianceState, create_initial_state
from .screening import ScreeningAgent
from .research import ResearchAgent
from .validation import ValidationAgent
# from .learning import LearningAgent
from .memory.combined_memory import get_all_combined_overlays
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from logs.logging_config import get_logger

class ComplianceOrchestrator:
    """LangGraph-powered multi-agent orchestrator with memory overlay support"""
    
    def __init__(self, memory_overlay: str = "", use_combined_memory: bool = True):
        # Store configuration
        self.memory_overlay = memory_overlay
        self.use_combined_memory = use_combined_memory
        
        # Initialize logger
        self.logger = get_logger(__name__)

        # Initialize agents with agent-specific memory overlays
        if use_combined_memory:
            # Get combined overlays (TikTok + agent-specific few-shot examples)
            combined_overlays = get_all_combined_overlays()
            self.screening_agent = ScreeningAgent(memory_overlay=combined_overlays["screening"])
            self.research_agent = ResearchAgent(memory_overlay=combined_overlays["research"])
            self.validation_agent = ValidationAgent(memory_overlay=combined_overlays["validation"])
        else:
            # Use traditional single memory overlay for all agents
            self.screening_agent = ScreeningAgent(memory_overlay=memory_overlay)
            self.research_agent = ResearchAgent(memory_overlay=memory_overlay)
            self.validation_agent = ValidationAgent(memory_overlay=memory_overlay)
        # self.learning_agent = LearningAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def update_agent_memory(self, new_memory_overlay: str = None):
        """Update memory overlay for all agents and rebuild workflow"""
        if new_memory_overlay:
            self.memory_overlay = new_memory_overlay
        
        if self.use_combined_memory:
            # Get fresh combined overlays
            combined_overlays = get_all_combined_overlays()
            self.screening_agent.update_memory(combined_overlays["screening"])
            self.research_agent.update_memory(combined_overlays["research"])
            self.validation_agent.update_memory(combined_overlays["validation"])
        else:
            # Update each agent's memory with the same overlay
            self.screening_agent.update_memory(self.memory_overlay)
            self.research_agent.update_memory(self.memory_overlay)
            self.validation_agent.update_memory(self.memory_overlay)
        
        # Rebuild the workflow with updated agents
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ComplianceState)
        
        # Add nodes - agents handle LangGraph state directly
        workflow.add_node("screening", self.screening_agent.process)
        workflow.add_node("research", self.research_agent.process)
        workflow.add_node("validation", self.validation_agent.process)
        # workflow.add_node("learning", self.learning_agent.process)
        
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
        workflow.add_edge("validation", END)
        # workflow.add_edge("learning", END)
        
        return workflow.compile()
    
    def _route_after_screening(self, state: ComplianceState) -> str:
        """Determine next step after screening"""
        screening_analysis = state.get("screening_analysis", {})
        
        # End on error
        if screening_analysis.get("error"):
            return END
        
        # Route based on research need
        if screening_analysis.get("needs_research", True):
            return "research"
        else:
            return END
    
    def _calculate_final_confidence(self, 
                                    screening_analysis: Dict, 
                                    research_analysis: Dict, 
                                    validation_analysis: Dict) -> float:
        """
        Calculate final confidence score using a weighted formula combining all agents.
        
        Formula:
        - Screening: 30% weight (initial assessment quality)
        - Research: 40% weight (evidence quality and quantity)
        - Validation: 30% weight (final decision confidence)
        """
        try:
            # Extract individual confidence scores
            screening_confidence = float(screening_analysis.get("confidence_score", 0.0))
            research_confidence = float(research_analysis.get("confidence_score", 0.0))
            validation_confidence = float(validation_analysis.get("confidence_score", 0.0)) if validation_analysis else 0.0
            
            # Calculate weighted final confidence
            final_confidence = (
                (screening_confidence * 0.3) +    # 30% weight
                (research_confidence * 0.4) +     # 40% weight
                (validation_confidence * 0.3)     # 30% weight
            )
            
            # Log the calculation
            self.logger.info(f"Final confidence calculation:")
            self.logger.info(f"  Screening: {screening_confidence:.3f} × 0.3 = {screening_confidence * 0.3:.3f}")
            self.logger.info(f"  Research: {research_confidence:.3f} × 0.4 = {research_confidence * 0.4:.3f}")
            self.logger.info(f"  Validation: {validation_confidence:.3f} × 0.3 = {validation_confidence * 0.3:.3f}")
            self.logger.info(f"  Final: {final_confidence:.3f}")
            
            return min(max(final_confidence, 0.0), 1.0)  # Ensure between 0.0 and 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating final confidence: {e}")
            # Fallback to average of available scores
            available_scores = []
            if screening_analysis:
                available_scores.append(float(screening_analysis.get("confidence_score", 0.0)))
            if research_analysis:
                available_scores.append(float(research_analysis.get("confidence_score", 0.0)))
            if validation_analysis:
                available_scores.append(float(validation_analysis.get("confidence_score", 0.0)))
            
            if available_scores:
                fallback_confidence = sum(available_scores) / len(available_scores)
                self.logger.info(f"Using fallback confidence: {fallback_confidence:.3f}")
                return fallback_confidence
            else:
                return 0.0
    
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
            self.logger.info(f"Final state keys: {list(final_state.keys())}")
            self.logger.info(f"Final state: {final_state}")
            
            # Extract and format final result
            screening_analysis = final_state.get("screening_analysis", {})
            research_analysis = final_state.get("research_analysis", {})
            validation_analysis = final_state.get("validation_analysis", {})
            
            # Determine which agents completed
            agents_completed = []
            if final_state.get("screening_completed", False):
                agents_completed.append("screening")
            if final_state.get("research_completed", False):
                agents_completed.append("research")
            if final_state.get("validation_completed", False):
                agents_completed.append("validation")
            
            # Extract final decision - prefer validation over screening
            if validation_analysis:
                needs_geo_logic = validation_analysis.get("needs_geo_logic", "UNKNOWN")
                reasoning = validation_analysis.get("reasoning", "")
                related_regulations = validation_analysis.get("related_regulations", [])
            else:
                needs_geo_logic = screening_analysis.get("compliance_required", "UNKNOWN")
                reasoning = screening_analysis.get("reasoning", "")
                related_regulations = []

            # Calculate final confidence using formula
            confidence_score = self._calculate_final_confidence(
                screening_analysis, 
                research_analysis, 
                validation_analysis
            )
            
            # Extract jurisdictions from research results
            applicable_jurisdictions = []
            if research_analysis and "regulations" in research_analysis:
                for reg in research_analysis["regulations"]:
                    if isinstance(reg, dict) and "jurisdiction" in reg:
                        jurisdiction = reg["jurisdiction"]
                        if jurisdiction and jurisdiction not in applicable_jurisdictions:
                            applicable_jurisdictions.append(jurisdiction)
            
            return {
                "session_id": final_state.get("session_id"),
                "feature_name": feature_name,
                "needs_geo_logic": needs_geo_logic,
                "reasoning": reasoning,
                "related_regulations": related_regulations,
                "confidence_score": confidence_score,
                "risk_level": screening_analysis.get("risk_level", "UNKNOWN"),
                "workflow_completed": final_state.get("workflow_completed", False),
                "agents_completed": agents_completed,
                "evidence_sources": len(research_analysis.get("regulations", [])) if research_analysis else 0,
                "research_confidence": research_analysis.get("confidence_score", 0.0) if research_analysis else 0.0,
                "applicable_jurisdictions": applicable_jurisdictions,
                
                # Pass through agent-specific analyses for UI display
                "screening_analysis": screening_analysis,
                "research_analysis": research_analysis,
                "validation_analysis": validation_analysis,
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
                "evidence_sources": 0,
                "needs_geo_logic": "UNKNOWN",
                "reasoning": f"Analysis failed due to workflow error: {str(e)}",
                "related_regulations": [],
                "research_confidence": 0.0,
                "applicable_jurisdictions": [],
                
                # Empty agent analyses for error state
                "screening_analysis": None,
                "research_analysis": None,
                "validation_analysis": None,
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow configuration status"""
        return {
            "has_memory_overlay": bool(self.memory_overlay),
            "memory_overlay_length": len(self.memory_overlay),
            "use_combined_memory": self.use_combined_memory,
            "agents_initialized": {
                "screening": bool(self.screening_agent),
                "research": bool(self.research_agent),
                "validation": bool(self.validation_agent),
                # "learning": bool(self.learning_agent)
            },
            "agent_memory_info": {
                "screening_overlay_length": len(self.screening_agent.memory_overlay) if self.screening_agent else 0,
                "research_overlay_length": len(self.research_agent.memory_overlay) if self.research_agent else 0,
                "validation_overlay_length": len(self.validation_agent.memory_overlay) if self.validation_agent else 0,
            }
        }