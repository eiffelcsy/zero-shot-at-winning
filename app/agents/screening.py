from pydantic import BaseModel, Field
from .base import BaseComplianceAgent
from .prompts.templates import build_screening_prompt
from langgraph.graph import END
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Ensure output data matches this data type
class ScreeningOutput(BaseModel):
    agent: str = Field(description="Agent name", default="ScreeningAgent")
    risk_level: str = Field(description="Risk level: LOW, MEDIUM, HIGH")
    compliance_required: bool = Field(description="Whether geo-compliance is required")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    trigger_keywords: List[str] = Field(description="Detected compliance keywords")
    reasoning: str = Field(description="Analysis reasoning")
    needs_research: bool = Field(description="Whether research agent is needed")
    geographic_scope: List[str] = Field(description="Geographic regions affected")
    age_sensitivity: bool = Field(description="Whether feature affects minors")
    data_sensitivity: str = Field(description="Data sensitivity level")

# Agent initialisation
class ScreeningAgent(BaseComplianceAgent):
    """First agent - analyzes features for compliance indicators"""
    
    def __init__(self, memory_overlay: str = ""):
        super().__init__("ScreeningAgent")
        self.memory_overlay = memory_overlay
        self.setup_prompts()
    
    def setup_prompts(self):
        """Setup LangChain prompt and chain with dynamic prompt building"""
        screening_prompt = build_screening_prompt(self.memory_overlay)
        self.create_chain(screening_prompt, ScreeningOutput)
    
    def update_memory(self, new_memory_overlay: str):
        """Allow runtime updates to the prompt for learning"""
        self.memory_overlay = new_memory_overlay
        self.setup_prompts()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature for compliance risk"""
        try:
            # Extract inputs from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            context_documents = state.get("context_documents", "")
            
            if not feature_name or not feature_description:
                raise ValueError("Missing feature name or description")
            
            # Prepare context documents string
            context_docs_str = self._format_context_documents(context_documents)
            
            # Run LLM analysis
            result = await self.safe_llm_call({
                "feature_name": feature_name,
                "feature_description": feature_description,
                "context_documents": context_docs_str
            })
            
            # Enhanced result with metadata
            enhanced_result = self._enhance_result(result, state)
            
            # Log interaction
            self.log_interaction(state, enhanced_result)
            
            # Return state update for LangGraph
            return {
                "screening_analysis": enhanced_result,
                "screening_completed": True,
                "screening_timestamp": datetime.now().isoformat(),
                "next_step": "research" if enhanced_result.get("needs_research", True) else "validation"
            }
            
        except Exception as e:
            self.logger.error(f"Screening agent failed: {e}")
            return {
                "screening_analysis": {
                    "agent": "ScreeningAgent",
                    "risk_level": "ERROR",
                    "compliance_required": None,
                    "confidence": 0.0,
                    "reasoning": f"Screening failed: {str(e)}",
                    "error": str(e),
                    "trigger_keywords": [],
                    "needs_research": False,
                    "geographic_scope": ["unknown"],
                    "age_sensitivity": False,
                    "data_sensitivity": "none"
                },
                "screening_completed": True,
                "next_step": END
            }
    
    def _format_context_documents(self, context_documents: Any) -> str:
        """Format context documents for the prompt"""
        if not context_documents:
            return "No additional context documents provided."
        
        if isinstance(context_documents, str):
            return context_documents
        
        if isinstance(context_documents, list):
            formatted_docs = []
            for i, doc in enumerate(context_documents, 1):
                if isinstance(doc, dict):
                    title = doc.get("title", f"Document {i}")
                    content = doc.get("content", str(doc))
                    formatted_docs.append(f"**{title}**:\n{content}")
                else:
                    formatted_docs.append(f"**Document {i}**:\n{str(doc)}")
            return "\n\n".join(formatted_docs)
        
        if isinstance(context_documents, dict):
            formatted_docs = []
            for key, value in context_documents.items():
                formatted_docs.append(f"**{key}**:\n{str(value)}")
            return "\n\n".join(formatted_docs)
        
        return str(context_documents)
    
    def _enhance_result(self, result: Dict, original_state: Dict) -> Dict:
        """Add metadata and validation to screening result"""
        
        # Ensure agent field is set
        result["agent"] = "ScreeningAgent"
        
        # Add session metadata
        result["session_metadata"] = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "feature_name": original_state.get("feature_name", ""),
            "has_context_docs": bool(original_state.get("context_documents"))
        }
        
        # Improved confidence score validation with logging
        result["confidence"] = self._validate_confidence_score(result.get("confidence"))
        
        # Ensure lists are properly formatted
        if not isinstance(result.get("geographic_scope"), list):
            if result.get("geographic_scope") == "global":
                result["geographic_scope"] = ["global"]
            elif result.get("geographic_scope") == "unknown":
                result["geographic_scope"] = ["unknown"]
            else:
                result["geographic_scope"] = ["unknown"]
        
        if not isinstance(result.get("trigger_keywords"), list):
            result["trigger_keywords"] = []
        
        # Validate boolean fields
        result["compliance_required"] = bool(result.get("compliance_required", False))
        result["age_sensitivity"] = bool(result.get("age_sensitivity", False))
        
        # Validate data sensitivity
        valid_sensitivities = ["T5", "T4", "T3", "T2", "T1", "none"]
        if result.get("data_sensitivity") not in valid_sensitivities:
            result["data_sensitivity"] = "none"
        
        # Validate risk level
        valid_risk_levels = ["LOW", "MEDIUM", "HIGH"]
        if result.get("risk_level") not in valid_risk_levels:
            result["risk_level"] = "MEDIUM"
        
        # Set research decision logic
        result["needs_research"] = (
            result.get("compliance_required", False) or 
            result.get("confidence", 0) < 0.8 or
            result.get("risk_level") in ["HIGH", "MEDIUM"]
        )
        
        return result

    def _validate_confidence_score(self, confidence: Any) -> float:
        """Validate and normalize confidence score with logging"""
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
            return float(confidence)
        
        # Log the invalid confidence for debugging
        self.logger.warning(f"Invalid confidence score received: {confidence}, type: {type(confidence)}")
        
        # Return neutral confidence instead of arbitrary 0.5
        return 0.5