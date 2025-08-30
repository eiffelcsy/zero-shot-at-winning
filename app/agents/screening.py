from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
from datetime import datetime

from .prompts.screening_prompt import build_screening_prompt
from .base import BaseComplianceAgent
from langgraph.graph import END

class ScreeningOutput(BaseModel):
    agent: str = Field(description="Agent name")
    feature_name: str = Field(description="Name of the feature being analyzed")
    feature_description: str = Field(description="Description of the feature")
    compliance_risk_level: str = Field(description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    needs_research: bool = Field(description="Whether the feature needs further research")
    trigger_keywords: List[str] = Field(description="Keywords that triggered compliance flags")
    geographic_scope: List[str] = Field(description="Geographic regions affected")
    data_sensitivity: str = Field(description="Data sensitivity level")
    age_sensitivity: bool = Field(description="Whether the feature involves age-sensitive content")
    reasoning: str = Field(description="Detailed reasoning for the assessment")
    confidence_score: float = Field(description="Confidence in the assessment (0.0-1.0)")
    tiktok_terminology_used: bool = Field(description="Whether TikTok terminology was used in analysis")

class ScreeningAgent(BaseComplianceAgent):
    """First agent - analyzes features for compliance indicators with TikTok terminology context"""

    def __init__(self, memory_overlay: str = ""):
        super().__init__("ScreeningAgent", memory_overlay=memory_overlay)
        
        # Setup LangChain components with TikTok terminology context
        self._setup_chain()

    def _setup_chain(self):
        """Setup LangChain prompt and parser with TikTok terminology context integration"""
        screening_prompt = build_screening_prompt(self.memory_overlay)
        
        # Enhanced logging for memory overlay integration
        if self.memory_overlay:
            self.logger.info(f"Screening agent initialized with memory overlay ({len(self.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("✓ TikTok terminology found in memory overlay - screening will understand TikTok acronyms")
                self.logger.info("✓ Can properly assess: NR, PF, GH, CDS, DRT, LCP, Redline, Softblock, Spanner, ShadowMode, T5, ASL, Glow, NSP, Jellybean, EchoTrace, BB, Snowcap, FR, IMT")
            else:
                self.logger.warning("⚠ TikTok terminology NOT found in memory overlay - screening may miss TikTok-specific compliance indicators")
        else:
            self.logger.warning("⚠ Screening agent initialized with NO memory overlay - will lack TikTok terminology context")
        
        self.create_chain(screening_prompt, ScreeningOutput)
    
    def update_memory(self, new_memory_overlay: str):
        """Update memory overlay and rebuild chain with new TikTok terminology context"""
        self.logger.info(f"Updating screening agent memory overlay: {len(self.memory_overlay or '')} -> {len(new_memory_overlay)} characters")
        
        # Call parent method to update memory overlay
        super().update_memory(new_memory_overlay)
        
        # Rebuild the chain with new memory context
        screening_prompt = build_screening_prompt(new_memory_overlay)
        self.create_chain(screening_prompt, ScreeningOutput)
        
        self.logger.info("✓ Screening agent chain rebuilt with updated TikTok terminology context")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature for compliance risk with TikTok terminology context"""
        try:
            # Extract inputs from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            context_documents = state.get("context_documents", "")
            
            if not feature_name or not feature_description:
                raise ValueError("Missing feature name or description")
            
            # Log the screening analysis attempt
            self.logger.info(f"Starting compliance screening for feature: '{feature_name}'")
            self.logger.info(f"Feature description length: {len(feature_description)} characters")
            
            # Log memory overlay status for this analysis
            if self.memory_overlay:
                if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                    self.logger.info("✓ Screening analysis includes TikTok terminology context")
                else:
                    self.logger.warning("⚠ Screening analysis missing TikTok terminology context")
            else:
                self.logger.warning("⚠ Screening analysis has no memory overlay")
            
            # Prepare context documents string
            context_docs_str = self._format_context_documents(context_documents)
            
            # Prepare input for LLM analysis with TikTok terminology context
            screening_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "context_documents": context_docs_str
            }
            
            # Log the screening input preparation
            self.logger.info(f"Prepared screening input with {len(screening_input)} variables")
            
            # Run LLM analysis with TikTok terminology context
            result = await self.safe_llm_call(screening_input)
            
            # Enhanced result with metadata and TikTok terminology usage tracking
            enhanced_result = self._enhance_result(result, state)
            
            # Log successful screening completion
            self.logger.info(f"✓ Screening analysis completed for '{feature_name}'")
            self.logger.info(f"✓ Risk level: {enhanced_result.get('compliance_risk_level', 'UNKNOWN')}")
            self.logger.info(f"✓ Needs research: {enhanced_result.get('needs_research', True)}")
            self.logger.info(f"✓ TikTok terminology used: {enhanced_result.get('tiktok_terminology_used', False)}")
            
            # Log interaction with enhanced context
            self.log_interaction(state, enhanced_result)
            
            # Return state update for LangGraph
            return {
                "screening_analysis": enhanced_result,
                "screening_completed": True,
                "screening_timestamp": datetime.now().isoformat(),
                "next_step": "research" if enhanced_result.get("needs_research", True) else "validation"
            }
            
        except Exception as e:
            self.log_error(e, state, "Screening agent process failed")
            return {
                "screening_analysis": {
                    "agent": "ScreeningAgent",
                    "error": str(e),
                    "feature_name": state.get("feature_name", "unknown"),
                    "feature_description": state.get("feature_description", ""),
                    "compliance_risk_level": "ERROR",
                    "needs_research": True,
                    "trigger_keywords": [],
                    "geographic_scope": [],
                    "data_sensitivity": "unknown",
                    "age_sensitivity": False,
                    "reasoning": f"Screening failed due to error: {str(e)}",
                    "confidence_score": 0.0,
                    "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
                },
                "screening_completed": False,
                "screening_timestamp": datetime.now().isoformat(),
                "next_step": "research"
            }
    
    def _format_context_documents(self, context_documents: Any) -> str:
        """Format context documents for LLM input with TikTok terminology awareness"""
        if not context_documents:
            return "No additional context documents provided."
        
        try:
            if isinstance(context_documents, dict):
                content = context_documents.get("content", "")
                metadata = context_documents.get("metadata", {})
                
                # Log context document information
                self.logger.info(f"Processing context document: {metadata.get('filename', 'unknown')}")
                self.logger.info(f"Document size: {len(content)} characters")
                
                # Check if context contains TikTok terminology
                if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                    tiktok_terms = ["NR", "PF", "GH", "CDS", "DRT", "LCP", "Redline", "Softblock", "Spanner", "ShadowMode", "T5", "ASL", "Glow", "NSP", "Jellybean", "EchoTrace", "BB", "Snowcap", "FR", "IMT"]
                    found_terms = [term for term in tiktok_terms if term in content]
                    if found_terms:
                        self.logger.info(f"✓ Context document contains TikTok terminology: {found_terms}")
                    else:
                        self.logger.info("Context document contains no specific TikTok terminology")
                
                return f"Context Document Content:\n{content[:2000]}{'...' if len(content) > 2000 else ''}"
            
            elif isinstance(context_documents, str):
                self.logger.info(f"Processing string context document: {len(context_documents)} characters")
                return f"Context Document Content:\n{context_documents[:2000]}{'...' if len(context_documents) > 2000 else ''}"
            
            else:
                self.logger.warning(f"Unknown context document type: {type(context_documents)}")
                return "Context documents provided but in unsupported format."
                
        except Exception as e:
            self.logger.warning(f"Error formatting context documents: {e}")
            return "Error processing context documents."
    
    def _enhance_result(self, result: Dict, state: Dict) -> Dict:
        """Enhance screening result with metadata and TikTok terminology usage tracking"""
        try:
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                self.logger.warning(f"Expected dict result, got {type(result)}")
                result = {}
            
            # Extract feature information from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            
            # Enhanced result with additional metadata
            enhanced_result = {
                "agent": "ScreeningAgent",
                "feature_name": feature_name,
                "feature_description": feature_description,
                "compliance_risk_level": result.get("compliance_risk_level", "UNKNOWN"),
                "needs_research": result.get("needs_research", True),
                "trigger_keywords": result.get("trigger_keywords", []),
                "geographic_scope": result.get("geographic_scope", []),
                "data_sensitivity": result.get("data_sensitivity", "unknown"),
                "age_sensitivity": result.get("age_sensitivity", False),
                "reasoning": result.get("reasoning", ""),
                "confidence_score": float(result.get("confidence_score", 0.0)),
                "tiktok_terminology_used": self._check_tiktok_terminology_usage(result),
                "memory_overlay_length": len(self.memory_overlay) if self.memory_overlay else 0,
                "screening_timestamp": datetime.now().isoformat()
            }
            
            # Log the enhancement process
            self.logger.info(f"Enhanced screening result with {len(enhanced_result)} fields")
            self.logger.info(f"TikTok terminology usage detected: {enhanced_result['tiktok_terminology_used']}")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error enhancing screening result: {e}")
            # Return basic result if enhancement fails
            return {
                "agent": "ScreeningAgent",
                "error": f"Result enhancement failed: {str(e)}",
                "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
            }
    
    def _check_tiktok_terminology_usage(self, result: Dict) -> bool:
        """Check if the screening result used TikTok terminology in its analysis"""
        if not self.memory_overlay or "TIKTOK TERMINOLOGY REFERENCE" not in self.memory_overlay:
            return False
        
        # Check if any TikTok terminology appears in the result
        result_text = json.dumps(result, default=str).lower()
        
        # Common TikTok terminology patterns
        tiktok_patterns = [
            "nr", "pf", "gh", "cds", "drt", "lcp", "redline", "softblock", 
            "spanner", "shadowmode", "t5", "asl", "glow", "nsp", "jellybean", 
            "echotrace", "bb", "snowcap", "fr", "imt"
        ]
        
        for pattern in tiktok_patterns:
            if pattern in result_text:
                self.logger.info(f"✓ TikTok terminology '{pattern.upper()}' detected in screening result")
                return True
        
        # Check reasoning field specifically
        reasoning = result.get("reasoning", "").lower()
        for pattern in tiktok_patterns:
            if pattern in reasoning:
                self.logger.info(f"✓ TikTok terminology '{pattern.upper()}' detected in reasoning")
                return True
        
        self.logger.info("No TikTok terminology detected in screening result")
        return False