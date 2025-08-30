from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, HttpUrl
from .base import BaseComplianceAgent
from .prompts.validation_prompt import build_validation_prompt
from .memory.memory_pg import PostgresMemoryStore
from typing import Dict, Any, List, Literal
from datetime import datetime
import json, os, sys

class RelatedRegulation(BaseModel):
    name: str = Field(description="Regulation name")
    jurisdiction: str = Field(description="Jurisdiction")
    section: str = Field(description="Specific section")
    url: HttpUrl = Field(description="URL to regulation")
    evidence_excerpt: str = Field(description="Supporting evidence text")

class ValidationOutput(BaseModel):
    agent: str = Field(description="Agent name")
    feature_name: str = Field(description="Name of the feature being validated")
    final_decision: str = Field(description="Final compliance decision: COMPLIANT, NON_COMPLIANT, NEEDS_REVIEW")
    confidence_score: float = Field(description="Confidence in the decision (0.0-1.0)")
    reasoning: str = Field(description="Detailed reasoning for the final decision")
    compliance_requirements: List[str] = Field(description="List of compliance requirements that must be met")
    risk_assessment: str = Field(description="Overall risk assessment")
    recommendations: List[str] = Field(description="Recommendations for compliance")
    tiktok_terminology_used: bool = Field(description="Whether TikTok terminology was used in validation")

class ValidationAgent(BaseComplianceAgent):
    """Final decision-maker agent - validates compliance requirements with TikTok terminology context"""

    def __init__(self, memory_overlay: str = ""):
        super().__init__("ValidationAgent", memory_overlay=memory_overlay)
        
        # Setup LangChain components with TikTok terminology context
        self._setup_chain()

    def _setup_chain(self):
        """Setup LangChain prompt and parser with TikTok terminology context integration"""
        validation_prompt = build_validation_prompt(self.memory_overlay)
        
        # Enhanced logging for memory overlay integration
        if self.memory_overlay:
            self.logger.info(f"Validation agent initialized with memory overlay ({len(self.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("✓ TikTok terminology found in memory overlay - validation will understand TikTok acronyms")
                self.logger.info("✓ Can properly validate: NR, PF, GH, CDS, DRT, LCP, Redline, Softblock, Spanner, ShadowMode, T5, ASL, Glow, NSP, Jellybean, EchoTrace, BB, Snowcap, FR, IMT")
            else:
                self.logger.warning("⚠ TikTok terminology NOT found in memory overlay - validation may miss TikTok-specific compliance requirements")
        else:
            self.logger.warning("⚠ Validation agent initialized with NO memory overlay - will lack TikTok terminology context")
        
        self.create_chain(validation_prompt, ValidationOutput)
    
    def update_memory(self, new_memory_overlay: str):
        """Update memory overlay and rebuild chain with new TikTok terminology context"""
        self.logger.info(f"Updating validation agent memory overlay: {len(self.memory_overlay or '')} -> {len(new_memory_overlay)} characters")
        
        # Call parent method to update memory overlay
        super().update_memory(new_memory_overlay)
        
        # Rebuild the chain with new memory context
        validation_prompt = build_validation_prompt(new_memory_overlay)
        self.create_chain(validation_prompt, ValidationOutput)
        
        self.logger.info("✓ Validation agent chain rebuilt with updated TikTok terminology context")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible process method with TikTok terminology context"""
        try:
            # Extract data from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            screening_analysis = state.get("screening_analysis", {})
            research_analysis = state.get("research_analysis", {})
            
            if not feature_name or not feature_description:
                raise ValueError("Missing feature name or description")
            
            if not screening_analysis:
                raise ValueError("Missing screening analysis from previous agent")
            
            if not research_analysis or not research_analysis.get("regulations"):
                raise ValueError("Missing research analysis from previous agent")

            # Log the validation attempt
            self.logger.info(f"Starting compliance validation for feature: '{feature_name}'")
            self.logger.info(f"Screening analysis present: {bool(screening_analysis)}")
            self.logger.info(f"Research analysis present: {bool(research_analysis)}")
            self.logger.info(f"Regulations found: {len(research_analysis.get('regulations', []))}")
            
            # Log memory overlay status for this validation
            if self.memory_overlay:
                if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                    self.logger.info("✓ Validation analysis includes TikTok terminology context")
                else:
                    self.logger.warning("⚠ Validation analysis missing TikTok terminology context")
            else:
                self.logger.warning("⚠ Validation analysis has no memory overlay")

            # Prepare input for validation with TikTok terminology context
            validation_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "research_analysis": json.dumps(research_analysis, indent=2)
            }
            
            # Log the validation input preparation
            self.logger.info(f"Prepared validation input with {len(validation_input)} variables")
            
            # Get LLM decision with TikTok terminology context
            result = await self.safe_llm_call(validation_input)
            
            # Enhanced result with metadata and TikTok terminology usage tracking
            enhanced_result = self._enhance_result(result, state)
            
            # Log successful validation completion
            self.logger.info(f"✓ Validation analysis completed for '{feature_name}'")
            self.logger.info(f"✓ Final decision: {enhanced_result.get('final_decision', 'UNKNOWN')}")
            self.logger.info(f"✓ Confidence score: {enhanced_result.get('confidence_score', 0.0)}")
            self.logger.info(f"✓ TikTok terminology used: {enhanced_result.get('tiktok_terminology_used', False)}")
            
            # Log interaction with enhanced context
            self.log_interaction(validation_input, enhanced_result)
            
            # Return final decision for LangGraph in enhanced format
            return {
                "validation_analysis": enhanced_result,
                "final_decision": enhanced_result,
                "validation_completed": True,
                "validation_timestamp": datetime.now().isoformat(),
                "next_step": "complete"
            }
            
        except Exception as e:
            self.log_error(e, state, "Validation agent process failed")
            return {
                "validation_analysis": {
                    "agent": "ValidationAgent",
                    "error": str(e),
                    "feature_name": state.get("feature_name", "unknown"),
                    "final_decision": "ERROR",
                    "confidence_score": 0.0,
                    "reasoning": f"Validation failed due to error: {str(e)}",
                    "compliance_requirements": [],
                    "risk_assessment": "ERROR",
                    "recommendations": ["Fix validation error and retry"],
                    "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
                },
                "final_decision": {
                    "agent": "ValidationAgent",
                    "error": str(e),
                    "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
                },
                "validation_completed": False,
                "validation_timestamp": datetime.now().isoformat(),
                "next_step": "complete"
            }
    
    def _enhance_result(self, result: Dict, state: Dict) -> Dict:
        """Enhance validation result with metadata and TikTok terminology usage tracking"""
        try:
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                self.logger.warning(f"Expected dict result, got {type(result)}")
                result = {}
            
            # Extract feature information from state
            feature_name = state.get("feature_name", "")
            
            # Enhanced result with additional metadata
            enhanced_result = {
                "agent": "ValidationAgent",
                "feature_name": feature_name,
                "final_decision": result.get("final_decision", "NEEDS_REVIEW"),
                "confidence_score": float(result.get("confidence_score", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "compliance_requirements": result.get("compliance_requirements", []),
                "risk_assessment": result.get("risk_assessment", "UNKNOWN"),
                "recommendations": result.get("recommendations", []),
                "tiktok_terminology_used": self._check_tiktok_terminology_usage(result),
                "memory_overlay_length": len(self.memory_overlay) if self.memory_overlay else 0,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # Log the enhancement process
            self.logger.info(f"Enhanced validation result with {len(enhanced_result)} fields")
            self.logger.info(f"TikTok terminology usage detected: {enhanced_result['tiktok_terminology_used']}")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error enhancing validation result: {e}")
            # Return basic result if enhancement fails
            return {
                "agent": "ValidationAgent",
                "error": f"Result enhancement failed: {str(e)}",
                "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
            }
    
    def _check_tiktok_terminology_usage(self, result: Dict) -> bool:
        """Check if the validation result used TikTok terminology in its analysis"""
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
                self.logger.info(f"✓ TikTok terminology '{pattern.upper()}' detected in validation result")
                return True
        
        # Check reasoning field specifically
        reasoning = result.get("reasoning", "").lower()
        for pattern in tiktok_patterns:
            if pattern in reasoning:
                self.logger.info(f"✓ TikTok terminology '{pattern.upper()}' detected in reasoning")
                return True
        
        # Check compliance requirements field
        requirements = result.get("compliance_requirements", [])
        for req in requirements:
            if isinstance(req, str):
                req_lower = req.lower()
                for pattern in tiktok_patterns:
                    if pattern in req_lower:
                        self.logger.info(f"✓ TikTok terminology '{pattern.upper()}' detected in compliance requirements")
                        return True
        
        # Check recommendations field
        recommendations = result.get("recommendations", [])
        for rec in recommendations:
            if isinstance(rec, str):
                rec_lower = rec.lower()
                for pattern in tiktok_patterns:
                    if pattern in rec_lower:
                        self.logger.info(f"✓ TikTok terminology '{pattern.upper()}' detected in recommendations")
                        return True
        
        self.logger.info("No TikTok terminology detected in validation result")
        return False

