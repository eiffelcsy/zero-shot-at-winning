from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from .base import BaseComplianceAgent
from prompts.templates import TIKTOK_CONTEXT, SCREENING_PROMPT
from typing import List, Dict, Any
from datetime import datetime
import json

class ScreeningOutput(BaseModel):
    risk_level: str = Field(description="Risk level: LOW, MEDIUM, HIGH")
    compliance_required: bool = Field(description="Whether geo-compliance is required")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    trigger_keywords: List[str] = Field(description="Detected compliance keywords")
    regulatory_indicators: List[str] = Field(description="Regulatory signals found")
    reasoning: str = Field(description="Analysis reasoning")
    needs_research: bool = Field(description="Whether research agent is needed")
    geographic_scope: List[str] = Field(description="Geographic regions affected")
    age_sensitivity: bool = Field(description="Whether feature affects minors")
    data_sensitivity: str = Field(description="Data sensitivity level")

class ScreeningAgent(BaseComplianceAgent):
    """First agent - analyzes features for compliance indicators"""
    
    def __init__(self):
        super().__init__("ScreeningAgent")
        self.setup_prompts()
    
    def setup_prompts(self):
        # Enhanced prompt with your TikTok terminology
        prompt_template = TIKTOK_CONTEXT + """

FEATURE TO ANALYZE: {feature_description}

ANALYSIS FRAMEWORK:
Analyze this feature for potential regulatory requirements across multiple jurisdictions.

KEY COMPLIANCE PATTERNS TO EVALUATE:
1. **Data Protection & Privacy**: Personal data collection, processing, retention
2. **Age Restrictions & Child Safety**: Minor protection, parental controls, age verification  
3. **Content Governance**: Moderation obligations, transparency requirements
4. **Geographic Enforcement**: Location-based restrictions, jurisdiction-specific behaviors
5. **Platform Responsibilities**: Regulatory reporting, user safety obligations

DETECTION CRITERIA:
- Age-sensitive functionality (ASL) or data processing involving minors
- Geographic targeting (GH) or location-aware enforcement mechanisms  
- Personal data handling (T5) including collection, processing, or retention
- Content control mechanisms including filtering, blocking, or moderation

Return ONLY valid JSON matching this schema:
{{
    "risk_level": "LOW|MEDIUM|HIGH",
    "compliance_required": true/false,
    "confidence": 0.0-1.0,
    "trigger_keywords": ["keyword1", "keyword2"],
    "regulatory_indicators": ["ASL", "GH", "T5"],
    "reasoning": "detailed explanation",
    "needs_research": true/false,
    "geographic_scope": ["region1", "region2"] or "global" or "unknown",
    "age_sensitivity": true/false,
    "data_sensitivity": "T5|T4|T3|T2|T1|none"
}}
"""
        
        prompt = PromptTemplate(
            input_variables=["feature_description"],
            template=prompt_template
        )
        self.create_chain(prompt, ScreeningOutput)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature for compliance risk"""
        try:
            feature_description = state.get("feature_description", "")
            
            if not feature_description:
                raise ValueError("Missing feature description")
            
            # Run LLM analysis
            result = await self.safe_llm_call({
                "feature_description": feature_description
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
                    "risk_level": "ERROR",
                    "compliance_required": None,
                    "confidence": 0.0,
                    "reasoning": f"Screening failed: {str(e)}",
                    "error": str(e)
                },
                "screening_completed": True,
                "next_step": "validation"  # Skip to validation on error
            }
    
    def _enhance_result(self, result: Dict, original_state: Dict) -> Dict:
        """Add metadata and validation to screening result"""
        
        # Add session metadata
        result["session_metadata"] = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "feature_name": original_state.get("feature_name")
        }
        
        # Validate confidence score
        if not isinstance(result.get("confidence"), (int, float)) or result["confidence"] < 0 or result["confidence"] > 1:
            result["confidence"] = 0.5
        
        # Ensure lists are properly formatted
        if not isinstance(result.get("geographic_scope"), list):
            result["geographic_scope"] = ["unknown"]
        
        if not isinstance(result.get("trigger_keywords"), list):
            result["trigger_keywords"] = []
        
        # Set research decision
        result["needs_research"] = (
            result.get("compliance_required", False) or 
            result.get("confidence", 0) < 0.8 or
            result.get("risk_level") in ["HIGH", "MEDIUM"]
        )
        
        return result