from typing import Dict, Any, Optional
from typing_extensions import TypedDict
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define structured input/output for LangGraph
class ScreeningInput(BaseModel):
    feature_name: str = Field(description="Name of the feature to analyze")
    feature_description: str = Field(description="Detailed feature description")

class ScreeningOutput(BaseModel):
    risk_level: str = Field(description="LOW, MEDIUM, or HIGH")
    compliance_required: bool = Field(description="Whether compliance is needed")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    geographic_scope: list = Field(description="Geographic regions affected")
    age_sensitivity: bool = Field(description="Whether feature affects minors")
    data_sensitivity: str = Field(description="Data sensitivity level")
    reasoning: str = Field(description="Analysis reasoning")
    trigger_keywords: list = Field(description="Detected compliance keywords")
    needs_research: bool = Field(description="Whether research agent is needed")

class LangGraphScreeningAgent:
    def __init__(self):
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1000
        )
        
        # Create structured output parser
        self.output_parser = JsonOutputParser(pydantic_object=ScreeningOutput)
        
        # Enhanced prompt template with TikTok context
        self.prompt_template = PromptTemplate(
            input_variables=["feature_name", "feature_description"],
            template=self._build_prompt_template(),
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        # Create the LangChain chain
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _build_prompt_template(self) -> str:
        return """
TIKTOK INTERNAL TERMINOLOGY (Critical for Analysis):
- ASL: Age-sensitive logic (age verification/restrictions for minors)
- GH: Geo-handler (region-based routing and enforcement)
- CDS: Compliance Detection System (automated compliance monitoring)
- T5: Tier 5 data (highest sensitivity level - more critical than T1-T4)
- Jellybean: Internal parental control system
- Snowcap: Child safety policy framework
- Spanner: Rule engine (not Google Spanner database)
- EchoTrace: Log tracing mode for compliance verification
- ShadowMode: Deploy feature without user impact for analytics collection
- Redline: Flag for legal review (not financial loss context)
- Softblock: Silent user limitation without notifications
- Glow: Compliance-flagging status for geo-based alerts
- NSP: Non-shareable policy (content sharing restrictions)
- DRT: Data retention threshold (how long data can be stored)
- LCP: Local compliance policy (region-specific rules)
- IMT: Internal monitoring trigger
- BB: Baseline Behavior (standard user behavior for anomaly detection)
- PF: Personalized feed (recommendation algorithm)
- FR: Feature rollout status
- NR: Not recommended (restriction/limitation level)

FEATURE TO ANALYZE:
Name: {feature_name}
Description: {feature_description}

ANALYSIS FRAMEWORK:
Analyze this feature for potential regulatory requirements across multiple jurisdictions and compliance domains.

KEY COMPLIANCE PATTERNS TO EVALUATE:
1. **Data Protection & Privacy**: Personal data collection, processing, retention, cross-border transfers, user consent
2. **Age Restrictions & Child Safety**: Minor protection, parental controls, age verification, content filtering for children
3. **Content Governance**: Moderation obligations, transparency requirements, algorithmic accountability, user reporting
4. **Geographic Enforcement**: Location-based restrictions, jurisdiction-specific behaviors, regional compliance variations
5. **Platform Responsibilities**: Regulatory reporting, user safety obligations, accessibility requirements

DETECTION CRITERIA:
- Age-sensitive functionality (ASL) or data processing involving minors
- Geographic targeting (GH) or location-aware enforcement mechanisms
- Personal data handling (T5) including collection, processing, or retention
- Content control mechanisms including filtering, blocking, or moderation
- Compliance indicators such as legal terminology, regulatory references, or policy enforcement

RISK ASSESSMENT GUIDELINES:
- **HIGH RISK**: Clear regulatory compliance required (combines multiple patterns like age+location+data)
- **MEDIUM RISK**: Potential compliance needs (one or two patterns present)
- **LOW RISK**: Business functionality with minimal regulatory implications

{format_instructions}

Provide analysis in valid JSON format only.
"""
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph-compatible process method
        
        Args:
            state: LangGraph state containing feature information
            
        Returns:
            Updated state with screening results
        """
        try:
            # Extract input from LangGraph state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            
            if not feature_name or not feature_description:
                raise ValueError("Missing required feature name or description")
            
            # Run LangChain chain
            result = await self.chain.ainvoke({
                "feature_name": feature_name,
                "feature_description": feature_description
            })
            
            # Validate and enhance result
            enhanced_result = self._enhance_screening_result(result, state)
            
            # Return LangGraph-compatible state update
            return {
                "screening_result": enhanced_result,
                "screening_completed": True,
                "screening_timestamp": datetime.now().isoformat(),
                "next_step": "research" if enhanced_result.get("needs_research", True) else "validation"
            }
            
        except Exception as e:
            # Error handling for LangGraph
            error_result = {
                "risk_level": "ERROR",
                "compliance_required": None,
                "confidence": 0.0,
                "reasoning": f"Screening failed: {str(e)}",
                "error": str(e),
                "needs_research": False
            }
            
            return {
                "screening_result": error_result,
                "screening_completed": True,
                "screening_error": str(e),
                "next_step": "validation"  # Skip to validation on error
            }
    
    def _enhance_screening_result(self, result: Dict[str, Any], original_state: Dict) -> Dict[str, Any]:
        """Enhance screening result with additional metadata"""
        
        # Add session tracking
        result["session_metadata"] = {
            "agent": "LangGraphScreeningAgent",
            "langchain_model": self.llm.model_name,
            "processing_timestamp": datetime.now().isoformat(),
            "feature_name": original_state.get("feature_name")
        }
        
        # Validate confidence score
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            result["confidence"] = 0.5  # Default to medium confidence
        
        # Ensure geographic scope is a list
        if not isinstance(result.get("geographic_scope"), list):
            result["geographic_scope"] = ["unknown"]
        
        # Add research decision logic
        result["needs_research"] = (
            result.get("compliance_required", False) or 
            result.get("confidence", 0) < 0.8 or
            result.get("risk_level") in ["HIGH", "MEDIUM"]
        )
        
        return result