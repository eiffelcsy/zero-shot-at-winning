from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, HttpUrl
from .base import BaseComplianceAgent
from .templates import TIKTOK_CONTEXT
from typing import Dict, Any, List, Literal
from datetime import datetime
import json

class RelatedRegulation(BaseModel):
    name: str = Field(description="Regulation name")
    jurisdiction: str = Field(description="Jurisdiction")
    section: str = Field(description="Specific section")
    url: HttpUrl = Field(description="URL to regulation")
    evidence_excerpt: str = Field(description="Supporting evidence text")

class ValidationOutput(BaseModel):
    needs_geo_logic: Literal["YES", "NO", "REVIEW"] = Field(description="Final compliance decision")
    reasoning: str = Field(description="Detailed reasoning for decision", min_length=10, max_length=1200)
    related_regulations: List[RelatedRegulation] = Field(description="Regulations that support the decision")
    confidence_score: float = Field(description="Confidence in the decision 0.0-1.0")

class ValidationAgent(BaseComplianceAgent):
    """Final decision-maker agent - validates compliance requirements"""
    
    def __init__(self):
        super().__init__("ValidationAgent", temperature=0.0)  # Low temperature for consistency
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain prompt and parser"""
        prompt_template = TIKTOK_CONTEXT + """

FEATURE ANALYSIS TO VALIDATE:
Feature Name: {feature_name}
Feature Description: {feature_description}

SCREENING AGENT ANALYSIS:
{screening_analysis}

RESEARCH AGENT EVIDENCE:
{research_evidence}

VALIDATION TASK:
You are the final decision-maker. Based on the screening analysis and research evidence, determine:
1. Does this feature need geo-specific compliance logic? (YES/NO/REVIEW)
2. Provide clear reasoning citing the evidence
3. List related regulations with proper citations

DECISION CRITERIA:
- YES: Feature enforces different behavior by region due to legal requirements
- NO: Feature is business-driven or has no legal geo-requirements  
- REVIEW: Insufficient evidence or ambiguous requirements

EVIDENCE REQUIREMENTS:
- Only cite regulations that appear in the research evidence
- Include specific excerpts that support your decision
- If evidence is thin or contradictory, choose REVIEW

Return ONLY valid JSON:
{{
    "needs_geo_logic": "YES|NO|REVIEW",
    "reasoning": "detailed explanation citing specific evidence",
    "related_regulations": [
        {{
            "name": "regulation name",
            "jurisdiction": "jurisdiction",
            "section": "specific section", 
            "url": "regulation URL",
            "evidence_excerpt": "supporting text from research"
        }}
    ],
    "confidence_score": 0.85
}}
"""
        
        self.prompt_template = PromptTemplate(
            input_variables=["feature_name", "feature_description", "screening_analysis", "research_evidence"],
            template=prompt_template
        )
        
        self.output_parser = JsonOutputParser(pydantic_object=ValidationOutput)
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible process method"""
        try:
            # Extract data from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            screening_analysis = state.get("screening_analysis", {})
            research_evidence = state.get("research_evidence", [])
            
            # Prepare input for validation
            validation_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "research_evidence": json.dumps(research_evidence[:8], indent=2)  # Top 8 for context
            }
            
            # Get LLM decision
            result = await self.chain.ainvoke(validation_input)
            
            # Validate and enhance result
            enhanced_result = self._validate_result(result, research_evidence)
            
            self.log_interaction(validation_input, enhanced_result)
            
            # Return final decision for LangGraph
            return {
                "final_decision": {
                    "needs_geo_logic": enhanced_result["needs_geo_logic"],
                    "reasoning": enhanced_result["reasoning"],
                    "related_regulations": enhanced_result["related_regulations"],
                    "confidence": enhanced_result["confidence_score"],
                    "agent": self.name,
                    "validation_method": "evidence_based_llm_decision"
                },
                "validation_completed": True,
                "validation_timestamp": datetime.now().isoformat(),
                "workflow_completed": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Validation agent failed: {e}")
            return {
                "final_decision": {
                    "needs_geo_logic": "REVIEW",
                    "reasoning": f"Validation failed due to processing error: {str(e)}. Human review required.",
                    "related_regulations": [],
                    "confidence": 0.0,
                    "error": str(e)
                },
                "validation_completed": True,
                "validation_error": str(e)
            }
    
    def _validate_result(self, result: Dict, research_evidence: List[Dict]) -> Dict:
        """Validate and enhance the LLM result"""
        
        # Ensure decision is valid
        if result.get("needs_geo_logic") not in ["YES", "NO", "REVIEW"]:
            result["needs_geo_logic"] = "REVIEW"
            result["reasoning"] = "Invalid decision format detected. " + result.get("reasoning", "")
        
        # Validate confidence score
        confidence = result.get("confidence_score", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            result["confidence_score"] = 0.5
        
        # Validate related regulations against evidence
        evidence_urls = {e.get("url") for e in research_evidence if e.get("url")}
        valid_regulations = []
        
        for reg in result.get("related_regulations", []):
            # Check if regulation URL exists in evidence
            if isinstance(reg, dict) and reg.get("url") in evidence_urls:
                valid_regulations.append(reg)
        
        result["related_regulations"] = valid_regulations
        
        # Downgrade decision if no supporting regulations for YES
        if result["needs_geo_logic"] == "YES" and not result["related_regulations"]:
            result["needs_geo_logic"] = "REVIEW"
            result["reasoning"] += " | Decision downgraded to REVIEW: no supporting regulations found in evidence."
        
        # Add validation metadata
        result["validation_metadata"] = {
            "agent": self.name,
            "evidence_pieces_reviewed": len(research_evidence),
            "regulations_cited": len(result["related_regulations"]),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    # Synchronous method for backward compatibility (if needed)
    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous decision method for compatibility"""
        import asyncio
        
        # Convert payload to state format
        state = {
            "feature_name": payload.get("feature_name", ""),
            "feature_description": payload.get("feature_description", ""),
            "screening_analysis": payload.get("screening", {}),
            "research_evidence": payload.get("research", {}).get("evidence", [])
        }
        
        # Run async process
        result = asyncio.run(self.process(state))
        
        # Return in expected format
        final_decision = result.get("final_decision", {})
        return {
            "agent": self.name,
            "decision": final_decision.get("needs_geo_logic", "REVIEW"),
            "reasoning": final_decision.get("reasoning", ""),
            "related_regulations": final_decision.get("related_regulations", []),
            "raw": final_decision
        }