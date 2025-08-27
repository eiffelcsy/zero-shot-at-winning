from typing import Dict, Any
from langchain.prompts import PromptTemplate
from .base import BaseComplianceAgent
from .prompts.templates import TIKTOK_CONTEXT
import re

class ScreeningAgent(BaseComplianceAgent):
    """Initial screening agent for compliance analysis"""
    
    def __init__(self, llm):
        super().__init__("ScreeningAgent", llm)
        self.screening_prompt = PromptTemplate(
            input_variables=["feature_description"],
            template=f"""
{TIKTOK_CONTEXT}

FEATURE TO ANALYZE: {{feature_description}}

Perform initial compliance screening. Look for:
1. Age-related restrictions (ASL, minors, under 18)
2. Geographic restrictions (GH, region-specific, location-based)
3. Data handling (T5, personal data, retention)
4. Content control (blocking, filtering, moderation)
5. Legal compliance indicators (law names, regulatory terms)

Return ONLY valid JSON:
{{
    "risk_level": "LOW|MEDIUM|HIGH",
    "compliance_required": true/false,
    "confidence": 0.0-1.0,
    "trigger_keywords": ["keyword1", "keyword2"],
    "regulatory_indicators": ["indicator1", "indicator2"],
    "reasoning": "detailed explanation",
    "needs_research": true/false,
    "geographic_scope": ["region1", "region2"] or "global",
    "age_sensitivity": true/false
}}
"""
        )
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Screen feature for compliance requirements"""
        try:
            feature_description = input_data.get("feature_description", "")
            
            # Generate LLM analysis
            llm_response = await self.llm.apredict(
                self.screening_prompt.format(feature_description=feature_description)
            )
            
            # Parse response
            analysis = self._parse_llm_output(llm_response)
            
            # Validate and enhance analysis
            if "error" not in analysis:
                analysis = self._enhance_screening_analysis(analysis, feature_description)
            
            self._log_interaction(input_data, analysis)
            
            return {
                "agent": self.name,
                "analysis": analysis,
                "next_step": self._determine_next_step(analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return self._get_error_response(str(e))
    
    def _enhance_screening_analysis(self, analysis: Dict, feature_desc: str) -> Dict:
        """Enhance LLM analysis with rule-based checks"""
        
        # Confidence adjustment based on keyword matching
        high_risk_keywords = ["law", "regulation", "compliance", "legal", "restrict", "block"]
        keyword_matches = sum(1 for keyword in high_risk_keywords 
                            if keyword.lower() in feature_desc.lower())
        
        if keyword_matches >= 2:
            analysis["confidence"] = min(1.0, analysis.get("confidence", 0.5) + 0.2)
        
        # Geographic scope enhancement
        geo_keywords = ["GH", "region", "location", "country", "state", "EU", "California", "Utah"]
        if any(keyword in feature_desc for keyword in geo_keywords):
            analysis["geographic_scope"] = self._extract_regions(feature_desc)
        
        return analysis
    
    def _extract_regions(self, text: str) -> list[str]:
        """Extract geographic regions from text"""
        regions = []
        region_patterns = {
            "EU": r"(?i)\bEU\b|European?\s+Union|GDPR",
            "California": r"(?i)California|CA\b|CCPA|SB-?976",
            "Utah": r"(?i)Utah|UT\b",
            "Florida": r"(?i)Florida|FL\b",
            "US": r"(?i)United\s+States|US\b|NCMEC"
        }
        
        for region, pattern in region_patterns.items():
            if re.search(pattern, text):
                regions.append(region)
        
        return regions if regions else ["unknown"]
    
    def _determine_next_step(self, analysis: Dict) -> str:
        """Determine next step in workflow"""
        if analysis.get("error"):
            return "human_review"
        elif analysis.get("compliance_required") and analysis.get("confidence", 0) > 0.6:
            return "research"
        elif analysis.get("risk_level") == "HIGH":
            return "research"
        elif analysis.get("confidence", 0) < 0.5:
            return "human_review"
        else:
            return "validation"
    
    def _get_error_response(self, error_msg: str) -> Dict:
        """Generate error response"""
        return {
            "agent": self.name,
            "analysis": {
                "error": error_msg,
                "risk_level": "UNKNOWN",
                "compliance_required": None,
                "confidence": 0.0,
                "needs_human_review": True
            },
            "next_step": "human_review"
        }