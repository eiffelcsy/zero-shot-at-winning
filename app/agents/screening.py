from typing import Dict, Any, List
import re
from .base import BaseComplianceAgent
from .prompts.templates import SCREENING_PROMPT, TIKTOK_CONTEXT

class ScreeningAgent(BaseComplianceAgent):
    """Initial screening agent for compliance analysis"""
    
    def __init__(self, llm):
        super().__init__("ScreeningAgent", llm)
        self.screening_prompt = SCREENING_PROMPT
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Screen feature for compliance requirements"""
        try:
            feature_description = input_data.get("feature_description", "")
            feature_name = input_data.get("feature_name", "")
            
            # Generate LLM analysis using the safe_llm_call method
            llm_response = await self.safe_llm_call(
                self.screening_prompt.format(feature_description=feature_description)
            )
            
            # Parse the LLM response
            analysis = self._parse_llm_output(llm_response)
            
            # Validate and enhance analysis
            if "error" not in analysis:
                analysis = self._enhance_screening_analysis(analysis, feature_description)
                
                # Calculate confidence using inherited method
                confidence = self.calculate_confidence(
                    reasoning=analysis.get("reasoning", ""),
                    evidence=analysis.get("trigger_keywords", []),
                    context={
                        "has_legal_keywords": self._has_legal_keywords(feature_description),
                        "geographic_specificity": self._has_geographic_indicators(feature_description)
                    }
                )
                analysis["confidence"] = confidence
            
            # Log the interaction for audit trail
            self._log_interaction(input_data, analysis)
            
            return {
                "agent": self.name,
                "analysis": analysis,
                "next_step": self._determine_next_step(analysis),
                "feature_name": feature_name
            }
            
        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return self._get_error_response(str(e), input_data.get("feature_name", ""))
    
    def _enhance_screening_analysis(self, analysis: Dict, feature_desc: str) -> Dict:
        """Enhance LLM analysis with rule-based checks"""
        
        # Extract geographic scope
        analysis["geographic_scope"] = self._extract_regions(feature_desc)
        
        # Enhance trigger keywords with rule-based detection
        detected_keywords = self._detect_compliance_keywords(feature_desc)
        existing_keywords = analysis.get("trigger_keywords", [])
        analysis["trigger_keywords"] = list(set(existing_keywords + detected_keywords))
        
        # Age sensitivity detection
        analysis["age_sensitivity"] = self._detect_age_sensitivity(feature_desc)
        
        # Data sensitivity classification
        analysis["data_sensitivity"] = self._classify_data_sensitivity(feature_desc)
        
        return analysis
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract geographic regions from feature description"""
        regions = []
        region_patterns = {
            "EU": r"(?i)\bEU\b|European?\s+Union|GDPR|DSA",
            "California": r"(?i)California|CA\b|CCPA|SB-?976",
            "Utah": r"(?i)Utah|UT\b|Utah.*Act",
            "Florida": r"(?i)Florida|FL\b|Florida.*law",
            "US": r"(?i)United\s+States|US\b|NCMEC|federal"
        }
        
        for region, pattern in region_patterns.items():
            if re.search(pattern, text):
                regions.append(region)
        
        return regions if regions else ["unknown"]
    
    def _detect_compliance_keywords(self, text: str) -> List[str]:
        """Detect compliance-related keywords using pattern matching"""
        compliance_patterns = {
            "age_restrictions": ["ASL", "minor", "age", "under 18", "parental", "Jellybean", "Snowcap"],
            "geo_enforcement": ["GH", "region", "location", "geographic", "geo"],
            "data_sensitivity": ["T5", "personal data", "retention", "DRT", "privacy"],
            "legal_compliance": ["law", "regulation", "compliance", "legal", "act", "requirement"],
            "content_control": ["block", "filter", "moderate", "NSP", "Softblock", "Redline"]
        }
        
        found_keywords = []
        text_lower = text.lower()
        
        for category, keywords in compliance_patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _detect_age_sensitivity(self, text: str) -> bool:
        """Detect if feature involves age-sensitive functionality"""
        age_indicators = ["ASL", "minor", "age", "under 18", "parental", "Jellybean", "Snowcap", "child"]
        return any(indicator.lower() in text.lower() for indicator in age_indicators)
    
    def _classify_data_sensitivity(self, text: str) -> str:
        """Classify data sensitivity level"""
        if "T5" in text:
            return "T5"
        elif any(term in text.lower() for term in ["personal data", "location", "biometric"]):
            return "T4"
        elif any(term in text.lower() for term in ["user data", "analytics", "logs"]):
            return "T3"
        else:
            return "T1"
    
    def _has_legal_keywords(self, text: str) -> bool:
        """Check if text contains legal/regulatory keywords"""
        legal_keywords = ["law", "regulation", "compliance", "legal", "act", "requirement", "mandate"]
        return any(keyword.lower() in text.lower() for keyword in legal_keywords)
    
    def _has_geographic_indicators(self, text: str) -> bool:
        """Check if text has geographic specificity"""
        geo_indicators = ["GH", "region", "location", "country", "state", "EU", "California", "Utah", "Florida"]
        return any(indicator in text for indicator in geo_indicators)
    
    def _determine_next_step(self, analysis: Dict) -> str:
        """Determine next step in the workflow based on analysis"""
        if analysis.get("error"):
            return "human_review"
        
        # Normalize values for consistent comparison    
        risk_level = str(analysis.get("risk_level", "")).upper()
        compliance_required = bool(analysis.get("compliance_required", False))
        confidence = float(analysis.get("confidence", 0.0))
        
        # Decision logic
        if not risk_level or risk_level == "UNKNOWN":
            return "human_review"
        elif compliance_required and confidence > 0.6:
            return "research"
        elif risk_level == "HIGH":
            return "research"
        elif confidence < 0.5:
            return "human_review"
        else:
            return "validation"
    
    def _get_error_response(self, error_msg: str, feature_name: str) -> Dict:
        """Generate standardized error response"""
        return {
            "agent": self.name,
            "analysis": {
                "error": error_msg,
                "risk_level": "UNKNOWN",
                "compliance_required": None,
                "confidence": 0.0,
                "needs_human_review": True
            },
            "next_step": "human_review",
            "feature_name": feature_name
        }