from pydantic import BaseModel
from typing import List, Dict, Any
import json
import re
from datetime import datetime

import chromadb

from .prompts.templates import RESEARCH_PROMPT
from .base import BaseComplianceAgent
from .tools.jurisdiction_check import JurisdictionChecker
from .tools.regulation_search import RegulationSearcher
from .tools.risk_calculator import RiskCalculator


class ResearchOutput(BaseModel):
    agent: str
    candidates: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    query_used: str
    confidence_score: float


class ResearchAgent(BaseComplianceAgent):
    """Research Agent - finds relevant regulations using ChromaDB knowledge base only"""

    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8001):
        super().__init__("ResearchAgent")
        
        # Initialize ChromaDB client (no fallback to JSONL)
        try:
            self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        except Exception:
            # Fallback to persistent client
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Initialize ChromaDB tools
        self.jurisdiction_checker = JurisdictionChecker(self.chroma_client)
        self.regulation_searcher = RegulationSearcher(self.chroma_client)
        self.risk_calculator = RiskCalculator(self.chroma_client)

        # Setup LangChain components
        self._setup_chain()

    def _setup_chain(self):
        """Setup LangChain prompt and parser"""
        self.create_chain(RESEARCH_PROMPT, ResearchOutput)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB-only research process"""
        try:
            # Extract screening analysis from state
            screening_analysis = state.get("screening_analysis", {})
            
            if not screening_analysis:
                raise ValueError("Missing screening analysis from previous agent")

            # Extract key parameters
            geographic_scope = screening_analysis.get("geographic_scope", [])
            trigger_keywords = screening_analysis.get("trigger_keywords", [])
            age_sensitivity = screening_analysis.get("age_sensitivity", False)
            data_sensitivity = screening_analysis.get("data_sensitivity", "none")

            # Step 1: Check applicable jurisdictions using ChromaDB
            applicable_jurisdictions = self.jurisdiction_checker.check_applicable_jurisdictions(
                geographic_scope, trigger_keywords
            )

            # Step 2: Extract compliance patterns and search regulations
            compliance_patterns = self._extract_compliance_patterns(screening_analysis)
            chroma_regulations = self.regulation_searcher.search_by_compliance_patterns(
                compliance_patterns, geographic_scope
            )

            # Step 3: Calculate enhanced risk assessment
            risk_assessment = self.risk_calculator.calculate_compliance_risk(
                feature_description=state.get("feature_description", ""),
                geographic_scope=geographic_scope,
                age_sensitivity=age_sensitivity,
                data_sensitivity=data_sensitivity,
                trigger_keywords=trigger_keywords
            )

            # Step 4: Prepare data for LLM synthesis
            candidates = self._format_candidates(applicable_jurisdictions)
            evidence = chroma_regulations[:15]  # Top 15 evidence pieces

            # Build search query for logging
            search_query = " ".join(compliance_patterns + geographic_scope + trigger_keywords)

            # Step 5: Use LLM for final synthesis
            llm_input = {
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(evidence[:5], indent=2)  # Top 5 for LLM context
            }

            result = await self.safe_llm_call(llm_input)

            # Step 6: Enhance result with ChromaDB insights
            result["evidence"] = evidence
            result["candidates"] = candidates
            result["query_used"] = search_query
            result["agent"] = "ResearchAgent"
            result["risk_assessment"] = risk_assessment
            result["applicable_jurisdictions"] = applicable_jurisdictions

            # Use ChromaDB risk assessment for confidence if available
            if risk_assessment.get("confidence"):
                result["confidence_score"] = risk_assessment["confidence"]
            elif not isinstance(result.get("confidence_score"), (int, float)):
                result["confidence_score"] = 0.7

            self.log_interaction(state, result)

            # Return enhanced state update
            return {
                "research_evidence": result["evidence"],
                "research_candidates": result["candidates"],
                "research_query": result["query_used"],
                "research_confidence": result["confidence_score"],
                "research_analysis": result,
                "research_risk_assessment": risk_assessment,
                "research_jurisdictions": applicable_jurisdictions,
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"
            }

        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
            return self._create_error_response(str(e))

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "research_evidence": [],
            "research_candidates": [],
            "research_query": "",
            "research_analysis": {
                "agent": "ResearchAgent",
                "candidates": [],
                "evidence": [],
                "query_used": "",
                "confidence_score": 0.0,
                "error": error_message
            },
            "research_risk_assessment": {},
            "research_jurisdictions": [],
            "research_error": error_message,
            "research_completed": True,
            "research_timestamp": datetime.now().isoformat(),
            "next_step": "validation"
        }

    def _extract_compliance_patterns(self, screening_analysis: Dict) -> List[str]:
        """Extract compliance patterns from screening analysis"""
        patterns = []
        
        if screening_analysis.get("age_sensitivity"):
            patterns.append("age_restrictions")
        
        if screening_analysis.get("data_sensitivity") in ["T5", "T4"]:
            patterns.append("data_protection")
        
        if screening_analysis.get("compliance_required"):
            patterns.extend(["content_governance", "platform_responsibilities"])
        
        # Always include geographic enforcement if scope is defined
        if screening_analysis.get("geographic_scope", []) != ["unknown"]:
            patterns.append("geographic_enforcement")
        
        return patterns

    def _format_candidates(self, jurisdiction_results: List[Dict]) -> List[Dict]:
        """Format jurisdiction results as regulation candidates"""
        candidates = []
        
        for jur_result in jurisdiction_results:
            reg_code = jur_result.get("regulation_code", "")
            if reg_code:
                candidates.append({
                    "reg": reg_code,
                    "why": f"Applicable to {jur_result.get('jurisdiction', 'jurisdiction')}",
                    "score": jur_result.get("relevance_score", 0.5)
                })
        
        return candidates