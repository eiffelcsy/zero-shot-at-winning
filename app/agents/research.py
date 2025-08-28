from pydantic import BaseModel, Field
from .base import BaseComplianceAgent
from .prompts.templates import RESEARCH_PROMPT
from typing import Dict, Any, List
from datetime import datetime
import json
import os
import glob
import re
from collections import defaultdict
import math


class ResearchOutput(BaseModel):
    agent: str = Field(description="Agent name", default="ResearchAgent")
    candidates: List[Dict[str, Any]] = Field(description="Candidate regulations identified")
    evidence: List[Dict[str, Any]] = Field(description="Evidence snippets with sources")
    query_used: str = Field(description="Search query constructed")
    confidence_score: float = Field(description="Overall confidence in research findings")


class KBRecord:
    """Knowledge base record for regulation information"""
    def __init__(self, jurisdiction: str, reg_code: str, name: str, section: str, url: str, excerpt: str):
        self.jurisdiction = jurisdiction
        self.reg_code = reg_code
        self.name = name
        self.section = section
        self.url = url
        self.excerpt = excerpt
        self.tokens = self._tokenize(f"{name} {section} {excerpt}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for search"""
        STOP_WORDS = {"the", "a", "an", "of", "and", "for", "to", "in", "on", "with", "by", "or", "is", "are", "be"}
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t not in STOP_WORDS]


class ResearchAgent(BaseComplianceAgent):
    """Research Agent - finds relevant regulations using local knowledge base"""
    
    def __init__(self, kb_dir: str = "data/kb"):
        super().__init__("ResearchAgent")
        self.kb_dir = kb_dir
        
        # Load knowledge base
        self.kb_records = self._load_knowledge_base()
        self.idf_scores = self._build_idf_scores()
        
        # Setup LangChain components
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain prompt and parser"""
        self.create_chain(RESEARCH_PROMPT, ResearchOutput)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible process method"""
        try:
            # Extract screening analysis from state
            screening_analysis = state.get("screening_analysis", {})
            
            if not screening_analysis:
                raise ValueError("Missing screening analysis from previous agent")
            
            # Step 1: Build search query based on screening results
            search_query = self._build_search_query_from_screening(screening_analysis)
            
            # Step 2: Find candidate regulations based on screening flags
            candidates = self._identify_candidate_regulations(screening_analysis)
            
            # Step 3: Search knowledge base for evidence
            evidence = self._search_evidence(search_query, candidates)
            
            # Step 4: Use LLM to synthesize findings
            llm_input = {
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(evidence[:5], indent=2)  # Top 5 for LLM context
            }
            
            result = await self.safe_llm_call(llm_input)
            
            # Step 5: Enhance result with full evidence and candidates
            result["evidence"] = evidence[:10]  # Return top 10 evidence pieces
            result["candidates"] = [{"reg": c[0], "why": c[1], "score": c[2]} for c in candidates]
            result["query_used"] = search_query
            
            # Ensure agent field is set
            result["agent"] = "ResearchAgent"
            
            # Validate confidence score
            if not isinstance(result.get("confidence_score"), (int, float)):
                result["confidence_score"] = 0.7
            
            self.log_interaction(state, result)
            
            # Return LangGraph state update
            return {
                "research_evidence": result["evidence"],
                "research_candidates": result["candidates"],
                "research_query": result["query_used"],
                "research_confidence": result.get("confidence_score", 0.7),
                "research_analysis": result,  # Full research output
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"
            }
            
        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
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
                    "error": str(e)
                },
                "research_error": str(e),
                "research_completed": True,
                "next_step": "validation"
            }
    
    def _build_search_query_from_screening(self, screening_analysis: Dict) -> str:
        """Build targeted search query based on screening analysis flags"""
        query_parts = []
        
        # Add trigger keywords from screening
        trigger_keywords = screening_analysis.get("trigger_keywords", [])
        if trigger_keywords:
            query_parts.extend(trigger_keywords[:3])  # Top 3 keywords
        
        # Add geographic scope
        geo_scope = screening_analysis.get("geographic_scope", [])
        if isinstance(geo_scope, list) and geo_scope != ["unknown"]:
            query_parts.extend(geo_scope)
        
        # Add compliance pattern terms based on screening flags
        if screening_analysis.get("age_sensitivity"):
            query_parts.extend(["minors", "under 18", "child protection", "parental controls"])
        
        if screening_analysis.get("data_sensitivity") in ["T5", "T4"]:
            query_parts.extend(["personal data", "privacy", "data protection"])
        
        if screening_analysis.get("compliance_required"):
            query_parts.extend(["regulatory compliance", "legal requirements"])
        
        # Add risk level specific terms
        risk_level = screening_analysis.get("risk_level", "")
        if risk_level == "HIGH":
            query_parts.extend(["mandatory", "required", "must comply"])
        elif risk_level == "MEDIUM":
            query_parts.extend(["recommended", "should comply"])
        
        return " ".join(query_parts)
    
    def _identify_candidate_regulations(self, screening_analysis: Dict) -> List[tuple]:
        """Identify candidate regulations based on screening flags"""
        candidates = []
        geo_scope = [g.upper() for g in screening_analysis.get("geographic_scope", [])]
        age_sensitive = screening_analysis.get("age_sensitivity", False)
        data_sensitive = screening_analysis.get("data_sensitivity", "none")
        compliance_required = screening_analysis.get("compliance_required", False)
        
        # California regulations
        if any(region in geo_scope for region in ["CALIFORNIA", "CA", "US", "UNITED STATES"]):
            if age_sensitive:
                candidates.append(("CA_SB976", "California minors protection law", 0.9))
                candidates.append(("CA_CCPA", "California Consumer Privacy Act", 0.7))
        
        # Utah regulations  
        if any(region in geo_scope for region in ["UTAH", "UT", "US", "UNITED STATES"]):
            if age_sensitive:
                candidates.append(("UT_MINORS", "Utah social media regulations for minors", 0.9))
        
        # US federal regulations
        if any(region in geo_scope for region in ["US", "UNITED STATES", "FEDERAL"]):
            candidates.append(("US_2258A", "US federal reporting requirements", 0.8))
            if age_sensitive:
                candidates.append(("COPPA", "Children's Online Privacy Protection Act", 0.9))
        
        # EU regulations
        if any(region in geo_scope for region in ["EU", "EUROPE", "EUROPEAN UNION"]):
            if data_sensitive in ["T5", "T4"]:
                candidates.append(("GDPR", "General Data Protection Regulation", 0.9))
            if compliance_required:
                candidates.append(("EU_DSA", "EU Digital Services Act", 0.8))
        
        # UK regulations
        if any(region in geo_scope for region in ["UK", "UNITED KINGDOM", "BRITAIN"]):
            if age_sensitive:
                candidates.append(("UK_OSA", "UK Online Safety Act", 0.8))
            if data_sensitive in ["T5", "T4"]:
                candidates.append(("UK_DPA", "UK Data Protection Act", 0.7))
        
        # Global/Multi-jurisdictional
        if "GLOBAL" in geo_scope or not geo_scope or geo_scope == ["unknown"]:
            if age_sensitive:
                candidates.append(("COPPA", "US Children's Online Privacy Protection Act", 0.6))
                candidates.append(("CA_SB976", "California Age-Appropriate Design Code", 0.5))
            if data_sensitive in ["T5", "T4"]:
                candidates.append(("GDPR", "EU General Data Protection Regulation", 0.6))
        
        # Fallback if no specific matches
        if not candidates:
            candidates.append(("GENERAL_COMPLIANCE", "General compliance requirements", 0.4))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates[:5]  # Return top 5 candidates
    
    def _search_evidence(self, query: str, candidates: List[tuple]) -> List[Dict]:
        """Search for evidence in knowledge base"""
        query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        evidence = []
        
        for record in self.kb_records:
            # Calculate relevance score
            doc_terms = set(record.tokens)
            overlap = query_terms.intersection(doc_terms)
            
            if overlap:
                # Base score from TF-IDF style calculation
                score = sum(self.idf_scores.get(term, 1.0) for term in overlap)
                
                # Boost score for candidate regulations
                for candidate_reg, _, candidate_score in candidates:
                    if record.reg_code == candidate_reg:
                        score *= (1 + candidate_score)  # Boost by candidate score
                
                # Boost score for exact jurisdiction matches
                if any(record.jurisdiction.upper() in query.upper() for _ in [1]):
                    score *= 1.2
                
                evidence.append({
                    "reg": record.reg_code,
                    "jurisdiction": record.jurisdiction,
                    "name": record.name,
                    "section": record.section,
                    "url": record.url,
                    "excerpt": record.excerpt,
                    "score": round(score, 2)
                })
        
        # Sort by score and return top results
        evidence.sort(key=lambda x: x["score"], reverse=True)
        return evidence
    
    def _load_knowledge_base(self) -> List[KBRecord]:
        """Load regulation knowledge base from JSONL files"""
        records = []
        
        # Create directory if it doesn't exist
        os.makedirs(self.kb_dir, exist_ok=True)
        
        # Load from JSONL files
        for filepath in glob.glob(os.path.join(self.kb_dir, "*.jsonl")):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            record = KBRecord(
                                jurisdiction=data.get("jurisdiction", ""),
                                reg_code=data.get("reg_code", ""),
                                name=data.get("name", ""),
                                section=data.get("section", ""),
                                url=data.get("url", ""),
                                excerpt=data.get("excerpt", "")
                            )
                            records.append(record)
            except Exception as e:
                self.logger.warning(f"Failed to load {filepath}: {e}")
        
        # If no records found, create sample data
        if not records:
            self._create_sample_kb()
            records = self._load_knowledge_base()  # Reload
        
        return records
    
    def _create_sample_kb(self):
        """Create sample knowledge base for development"""
        sample_data = [
            {
                "jurisdiction": "CA",
                "reg_code": "CA_SB976",
                "name": "California SB-976",
                "section": "Section 1 - Default Settings",
                "url": "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB976",
                "excerpt": "Social media platforms must disable personalized feeds by default for users under 18 in California"
            },
            {
                "jurisdiction": "UT",
                "reg_code": "UT_MINORS",
                "name": "Utah Social Media Regulation Act",
                "section": "Curfew Restrictions",
                "url": "https://en.wikipedia.org/wiki/Utah_Social_Media_Regulation_Act",
                "excerpt": "Requires social media platforms to implement curfew restrictions for minor users"
            },
            {
                "jurisdiction": "US",
                "reg_code": "US_2258A",
                "name": "18 USC §2258A",
                "section": "Reporting Requirements",
                "url": "https://www.law.cornell.edu/uscode/text/18/2258A",
                "excerpt": "Electronic service providers must report child sexual abuse material to NCMEC"
            },
            {
                "jurisdiction": "US",
                "reg_code": "COPPA",
                "name": "Children's Online Privacy Protection Act",
                "section": "Age Verification Requirements",
                "url": "https://www.ftc.gov/enforcement/rules/rulemaking-regulatory-reform-proceedings/childrens-online-privacy-protection-rule",
                "excerpt": "Requires parental consent for collection of personal information from children under 13"
            },
            {
                "jurisdiction": "EU",
                "reg_code": "GDPR",
                "name": "General Data Protection Regulation",
                "section": "Article 8 - Conditions for child's consent",
                "url": "https://gdpr-info.eu/art-8-gdpr/",
                "excerpt": "Processing of personal data of children requires parental consent for children under 16"
            },
            {
                "jurisdiction": "EU",
                "reg_code": "EU_DSA",
                "name": "Digital Services Act",
                "section": "Article 28 - Online protection of minors",
                "url": "https://digital-strategy.ec.europa.eu/en/policies/digital-services-act-package",
                "excerpt": "Very large online platforms must put in place appropriate measures to ensure a high level of privacy, safety and security for minors"
            }
        ]
        
        # Write sample data
        sample_file = os.path.join(self.kb_dir, "sample_regulations.jsonl")
        with open(sample_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
    
    def _build_idf_scores(self) -> Dict[str, float]:
        """Build IDF scores for terms in knowledge base"""
        if not self.kb_records:
            return {}
        
        term_doc_count = defaultdict(int)
        total_docs = len(self.kb_records)
        
        for record in self.kb_records:
            unique_terms = set(record.tokens)
            for term in unique_terms:
                term_doc_count[term] += 1
        
        # Calculate IDF scores
        idf_scores = {}
        for term, doc_count in term_doc_count.items():
            idf_scores[term] = math.log((total_docs + 1) / (doc_count + 1)) + 1.0
        
        return idf_scores