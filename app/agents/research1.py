from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from .base import BaseComplianceAgent
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
import glob
import re
from collections import defaultdict
import math

class ResearchOutput(BaseModel):
    agent: str = Field(description="Agent name")
    candidates: List[Dict[str, Any]] = Field(description="Candidate regulations identified")
    evidence: List[Dict[str, Any]] = Field(description="Evidence snippets with sources")
    query_used: str = Field(description="Search query constructed")
    confidence_score: float = Field(description="Overall confidence in research findings")

class KBRecord:
    def __init__(self, jurisdiction: str, reg_code: str, name: str, section: str, url: str, excerpt: str):
        self.jurisdiction = jurisdiction
        self.reg_code = reg_code
        self.name = name
        self.section = section
        self.url = url
        self.excerpt = excerpt
        self.tokens = self._tokenize(f"{name} {section} {excerpt}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
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
        prompt_template = """
You are a legal research expert analyzing feature descriptions for regulatory compliance.

SCREENING ANALYSIS:
{screening_analysis}

RESEARCH EVIDENCE FOUND:
{evidence_summary}

TASK: Based on the screening analysis and evidence found, provide your research assessment.

Return ONLY valid JSON:
{{
    "agent": "ResearchAgent",
    "candidates": [
        {{"reg": "regulation_code", "why": "reason_for_selection", "score": 0.85}}
    ],
    "evidence": [
        {{
            "reg": "regulation_code",
            "jurisdiction": "jurisdiction_name", 
            "name": "regulation_name",
            "section": "section_reference",
            "url": "regulation_url",
            "excerpt": "relevant_text_snippet",
            "score": 8.5
        }}
    ],
    "query_used": "search_query_constructed",
    "confidence_score": 0.85
}}
"""
        
        self.prompt_template = PromptTemplate(
            input_variables=["screening_analysis", "evidence_summary"],
            template=prompt_template
        )
        
        self.output_parser = JsonOutputParser(pydantic_object=ResearchOutput)
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible process method"""
        try:
            # Extract data from state
            screening_analysis = state.get("screening_analysis", {})
            feature_description = state.get("feature_description", "")
            
            # Build search query
            search_query = self._build_search_query(screening_analysis, feature_description)
            
            # Find candidate regulations
            candidates = self._identify_candidate_regulations(screening_analysis)
            
            # Search for evidence
            evidence = self._search_evidence(search_query, candidates)
            
            # Use LLM to synthesize findings
            llm_input = {
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_summary": json.dumps(evidence[:5], indent=2)  # Top 5 for LLM
            }
            
            result = await self.chain.ainvoke(llm_input)
            
            # Enhance with our local search results
            result["evidence"] = evidence[:10]  # Return top 10 evidence pieces
            result["candidates"] = [{"reg": c[0], "why": c[1], "score": 0.8} for c in candidates]
            result["query_used"] = search_query
            
            self.log_interaction(state, result)
            
            # Return LangGraph state update
            return {
                "research_evidence": result["evidence"],
                "research_candidates": result["candidates"],
                "research_query": result["query_used"],
                "research_confidence": result.get("confidence_score", 0.7),
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
                "research_error": str(e),
                "research_completed": True,
                "next_step": "validation"
            }
    
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
            }
        ]
        
        # Write sample data
        sample_file = os.path.join(self.kb_dir, "sample_regulations.jsonl")
        with open(sample_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
    
    def _build_idf_scores(self) -> Dict[str, float]:
        """Build IDF scores for terms in knowledge base"""
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
    
    def _build_search_query(self, screening_analysis: Dict, feature_description: str) -> str:
        """Build search query from screening results"""
        query_parts = [feature_description]
        
        # Add geographic scope
        geo_scope = screening_analysis.get("geographic_scope", [])
        if isinstance(geo_scope, list):
            query_parts.extend(geo_scope)
        
        # Add age sensitivity terms
        if screening_analysis.get("age_sensitivity"):
            query_parts.append("minors under 18 age verification")
        
        # Add trigger keywords
        trigger_keywords = screening_analysis.get("trigger_keywords", [])
        if trigger_keywords:
            query_parts.extend(trigger_keywords[:3])  # Top 3
        
        return " ".join(query_parts)
    
    def _identify_candidate_regulations(self, screening_analysis: Dict) -> List[tuple]:
        """Identify candidate regulations based on screening"""
        candidates = []
        geo_scope = [g.upper() for g in screening_analysis.get("geographic_scope", [])]
        age_sensitive = screening_analysis.get("age_sensitivity", False)
        
        # California regulations
        if "CALIFORNIA" in geo_scope or "CA" in geo_scope:
            if age_sensitive:
                candidates.append(("CA_SB976", "California minors protection law"))
        
        # Utah regulations  
        if "UTAH" in geo_scope or "UT" in geo_scope:
            if age_sensitive:
                candidates.append(("UT_MINORS", "Utah social media regulations for minors"))
        
        # US federal regulations
        if "US" in geo_scope or "UNITED STATES" in geo_scope:
            candidates.append(("US_2258A", "US federal reporting requirements"))
        
        # EU regulations
        if "EU" in geo_scope or "EUROPE" in geo_scope:
            candidates.append(("EU_DSA", "EU Digital Services Act"))
        
        # Fallback if no specific matches
        if not candidates:
            candidates.append(("US_2258A", "General US compliance requirements"))
        
        return candidates
    
    def _search_evidence(self, query: str, candidates: List[tuple]) -> List[Dict]:
        """Search for evidence in knowledge base"""
        query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        evidence = []
        
        for record in self.kb_records:
            # Calculate relevance score
            doc_terms = set(record.tokens)
            overlap = query_terms.intersection(doc_terms)
            
            if overlap:
                score = sum(self.idf_scores.get(term, 1.0) for term in overlap)
                
                # Boost score for candidate regulations
                for candidate_reg, _ in candidates:
                    if record.reg_code == candidate_reg:
                        score *= 1.5
                
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