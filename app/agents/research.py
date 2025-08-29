from pydantic import BaseModel
from typing import List, Dict, Any
import json
from datetime import datetime

from .prompts.templates import build_research_prompt, build_search_query_prompt
from .base import BaseComplianceAgent
from langgraph.graph import END
from rag.retrieval.query_processor import QueryProcessor
from rag.retrieval.retriever import RAGRetriever
from rag.tools.retrieval_tool import RetrievalTool
from chroma.chroma_connection import get_chroma_client, get_chroma_collection

class ResearchOutput(BaseModel):
    agent: str
    candidates: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    query_used: str
    confidence_score: float

class ResearchAgent(BaseComplianceAgent):
    """Research Agent - finds relevant regulations using RAG system with ChromaDB"""

    def __init__(self, 
                embedding_model: str = "text-embedding-3-large",
                memory_overlay: str = ""):
        super().__init__("ResearchAgent")
        self.memory_overlay = memory_overlay
        
        # Initialize RAG components
        self.client = get_chroma_client()
        self.collection = get_chroma_collection(self.client)

        # Initialize query processor and retriever for the retrieval tool
        self.query_processor = QueryProcessor(llm=self.llm if hasattr(self, 'llm') else None)
        self.retriever = RAGRetriever(self.collection)
        
        # Initialize the retrieval tool
        self.retrieval_tool = RetrievalTool(
            query_processor=self.query_processor,
            retriever=self.retriever,
            embedding_model=embedding_model
        )
        
        # Setup LangChain components
        self._setup_chain()

    def _setup_chain(self):
        """Setup LangChain prompt and parser with dynamic prompt building"""
        research_prompt = build_research_prompt(self.memory_overlay)
        self.create_chain(research_prompt, ResearchOutput)
    
    def update_memory(self, new_memory_overlay: str):
        """Allow runtime updates to the prompt for learning"""
        self.memory_overlay = new_memory_overlay
        self._setup_chain()

    async def _generate_search_query_llm(self, screening_analysis: Dict) -> str:
        """Generate search query using LLM based on screening analysis"""
        search_query_prompt = build_search_query_prompt(self.memory_overlay)
        
        llm_input = {
            "screening_analysis": json.dumps(screening_analysis, indent=2)
        }
        
        # Create simple chain for query generation
        chain = search_query_prompt | self.llm
        result = await chain.ainvoke(llm_input)
        
        # Extract just the string content from the result
        search_query = result.content.strip()
        
        self.logger.info(f"Generated search query: {search_query}")
        return search_query

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """RAG-based research process using vector storage and retrieval"""
        try:
            # Extract screening analysis from state
            screening_analysis = state.get("screening_analysis", {})
            
            if not screening_analysis:
                raise ValueError("Missing screening analysis from previous agent")

            # Extract key parameters from screening
            geographic_scope = screening_analysis.get("geographic_scope", [])

            # Step 1: Generate search query using LLM
            base_query = await self._generate_search_query_llm(screening_analysis)

            # Step 2: Retrieve documents using the RetrievalTool (handles query enhancement + retrieval)
            retrieved_documents = await self.retrieval_tool.ainvoke({
                "query": base_query,
                "n_results": 5
            })
            
            # Get the enhanced query for logging purposes
            expanded_queries = retrieved_documents["enhanced_queries"]

            # Step 4: Extract candidates and evidence from retrieved docs
            candidates = self._extract_candidates(retrieved_documents["raw_results"])
            evidence = self._format_evidence(retrieved_documents["raw_results"])

            # Step 5: Calculate confidence based on retrieval quality
            confidence_score = self._calculate_confidence(retrieved_documents["raw_results"], screening_analysis)

            # Step 6: Use LLM for final synthesis
            llm_input = {
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(evidence[:5], indent=2),
            }

            result = await self.safe_llm_call(llm_input)

            # Step 7: Enhance result with RAG insights
            result["evidence"] = evidence
            result["candidates"] = candidates
            result["queries_used"] = expanded_queries
            result["agent"] = "ResearchAgent"
            result["confidence_score"] = confidence_score
            result["retrieved_documents"] = retrieved_documents["raw_results"]

            self.log_interaction(state, result)

            # Return enhanced state update
            return {
                "research_evidence": result["evidence"],
                "research_candidates": result["candidates"],
                "research_queries": result["queries_used"],
                "research_confidence": result["confidence_score"],
                "research_retrieved_documents": result["retrieved_documents"],
                "research_analysis": result,
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": END
            }

        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
            return self._create_error_response(str(e))




    def _extract_candidates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract regulation candidates from retrieved documents"""
        candidates = []
        seen_regulations = set()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            reg_code = metadata.get("regulation_code") or metadata.get("regulation_name")
            
            if reg_code and reg_code not in seen_regulations:
                seen_regulations.add(reg_code)
                
                # Calculate score based on document distance/similarity
                distance = doc.get("distance", 1.0)
                score = max(0.0, 1.0 - distance) if distance is not None else 0.7
                
                candidates.append({
                    "reg": reg_code,
                    "why": f"Retrieved from regulatory knowledge base with relevance score {score:.2f}",
                    "score": score
                })
        
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:10]  # Top 10 candidates

    def _format_evidence(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format retrieved documents as evidence"""
        evidence = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            distance = doc.get("distance", 1.0)
            
            # Convert distance to similarity score (0-10 scale)
            score = max(0.0, 10.0 * (1.0 - distance)) if distance is not None else 7.0
            
            evidence_item = {
                "reg": metadata.get("regulation_code", "UNKNOWN"),
                "jurisdiction": metadata.get("geo_jurisdiction", "Unknown"),
                "name": metadata.get("regulation_name", "Regulatory Document"),
                "section": metadata.get("section", "General"),
                "url": metadata.get("source_url", ""),
                "excerpt": doc.get("document", "")[:500] + "..." if len(doc.get("document", "")) > 500 else doc.get("document", ""),
                "score": score
            }
            evidence.append(evidence_item)
        
        return evidence

    def _calculate_confidence(self, documents: List[Dict[str, Any]], 
                            screening_analysis: Dict) -> float:
        """Calculate confidence based on retrieval quality and screening alignment"""
        if not documents:
            return 0.1
        
        # Base confidence from document relevance
        distances = [doc.get("distance", 1.0) for doc in documents if doc.get("distance") is not None]
        if distances:
            avg_distance = sum(distances) / len(distances)
            base_confidence = max(0.1, 1.0 - avg_distance)
        else:
            base_confidence = 0.7
        
        # Adjust based on geographic scope alignment
        geo_scope = screening_analysis.get("geographic_scope", [])
        if geo_scope != ["unknown"]:
            geo_match_count = 0
            for doc in documents[:5]:  # Check top 5 docs
                doc_jurisdiction = doc.get("metadata", {}).get("geo_jurisdiction", "")
                if any(geo in doc_jurisdiction for geo in geo_scope):
                    geo_match_count += 1
            
            if geo_match_count > 0:
                base_confidence = min(0.95, base_confidence + 0.1 * (geo_match_count / 5))
        
        return round(base_confidence, 2)

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
            "research_error": error_message,
            "research_completed": True,
            "research_timestamp": datetime.now().isoformat(),
            "next_step": "validation"
        }