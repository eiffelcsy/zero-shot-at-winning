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

            # Step 1: Generate search query using LLM
            base_query = await self._generate_search_query_llm(screening_analysis)

            # Step 2: Retrieve documents using the RetrievalTool (handles query enhancement + retrieval)
            retrieved_documents = await self.retrieval_tool.ainvoke({
                "query": base_query,
                "n_results": 5
            })
            
            # Get the enhanced query for logging purposes
            expanded_queries = retrieved_documents["enhanced_queries"]

            # Step 4: Extract regulations from retrieved docs (combining candidates and evidence)
            regulations = self._extract_regulations(retrieved_documents["raw_results"])
            
            # Fix: Use proper confidence calculation with error handling
            confidence_score = self._calculate_overall_confidence(regulations)

            # Step 5: Use LLM for final synthesis
            llm_input = {
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(regulations[:5], indent=2),
            }

            result = await self.safe_llm_call(llm_input)

            # Step 6: Enhance result with RAG insights
            result["regulations"] = regulations
            result["queries_used"] = expanded_queries
            result["agent"] = "ResearchAgent"
            result["confidence_score"] = confidence_score
            result["retrieved_documents"] = retrieved_documents["raw_results"]

            self.log_interaction(state, result)

            # Return enhanced state update with consistent field names
            return {
                "research_regulations": result["regulations"],
                "research_queries": result["queries_used"],
                "research_confidence": result["confidence_score"],
                "research_retrieved_documents": result["retrieved_documents"],
                "research_analysis": result,
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"  # Fixed: Go to validation agent, not END
            }

        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
            # Simplified error handling - return error in state
            return {
                "research_regulations": [],
                "research_queries": [],
                "research_confidence": 0.0,
                "research_retrieved_documents": [],
                "research_analysis": {
                    "agent": "ResearchAgent",
                    "regulations": [],
                    "query_used": "",
                    "confidence_score": 0.0,
                    "error": str(e)
                },
                "research_error": str(e),
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "END"
            }

    def _calculate_overall_confidence(self, regulations: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score from individual regulation scores"""
        if not regulations:
            return 0.0
        
        try:
            # Normalize individual scores from 0-100 to 0-1 range
            normalized_scores = [reg["confidence_score"] / 100.0 for reg in regulations]
            
            # Calculate weighted average (higher confidence regulations get more weight)
            total_weight = sum(normalized_scores)
            if total_weight == 0:
                return 0.0
            
            weighted_avg = sum(score * score for score in normalized_scores) / total_weight
            return round(weighted_avg, 3)
        except (KeyError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"Error calculating confidence: {e}, using fallback")
            return 0.5

    def _extract_regulations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract regulations from retrieved documents with specified fields"""
        regulations = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            reg_name = metadata.get("regulation_name", "Unknown Regulation")
            source_filename = metadata.get("source_filename", "unknown.pdf")
            
            # Get the document content as excerpt (quoted verbatim)
            excerpt = doc.get("document", "")
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "..."
            
            # Fix: Use improved confidence calculation
            distance = doc.get("distance", 1.0)
            confidence_score = self._calculate_regulation_confidence(distance)
            
            # Create regulation entry
            regulation_entry = {
                "source_filename": source_filename,
                "regulation_name": reg_name,
                "excerpt": excerpt,
                "confidence_score": confidence_score
            }
            
            regulations.append(regulation_entry)
        
        # Sort by confidence score descending
        regulations.sort(key=lambda x: x["confidence_score"], reverse=True)
        return regulations[:10]  # Top 10 regulations

    def _calculate_regulation_confidence(self, distance: float) -> float:
        """Calculate confidence score from document distance"""
        if distance is None:
            return 50.0  # Neutral confidence for unknown distance
        
        try:
            # Normalize distance to 0-1 range (assuming ChromaDB returns 0-1)
            # Higher distance = lower confidence
            confidence = max(0.0, 1.0 - distance)
            
            # Apply sigmoid-like transformation for better score distribution
            confidence = confidence ** 0.5  # Square root for better curve
            
            return round(confidence * 100, 1)  # Return 0-100 range for individual scores
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Error calculating regulation confidence: {e}, using fallback")
            return 50.0

