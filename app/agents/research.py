from pydantic import BaseModel, Field
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
    agent: str = Field(description="Agent name")
    regulations: List[Dict[str, Any]] = Field(description="Regulations found")
    queries_used: List[str] = Field(description="Queries used to retrieve documents")
    confidence_score: float = Field(description="Confidence score")
    retrieved_documents: List[Dict[str, Any]] = Field(description="Retrieved documents")

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
        return search_query

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """RAG-based research process using vector storage and retrieval"""
        try:
            # Extract inputs from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            screening_analysis = state.get("screening_analysis", {})
            
            if not feature_name or not feature_description:
                raise ValueError("Missing feature name or description")
            
            if not screening_analysis:
                raise ValueError("Missing screening analysis from previous agent")

            # Step 1: Generate search query using LLM
            base_query = await self._generate_search_query_llm(screening_analysis)

            # Step 2: Retrieve documents using the RetrievalTool (handles query enhancement + retrieval)
            retrieved_documents = await self.retrieval_tool.ainvoke({
                "query": base_query,
                "n_results_per_query": 5
            })
            
            # Get the enhanced query for logging purposes
            expanded_queries = retrieved_documents["enhanced_queries"]

            # Step 4: Extract regulations from retrieved docs (combining candidates and evidence)
            regulations = self._extract_regulations(retrieved_documents["raw_results"])
            
            # Step 5: Use LLM for final synthesis
            llm_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(regulations, indent=2),
            }

            result = await self.safe_llm_call(llm_input)

            # Step 6: Holistic confidence score of document similarity and LLM reasoning quality
            confidence_score = self._calculate_overall_confidence(regulations, result)

            # Step 7: Enhance result with RAG insights
            result["regulations"] = regulations
            result["queries_used"] = expanded_queries
            result["agent"] = "ResearchAgent"
            result["confidence_score"] = confidence_score

            self.log_interaction(state, result)

            # Return enhanced state update with consistent field names
            return {
                "research_analysis": {
                    "agent": "ResearchAgent",
                    "regulations": result["regulations"],
                    "queries_used": result["queries_used"],
                    "confidence_score": result["confidence_score"],
                    "retrieved_documents": retrieved_documents["raw_results"]
                },
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"
            }

        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
            # Simplified error handling - return error in state
            return {
                "research_analysis": {
                    "agent": "ResearchAgent",
                    "regulations": [],
                    "queries_used": [],
                    "confidence_score": 0.0,
                    "retrieved_documents": [],
                    "error": str(e)
                },
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"
            }

    def _calculate_overall_confidence(self, regulations: List[Dict[str, Any]], llm_result: Dict) -> float:
        """Calculate confidence considering both RAG results and LLM analysis"""
        
        # Base confidence from document similarity
        if not regulations:
            rag_confidence = 0.0
        else:
            normalized_scores = [reg["relevance_score"] for reg in regulations]
            rag_confidence = sum(normalized_scores) / len(normalized_scores)
        
        # LLM confidence (if provided)
        llm_confidence = llm_result.get("confidence_score", 0.5)
        
        # Combine both confidences (weighted average)
        combined_confidence = (rag_confidence * 0.6) + (llm_confidence * 0.4)
        
        return round(combined_confidence, 3)

    def _extract_regulations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract regulations from retrieved documents with specified fields"""
        regulations = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            reg_name = metadata.get("regulation_name", "Unknown Regulation")
            source_filename = metadata.get("source_filename", "unknown.pdf")
            
            # Get the document content as excerpt (quoted verbatim)
            excerpt = doc.get("document", "")
            
            # Fix: Use improved confidence calculation
            distance = doc.get("distance", 1.0)
            relevance_score = self._calculate_regulation_confidence(distance)
            
            # Create regulation entry
            regulation_entry = {
                "source_filename": source_filename,
                "regulation_name": reg_name,
                "excerpt": excerpt,
                "relevance_score": relevance_score
            }
            
            regulations.append(regulation_entry)
        
        # Sort by confidence score descending
        regulations.sort(key=lambda x: x["relevance_score"], reverse=True)
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

