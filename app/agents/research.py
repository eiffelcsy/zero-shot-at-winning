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
            confidence_score = sum([regulation["confidence_score"] for regulation in regulations]) / len(regulations)

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

            # Return enhanced state update
            return {
                "research_regulations": result["regulations"],
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

    def _extract_regulations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract regulations from retrieved documents with specified fields"""
        regulations = []
        seen_regulations = set()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            reg_name = metadata.get("regulation_name", "Unknown Regulation")
            source_filename = metadata.get("source_filename", "unknown.pdf")
            
            # Get the document content as excerpt (quoted verbatim)
            excerpt = doc.get("document", "")
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "..."
            
            # Calculate confidence score based on document distance/similarity
            distance = doc.get("distance", 1.0)
            confidence_score = max(0.0, 100.0 * (1.0 - distance)) if distance is not None else 70.0
            
            # Create regulation entry
            regulation_entry = {
                "source_filename": source_filename,
                "regulation_name": reg_name,
                "excerpt": excerpt,
                "confidence_score": round(confidence_score, 1)
            }
            
            regulations.append(regulation_entry)
        
        # Sort by confidence score descending
        regulations.sort(key=lambda x: x["confidence_score"], reverse=True)
        return regulations[:10]  # Top 10 regulations

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "research_regulations": [],
            "research_query": "",
            "research_analysis": {
                "agent": "ResearchAgent",
                "regulations": [],
                "query_used": "",
                "confidence_score": 0.0,
                "error": error_message
            },
            "research_error": error_message,
            "research_completed": True,
            "research_timestamp": datetime.now().isoformat(),
            "next_step": "validation"
        }