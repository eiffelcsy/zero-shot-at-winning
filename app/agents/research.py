from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
from datetime import datetime

from .prompts.research_prompt import build_research_prompt, build_search_query_prompt
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
        
        # Debug: Log memory overlay status
        if self.memory_overlay:
            self.logger.info(f"Research agent initialized with memory overlay ({len(self.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("TikTok terminology found in memory overlay")
            else:
                self.logger.warning("TikTok terminology NOT found in memory overlay")
        else:
            self.logger.warning("Research agent initialized with NO memory overlay")
        
        self.create_chain(research_prompt, ResearchOutput)
    
    def update_memory(self, new_memory_overlay: str):
        """Allow runtime updates to the prompt for learning"""
        self.memory_overlay = new_memory_overlay
        self._setup_chain()

    async def _generate_search_query_llm(self, screening_analysis: Dict) -> str:
        """Generate optimized search query using LLM with TikTok terminology context"""
        
        # Extract key information for query generation
        trigger_keywords = screening_analysis.get("trigger_keywords", [])
        geographic_scope = screening_analysis.get("geographic_scope", [])
        data_sensitivity = screening_analysis.get("data_sensitivity", "")
        age_sensitivity = screening_analysis.get("age_sensitivity", False)
        
        # Build context for LLM
        context = {
            "trigger_keywords": trigger_keywords,
            "geographic_scope": geographic_scope,
            "data_sensitivity": data_sensitivity,
            "age_sensitivity": age_sensitivity,
            "terminology_analysis": screening_analysis.get("terminology_analysis", {})
        }
        
        try:
            # Create a direct LLM call for search query generation
            # This bypasses the validation that expects research prompt variables
            search_prompt = build_search_query_prompt(self.memory_overlay)
            
            # Debug: Log search prompt status
            if self.memory_overlay:
                self.logger.info(f"Search query generation using memory overlay ({len(self.memory_overlay)} characters)")
                if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                    self.logger.info("TikTok terminology available for search query generation")
                else:
                    self.logger.warning("TikTok terminology NOT available for search query generation")
            else:
                self.logger.warning("Search query generation with NO memory overlay")
            
            formatted_prompt = search_prompt.format(screening_analysis=context)
            result = await self.llm.ainvoke(formatted_prompt)
            if isinstance(result, str):
                return result
            elif isinstance(result, dict) and "query" in result:
                return result["query"]
            else:
                # Fallback to basic query construction
                return self._build_fallback_query(context)
        except Exception as e:
            self.logger.warning(f"LLM query generation failed: {e}, using fallback")
            return self._build_fallback_query(context)
    
    def _build_fallback_query(self, context: Dict) -> str:
        """Build a fallback search query when LLM generation fails"""
        keywords = context.get("trigger_keywords", [])
        geo_scope = context.get("geographic_scope", [])
        
        # Basic query construction
        query_parts = []
        
        # Add compliance-related terms
        if keywords:
            query_parts.extend(keywords)
        
        # Add geographic context
        if geo_scope and geo_scope != ["global"]:
            query_parts.extend(geo_scope)
        
        # Add data sensitivity context
        if context.get("data_sensitivity"):
            query_parts.append("data protection")
            query_parts.append("privacy")
        
        # Add age sensitivity context
        if context.get("age_sensitivity"):
            query_parts.append("minor protection")
            query_parts.append("age verification")
        
        # Add TikTok terminology context
        terminology = context.get("terminology_analysis", {})
        if terminology.get("acronyms_found"):
            for acronym in terminology["acronyms_found"]:
                meaning = terminology.get("acronym_meanings", {}).get(acronym, "")
                if meaning:
                    query_parts.append(meaning)
        
        # Ensure we have a meaningful query
        if not query_parts:
            query_parts = ["compliance", "regulations", "legal requirements"]
        
        return " ".join(query_parts)

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

            # Step 1: Generate search query using LLM with TikTok terminology context
            base_query = await self._generate_search_query_llm(screening_analysis)
            
            # Log the generated query for debugging
            self.logger.info(f"Generated search query: {base_query}")

            # Step 2: Retrieve documents using the RetrievalTool (handles query enhancement + retrieval)
            retrieved_documents = await self.retrieval_tool.ainvoke({
                "query": base_query,
                "n_results_per_query": 5
            })
            
            # Get the enhanced query for logging purposes
            expanded_queries = retrieved_documents.get("enhanced_queries", [base_query])
            
            # Log retrieval results for debugging
            self.logger.info(f"Retrieved {len(retrieved_documents.get('raw_results', []))} documents")
            self.logger.info(f"Enhanced queries: {expanded_queries}")

            # Step 4: Extract regulations from retrieved docs with proper relevance scoring
            regulations = self._extract_regulations(retrieved_documents.get("raw_results", []))
            
            # Step 5: Use LLM for final synthesis with proper input format
            llm_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(regulations, indent=2),
            }

            result = await self.safe_llm_call(llm_input)

            # Step 6: Calculate confidence based on regulation relevance scores
            confidence_score = self._calculate_overall_confidence(regulations, result)

            # Step 7: Ensure proper output schema
            if not isinstance(result, dict):
                result = {}
            
            result["regulations"] = regulations
            result["queries_used"] = expanded_queries
            result["agent"] = "ResearchAgent"
            result["confidence_score"] = confidence_score
            
            # Log the final result for debugging
            self.logger.info(f"Research completed with {len(regulations)} regulations, confidence: {confidence_score}")

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
                "workflow_completed": False,
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
            # Extract relevance scores and filter out invalid ones
            relevance_scores = []
            for reg in regulations:
                score = reg.get("relevance_score", 0.0)
                if isinstance(score, (int, float)) and 0 <= score <= 1:
                    relevance_scores.append(score)
                else:
                    self.logger.warning(f"Invalid relevance score: {score}, skipping")
            
            if relevance_scores:
                rag_confidence = sum(relevance_scores) / len(relevance_scores)
            else:
                rag_confidence = 0.0
        
        # LLM confidence (if provided)
        llm_confidence = llm_result.get("confidence_score", 0.5)
        if not isinstance(llm_confidence, (int, float)) or not (0 <= llm_confidence <= 1):
            llm_confidence = 0.5
        
        # Combine both confidences (weighted average)
        combined_confidence = (rag_confidence * 0.7) + (llm_confidence * 0.3)
        
        # Log confidence calculation for debugging
        self.logger.info(f"Confidence calculation: RAG={rag_confidence:.3f}, LLM={llm_confidence:.3f}, Combined={combined_confidence:.3f}")
        
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
            return 0.5  # Neutral confidence for unknown distance
        
        try:
            # ChromaDB typically returns cosine distances where:
            # - Lower distance = higher similarity (more relevant)
            # - Higher distance = lower similarity (less relevant)
            
            # For cosine similarity: distance 0.0 = perfect match, distance 1.0 = no similarity
            # Convert to confidence: 0.0 distance = 1.0 confidence, 1.0 distance = 0.0 confidence
            confidence = max(0.0, 1.0 - distance)
            
            # Apply sigmoid-like transformation for better score distribution
            confidence = confidence ** 0.5  # Square root for better curve
            
            # Ensure we don't get 0.0 confidence for very similar documents
            if confidence < 0.1:
                confidence = 0.1  # Minimum confidence threshold
            
            return round(confidence, 3)  # Return 0.1-1.0 range for individual scores
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Error calculating regulation confidence: {e}, using fallback")
            return 0.5

