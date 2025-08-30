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

class ResearchAgent(BaseComplianceAgent):
    """Research Agent - finds relevant regulations using RAG system with ChromaDB"""

    def __init__(self, 
                embedding_model: str = "text-embedding-3-large",
                memory_overlay: str = ""):
        super().__init__("ResearchAgent", memory_overlay=memory_overlay)
        
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
        """Setup LangChain prompt and parser with dynamic prompt building and TikTok terminology context"""
        research_prompt = build_research_prompt(self.memory_overlay)
        
        # Enhanced logging for memory overlay integration
        if self.memory_overlay:
            self.logger.info(f"Research agent initialized with memory overlay ({len(self.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("✓ TikTok terminology found in memory overlay - agents will understand TikTok acronyms")
                self.logger.info("✓ Agents can now properly interpret: NR, PF, GH, CDS, DRT, LCP, Redline, Softblock, Spanner, ShadowMode, T5, ASL, Glow, NSP, Jellybean, EchoTrace, BB, Snowcap, FR, IMT")
            else:
                self.logger.warning("⚠ TikTok terminology NOT found in memory overlay - agents may miss TikTok-specific context")
        else:
            self.logger.warning("⚠ Research agent initialized with NO memory overlay - will lack TikTok terminology context")
        
        self.create_chain(research_prompt, ResearchOutput)
    
    def update_memory(self, new_memory_overlay: str):
        """Allow runtime updates to the prompt for learning with TikTok terminology context"""
        self.logger.info(f"Updating research agent memory overlay: {len(self.memory_overlay or '')} -> {len(new_memory_overlay)} characters")
        
        # Call parent method to update memory overlay
        super().update_memory(new_memory_overlay)
        
        # Rebuild the chain with new memory context
        research_prompt = build_research_prompt(new_memory_overlay)
        self.create_chain(research_prompt, ResearchOutput)
        
        self.logger.info("✓ Research agent chain rebuilt with updated TikTok terminology context")
    
    def _build_fallback_query(self, context: Dict) -> str:
        """Build a fallback query when LLM query generation fails"""
        feature_name = context.get("feature_name", "feature")
        feature_description = context.get("feature_description", "")
        
        # Basic query construction with TikTok terminology awareness
        query_parts = [feature_name]
        
        if feature_description:
            # Extract key terms from description
            words = feature_description.split()[:10]  # Limit to first 10 words
            query_parts.extend(words)
        
        # Add TikTok compliance context if available
        if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
            query_parts.extend(["TikTok", "compliance", "regulation"])
        
        return " ".join(query_parts)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """RAG-based research process using vector storage and retrieval with TikTok terminology context"""
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
            
            # Log the generated query for compliance tracking
            self.log_search_query(
                query=base_query,
                context=f"Feature: {feature_name}",
                results_count=0  # Will be updated after retrieval
            )

            # Step 2: Retrieve documents using the RetrievalTool (handles query enhancement + retrieval)
            retrieved_documents = await self.retrieval_tool.ainvoke({
                "query": base_query,
                "n_results_per_query": 5
            })
            
            # Get the enhanced query for logging purposes
            expanded_queries = retrieved_documents.get("enhanced_queries", [base_query])
            
            # Log retrieval results for compliance tracking
            raw_results = retrieved_documents.get('raw_results', [])
            self.logger.info(f"Retrieved {len(raw_results)} documents for compliance analysis")
            self.logger.info(f"Enhanced queries: {expanded_queries}")
            
            # Log each enhanced query for detailed tracking
            for i, query in enumerate(expanded_queries):
                self.log_search_query(
                    query=query,
                    context=f"Enhanced query {i+1} for feature: {feature_name}",
                    results_count=len(raw_results)
                )

            # Step 4: Extract regulations from retrieved docs with proper relevance scoring
            regulations = self._extract_regulations(raw_results)
            
            # Step 5: Use LLM for final synthesis with proper input format and TikTok context
            synthesis_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "retrieved_regulations": json.dumps(regulations, indent=2),
                "queries_used": json.dumps(expanded_queries, indent=2)
            }
            
            # Log the synthesis attempt with memory context
            self.logger.info(f"Research agent synthesizing results with {len(regulations)} regulations found")
            if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("✓ Synthesis includes TikTok terminology context for better compliance analysis")
            else:
                self.logger.warning("⚠ Synthesis missing TikTok terminology context")
            
            final_result = await self.safe_llm_call(synthesis_input)
            
            # Step 6: Format final output with enhanced metadata
            research_analysis = {
                "agent": "ResearchAgent",
                "regulations": regulations,
                "queries_used": expanded_queries,
                "confidence_score": final_result.get("confidence_score", 0.0),
                "documents_retrieved": len(raw_results),
                "synthesis_result": final_result,
                "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or ""),
                "memory_overlay_length": len(self.memory_overlay) if self.memory_overlay else 0
            }
            
            # Log successful research completion
            self.logger.info(f"✓ Research agent completed analysis for '{feature_name}' with {len(regulations)} regulations")
            self.logger.info(f"✓ TikTok terminology context: {'Available' if research_analysis['tiktok_terminology_used'] else 'Missing'}")
            
            return {
                "research_analysis": research_analysis,
                "research_completed": True,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"
            }
            
        except Exception as e:
            self.log_error(e, state, "Research agent process failed")
            return {
                "research_analysis": {
                    "agent": "ResearchAgent",
                    "error": str(e),
                    "regulations": [],
                    "queries_used": [],
                    "confidence_score": 0.0,
                    "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
                },
                "research_completed": False,
                "research_timestamp": datetime.now().isoformat(),
                "next_step": "validation"
            }

    async def _generate_search_query_llm(self, screening_analysis: Dict) -> str:
        """Generate search query using LLM with TikTok terminology context"""
        try:
            # Build search query prompt with TikTok terminology context
            search_prompt = build_search_query_prompt(self.memory_overlay)
            
            # Log the search query generation attempt
            self.logger.info("Generating search query using LLM with TikTok terminology context")
            
            # Prepare input for search query generation
            search_input = {
                "feature_name": screening_analysis.get("feature_name", ""),
                "feature_description": screening_analysis.get("feature_description", ""),
                "screening_analysis": json.dumps(screening_analysis, indent=2)
            }
            
            # Generate search query
            search_result = await self.safe_llm_call(search_input)
            
            # Extract the generated query
            generated_query = search_result.get("search_query", "")
            
            if not generated_query:
                self.logger.warning("LLM failed to generate search query, using fallback")
                generated_query = self._build_fallback_query(search_input)
            
            # Log the generated query
            self.logger.info(f"Generated search query: '{generated_query}'")
            
            # Check if query includes TikTok terminology
            if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                tiktok_terms = ["NR", "PF", "GH", "CDS", "DRT", "LCP", "Redline", "Softblock", "Spanner", "ShadowMode", "T5", "ASL", "Glow", "NSP", "Jellybean", "EchoTrace", "BB", "Snowcap", "FR", "IMT"]
                found_terms = [term for term in tiktok_terms if term in generated_query]
                if found_terms:
                    self.logger.info(f"✓ Search query includes TikTok terminology: {found_terms}")
                else:
                    self.logger.info("Search query generated without specific TikTok terminology (may be appropriate)")
            
            return generated_query
            
        except Exception as e:
            self.logger.error(f"Failed to generate search query using LLM: {e}")
            self.logger.info("Using fallback query generation")
            return self._build_fallback_query({
                "feature_name": screening_analysis.get("feature_name", ""),
                "feature_description": screening_analysis.get("feature_description", "")
            })

    def _extract_regulations(self, raw_results: List[Dict]) -> List[Dict[str, Any]]:
        """Extract and format regulations from retrieved documents with TikTok terminology context"""
        regulations = []
        
        for i, result in enumerate(raw_results):
            try:
                # Extract document content and metadata
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                
                # Basic regulation extraction
                regulation = {
                    "id": f"reg_{i+1}",
                    "content": content[:500] + "..." if len(content) > 500 else content,  # Truncate for readability
                    "metadata": metadata,
                    "relevance_score": result.get("score", 0.0),
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", "unknown")
                }
                
                # Check if content contains TikTok terminology (if memory overlay is available)
                if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                    tiktok_terms = ["NR", "PF", "GH", "CDS", "DRT", "LCP", "Redline", "Softblock", "Spanner", "ShadowMode", "T5", "ASL", "Glow", "NSP", "Jellybean", "EchoTrace", "BB", "Snowcap", "FR", "IMT"]
                    found_terms = [term for term in tiktok_terms if term in content]
                    if found_terms:
                        regulation["tiktok_terminology_found"] = found_terms
                        regulation["relevance_score"] = min(regulation["relevance_score"] * 1.2, 1.0)  # Boost relevance for TikTok-specific content
                        self.logger.info(f"✓ Regulation {i+1} contains TikTok terminology: {found_terms}")
                
                regulations.append(regulation)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract regulation {i+1}: {e}")
                continue
        
        # Sort by relevance score
        regulations.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        self.logger.info(f"Extracted {len(regulations)} regulations from {len(raw_results)} retrieved documents")
        
        return regulations

