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
        # Setup main research chain for final synthesis
        research_prompt = build_research_prompt(self.memory_overlay)
        
        # Enhanced logging for memory overlay integration
        if self.memory_overlay:
            self.logger.info(f"Research agent initialized with memory overlay ({len(self.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("TikTok terminology found in memory overlay - agents will understand TikTok acronyms")
                self.logger.info("Agents can now properly interpret: NR, PF, GH, CDS, DRT, LCP, Redline, Softblock, Spanner, ShadowMode, T5, ASL, Glow, NSP, Jellybean, EchoTrace, BB, Snowcap, FR, IMT")
            else:
                self.logger.warning("TikTok terminology NOT found in memory overlay - agents may miss TikTok-specific context")
        else:
            self.logger.warning("Research agent initialized with NO memory overlay - will lack TikTok terminology context")
        
        # Create main research chain for final synthesis
        self.create_chain(research_prompt, ResearchOutput)
        
        # Create separate search query chain (no output model needed for plain string output)
        search_query_prompt = build_search_query_prompt(self.memory_overlay)
        self.search_query_chain = search_query_prompt | self.llm
    
    def update_memory(self, new_memory_overlay: str):
        """Allow runtime updates to the prompt for learning with TikTok terminology context"""
        self.logger.info(f"Updating research agent memory overlay: {len(self.memory_overlay or '')} -> {len(new_memory_overlay)} characters")
        
        # Call parent method to update memory overlay
        super().update_memory(new_memory_overlay)
        
        # Rebuild the chain with new memory context
        research_prompt = build_research_prompt(new_memory_overlay)
        self.create_chain(research_prompt, ResearchOutput)
        
        # Also rebuild the search query chain with new memory context
        search_query_prompt = build_search_query_prompt(new_memory_overlay)
        self.search_query_chain = search_query_prompt | self.llm
        
        self.logger.info("Research agent chain and search query chain rebuilt with updated TikTok terminology context")
    
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
            
            # Debug: Log what was extracted
            self.logger.info(f"Extracted {len(regulations)} regulations from {len(raw_results)} raw results")
            if regulations:
                self.logger.info(f"First regulation keys: {list(regulations[0].keys())}")
                self.logger.info(f"First regulation content length: {len(regulations[0].get('content', ''))}")
            else:
                self.logger.warning("No regulations extracted - this will cause the synthesis to fail!")
            
            # Step 5: Use LLM for final synthesis with proper input format and TikTok context
            synthesis_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "evidence_found": json.dumps(regulations, indent=2)
            }
            
            # Debug: Log the synthesis input
            self.logger.info(f"Synthesis input keys: {list(synthesis_input.keys())}")
            self.logger.info(f"Evidence found length: {len(synthesis_input['evidence_found'])} characters")
            
            # Log the synthesis attempt with memory context
            self.logger.info(f"Research agent synthesizing results with {len(regulations)} regulations found")
            if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info("Synthesis includes TikTok terminology context for better compliance analysis")
            else:
                self.logger.warning("Synthesis missing TikTok terminology context")
            
            final_result = await self.safe_llm_call(synthesis_input)
            
            # Step 6: Format final output with enhanced metadata
            research_analysis = {
                "agent": "ResearchAgent",
                "regulations": regulations,
                "queries_used": expanded_queries,
                "confidence_score": self._calculate_research_confidence(regulations, raw_results),  # Calculate based on retrieved documents
                "documents_retrieved": len(raw_results),
                "synthesis_result": final_result,
                "tiktok_terminology_used": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or ""),
                "memory_overlay_length": len(self.memory_overlay) if self.memory_overlay else 0
            }
            
            # Log successful research completion
            self.logger.info(f"Research agent completed analysis for '{feature_name}' with {len(regulations)} regulations")
            self.logger.info(f"TikTok terminology context: {'Available' if research_analysis['tiktok_terminology_used'] else 'Missing'}")
            
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
            # Log the search query generation attempt
            self.logger.info("Generating search query using LLM with TikTok terminology context")
            
            # Prepare input for search query generation - only screening_analysis is needed
            search_input = {
                "screening_analysis": json.dumps(screening_analysis, indent=2)
            }
            
            # Generate search query using the search query chain (not safe_llm_call)
            if hasattr(self, 'search_query_chain') and self.search_query_chain:
                search_result = await self.search_query_chain.ainvoke(search_input)
                generated_query = search_result.strip() if isinstance(search_result, str) else str(search_result)
            else:
                # Fallback to safe_llm_call if chain not available
                self.logger.warning("Search query chain not available, using safe_llm_call fallback")
                search_result = await self.safe_llm_call(search_input)
                generated_query = search_result.get("search_query", "")
            
            if not generated_query:
                self.logger.warning("LLM failed to generate search query, using fallback")
                generated_query = self._build_fallback_query({
                    "feature_name": screening_analysis.get("feature_name", ""),
                    "feature_description": screening_analysis.get("feature_description", "")
                })
            
            # Log the generated query
            self.logger.info(f"Generated search query: '{generated_query}'")
            
            # Check if query includes TikTok terminology
            if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                tiktok_terms = ["NR", "PF", "GH", "CDS", "DRT", "LCP", "Redline", "Softblock", "Spanner", "ShadowMode", "T5", "ASL", "Glow", "NSP", "Jellybean", "EchoTrace", "BB", "Snowcap", "FR", "IMT"]
                found_terms = [term for term in tiktok_terms if term in generated_query]
                if found_terms:
                    self.logger.info(f"Search query includes TikTok terminology: {found_terms}")
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
        
        for result in raw_results:
            try:
                # Extract document content and metadata
                content = result.get("document", "")  # Changed from "content" to "document"
                metadata = result.get("metadata", {})
                
                # Basic regulation extraction
                regulation = {
                    "content": content[:500] + "..." if len(content) > 500 else content,  # Truncate for readability
                    "metadata": metadata,
                    "relevance_score": 1 / (1 + result.get("distance", 0.0)),  # Convert distance to relevance score
                    "source": metadata.get("source_filename", "unknown"),  # Changed from "source" to "source_filename"
                }
                
                # Check if content contains TikTok terminology (if memory overlay is available)
                if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                    tiktok_terms = ["NR", "PF", "GH", "CDS", "DRT", "LCP", "Redline", "Softblock", "Spanner", "ShadowMode", "T5", "ASL", "Glow", "NSP", "Jellybean", "EchoTrace", "BB", "Snowcap", "FR", "IMT"]
                    found_terms = [term for term in tiktok_terms if term in content]
                    if found_terms:
                        regulation["tiktok_terminology_found"] = found_terms
                        regulation["relevance_score"] = min(regulation["relevance_score"] * 1.2, 1.0)  # Boost relevance for TikTok-specific content
                        self.logger.info(f"Content contains TikTok terminology: {found_terms}")
                
                regulations.append(regulation)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract content: {e}")
                continue
        
        # Sort by relevance score
        regulations.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        self.logger.info(f"Extracted {len(regulations)} regulations from {len(raw_results)} retrieved documents")
        
        return regulations

    def _calculate_research_confidence(self, regulations: List[Dict], raw_results: List[Dict]) -> float:
        """Calculate confidence score based on retrieved documents and their relevance"""
        if not regulations or not raw_results:
            return 0.0
        
        try:
            # Calculate confidence based on:
            # 1. Number of relevant documents found
            # 2. Average relevance scores
            # 3. Content quality (non-empty content)
            
            total_docs = len(raw_results)
            relevant_docs = len([r for r in regulations if r.get("content", "").strip()])
            avg_relevance = sum(r.get("relevance_score", 0.0) for r in regulations) / len(regulations) if regulations else 0.0
            
            # Content quality score (percentage of non-empty content)
            content_quality = relevant_docs / total_docs if total_docs > 0 else 0.0
            
            # Weighted confidence calculation
            confidence = (
                (content_quality * 0.4) +      # 40% weight on content quality
                (avg_relevance * 0.4) +        # 40% weight on relevance scores
                (min(relevant_docs / 5, 1.0) * 0.2)  # 20% weight on having sufficient documents
            )
            
            self.logger.info(f"Research confidence calculation: content_quality={content_quality:.2f}, avg_relevance={avg_relevance:.2f}, relevant_docs={relevant_docs}/{total_docs}, final_confidence={confidence:.2f}")
            
            return min(max(confidence, 0.0), 1.0)  # Ensure between 0.0 and 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating research confidence: {e}")
            return 0.0

