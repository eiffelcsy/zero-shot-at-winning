import re
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes and enhances queries for improved RAG retrieval.
    
    This class provides two main functionalities:
    1. Query expansion: Enhances queries with related compliance and regulatory terms
    2. Multiple query generation: Creates diverse variations of the original query
    
    Designed for use by research agents in a compliance-focused RAG system.
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the QueryProcessor.
        
        Args:
            llm: Optional language model for query enhancement. 
                 Should have an 'apredict' async method.
        """
        self.llm = llm
    
    async def expand_query(self, query: str) -> str:
        """
        Expand a query with related compliance and regulatory terms.
        
        Uses the LLM to generate additional relevant terms that can improve
        retrieval accuracy for compliance-related queries.
        
        Args:
            query: The original query string
            
        Returns:
            Enhanced query string with additional relevant terms
        """
        # Handle empty or None inputs
        if not query or not query.strip():
            return ""
        
        # If no LLM available, return original query
        if not self.llm:
            return query
        
        try:
            # Create a prompt for query expansion focused on compliance/regulatory context
            expansion_prompt = f"""
You are helping to expand a compliance and regulatory query for better document retrieval.

Original query: "{query}"

Generate relevant additional terms and phrases that would help find documents related to this query. Focus on:
- Legal and regulatory synonyms
- Related compliance concepts
- Jurisdictional variations
- Technical implementation terms
- Related regulatory frameworks

Provide only the additional terms as a comma-separated list, without explanations or the original query.
"""
            
            # Get expansion terms from LLM
            expansion_terms = await self.llm.apredict(expansion_prompt)
            
            # Combine original query with expansion terms
            expanded_query = f"{query} {expansion_terms}"
            return expanded_query
            
        except Exception as e:
            logger.warning(f"Error expanding query '{query}': {e}")
            # Fallback to original query on error
            return query
    
    async def generate_multiple_queries(self, query: str, count: int = 5) -> List[str]:
        """
        Generate multiple variations of a query for comprehensive retrieval.
        
        Creates diverse query variations to capture different aspects and 
        perspectives of the compliance question.
        
        Args:
            query: The original query string
            count: Number of query variations to generate (default: 5)
            
        Returns:
            List of query variations
        """
        # Handle empty or None inputs
        if not query or not query.strip():
            return []
        
        # If no LLM available, return original query as single item list
        if not self.llm:
            return [query]
        
        try:
            # Create a prompt for generating query variations
            variation_prompt = f"""
You are helping to generate {count} different variations of a compliance and regulatory query for comprehensive document retrieval.

Original query: "{query}"

Generate {count} distinct query variations that approach the topic from different angles:
- Different terminology and phrasing
- Various compliance perspectives (legal, technical, implementation)
- Different jurisdictional contexts
- Specific vs. general approaches
- Different stakeholder viewpoints

Format your response as a numbered list (1., 2., 3., etc.) with each variation on a new line.
Make each variation a complete, well-formed question or statement.
"""
            
            # Get variations from LLM
            variations_response = await self.llm.apredict(variation_prompt)
            
            # Parse the numbered list response
            variations = self._parse_numbered_list(variations_response)
            
            # Filter out empty or too-short variations
            filtered_variations = [
                v for v in variations 
                if v.strip() and len(v.strip()) > 5
            ]
            
            # Remove duplicates while preserving order
            unique_variations = []
            seen = set()
            for variation in filtered_variations:
                if variation.lower() not in seen:
                    unique_variations.append(variation)
                    seen.add(variation.lower())
            
            return unique_variations
            
        except Exception as e:
            logger.warning(f"Error generating multiple queries for '{query}': {e}")
            # Fallback to original query on error
            return [query]
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """
        Parse a numbered list from LLM response.
        
        Args:
            text: Text containing numbered list (e.g., "1. First item\n2. Second item")
            
        Returns:
            List of extracted items without numbers
        """
        lines = text.strip().split('\n')
        parsed_items = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match numbered list patterns: "1.", "1)", "1 -", etc.
            match = re.match(r'^\d+[\.\)\-\s]+(.+)$', line)
            if match:
                item = match.group(1).strip()
                if item:
                    parsed_items.append(item)
            else:
                # If line doesn't match numbered pattern but has content, include it
                if len(line) > 5:
                    parsed_items.append(line)
        
        return parsed_items