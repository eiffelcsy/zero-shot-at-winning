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
                 Should have an 'ainvoke' async method.
        """
        self.llm = llm
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a query with related compliance and regulatory terms.
        
        Uses the LLM to generate additional relevant terms that can improve
        retrieval accuracy for compliance-related queries.
        
        Args:
            query: The original query string
            
        Returns:
            List of enhanced query strings including the original query
        """
        # Handle empty or None inputs
        if not query or not query.strip():
            return []
        
        # If no LLM available, return original query as single item list
        if not self.llm:
            return [query]
        
        try:
            # Enhanced logging for query expansion debugging
            logger.info(f"Expanding query: '{query}' (length: {len(query)} characters, {len(query.split())} words)")
            
            # Create a prompt for query expansion focused on compliance/regulatory context
            expansion_prompt = f"""
You are helping to expand a compliance and regulatory query for better document retrieval.

Original query: "{query}"

Generate relevant additional terms and phrases that work as queries on their own and would help find documents related to this query. Focus on:
- Legal and regulatory synonyms
- Related compliance concepts
- Jurisdictional variations
- Technical implementation terms
- Related regulatory frameworks

Provide only the 5 most relevant additional terms as a comma-separated list, without explanations or the original query.
"""
            # Get expansion terms from LLM
            expansion_response = await self.llm.ainvoke(expansion_prompt)
            
            # Parse the comma-separated response into a list
            all_queries = self._parse_comma_separated_list(expansion_response.content)
            
            # Enhanced logging of expansion results
            logger.info(f"Query expansion generated {len(all_queries)} queries")
            logger.info(f"Raw expansion response: {expansion_response.content[:200]}...")
            
            # Remove duplicates while preserving order
            unique_queries = []
            seen = set()
            for q in all_queries:
                q_clean = q.strip()
                if q_clean and q_clean.lower() not in seen:
                    unique_queries.append(q_clean)
                    seen.add(q_clean.lower())
            
            # Enhanced logging of final queries
            logger.info(f"Final unique queries: {unique_queries}")
            for i, q in enumerate(unique_queries):
                logger.info(f"  Query {i+1}: '{q}' ({len(q)} characters)")
            
            return unique_queries
            
        except Exception as e:
            logger.warning(f"Error expanding query '{query}': {e}")
            # Fallback to original query on error
            return [query]
    
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
            variations_response = await self.llm.ainvoke(variation_prompt)
            
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
    
    def _parse_comma_separated_list(self, text: str) -> List[str]:
        """
        Parse a comma-separated list from LLM response.
        
        Args:
            text: Text containing comma-separated items
            
        Returns:
            List of extracted items without commas, cleaned and filtered
        """
        if not text or not text.strip():
            return []
        
        # Split by commas and clean each item
        items = [item.strip() for item in text.split(',')]
        
        # Filter out empty items and items that are too short
        filtered_items = [
            item for item in items 
            if item and len(item) > 2
        ]
        
        return filtered_items