import re
from typing import List, Optional, Any
import logging
from agents.prompts.query_prompt import build_query_expansion_prompt, build_query_variation_prompt

logger = logging.getLogger(__name__)


class QueryProcessor:

    def __init__(self, llm: Any):
        """
        Initialize the QueryProcessor.
        
        Args:
            llm: Language model for query enhancement. 
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
        if not query or not query.strip():
            return []
        
        if not self.llm:
            return [query]
        
        try:
            expansion_prompt_template = build_query_expansion_prompt()
            expansion_prompt = expansion_prompt_template.format(query=query)
            
            expansion_response = await self.llm.ainvoke(expansion_prompt)
            
            all_queries = self._parse_comma_separated_list(expansion_response.content)
            
            unique_queries = []
            seen = set()
            for q in all_queries:
                q_clean = q.strip()
                if q_clean and q_clean.lower() not in seen:
                    unique_queries.append(q_clean)
                    seen.add(q_clean.lower())
            
            return unique_queries
            
        except Exception as e:
            logger.warning(f"Error expanding query '{query}': {e}")
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