from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json
import re

class BaseComplianceAgent(ABC):
    """Base class for all compliance agents in the TikTok geo-regulation system"""
    
    def __init__(self, name: str, llm, tools: List = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.confidence_threshold = 0.7
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Initialize agent metadata
        self.created_at = datetime.now()
        self.interaction_count = 0
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core processing method that each agent must implement.
        
        Args:
            input_data: Dictionary containing input data for the agent
            
        Returns:
            Dictionary containing agent's analysis and results
        """
        pass
    
    def calculate_confidence(self, reasoning: str, evidence: List[str]) -> float:
        """Calculate confidence score based on evidence quality and reasoning"""
        if not evidence:
            return 0.3
        
        # Base confidence from evidence count
        evidence_score = min(0.6, len(evidence) * 0.15)
        
        # Reasoning quality score
        reasoning_keywords = ["specific", "clearly", "explicitly", "documented", "regulation"]
        reasoning_score = sum(0.08 for keyword in reasoning_keywords 
                            if keyword.lower() in reasoning.lower())
        
        # Combine scores
        total_confidence = min(1.0, evidence_score + reasoning_score + 0.2)
        return round(total_confidence, 2)
    
    def _parse_llm_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON output"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._fallback_parse(raw_output)
        except json.JSONDecodeError:
            return self._fallback_parse(raw_output)
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON fails"""
        return {
            "error": "Failed to parse LLM output",
            "raw_output": text,
            "confidence": 0.1,
            "needs_human_review": True
        }
    
    def _log_interaction(self, input_data: Dict, output_data: Dict):
        """Log agent interaction for audit trail"""
        self.interaction_count += 1
        self.logger.info(f"Agent {self.name} - Interaction {self.interaction_count}")
        self.logger.debug(f"Input: {input_data}")
        self.logger.debug(f"Output: {output_data}")