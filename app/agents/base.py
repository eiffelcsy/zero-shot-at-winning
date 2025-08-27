# app/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import logging
import json
import re
import asyncio

# LangChain imports for type hints
try:
    from langchain.schema.language_model import BaseLanguageModel
    from langchain.llms.base import LLM
    from langchain.chat_models.base import BaseChatModel
except ImportError:
    # Fallback for different LangChain versions
    BaseLanguageModel = Any
    LLM = Any
    BaseChatModel = Any


class BaseComplianceAgent(ABC):
    """Base class for all compliance agents in the TikTok geo-regulation system"""
    
    def __init__(self, name: str, llm: Union[BaseLanguageModel, LLM, BaseChatModel], tools: List = None):
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.confidence_threshold = 0.7
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Initialize agent metadata
        self.created_at = datetime.now(timezone.utc)
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
    
    async def safe_llm_call(self, prompt: str, max_retries: int = 3) -> str:
        """Safe LLM call with retries and error handling"""
        for attempt in range(max_retries):
            try:
                # Handle different LangChain LLM interfaces
                if hasattr(self.llm, 'apredict'):
                    return await self.llm.apredict(prompt)
                elif hasattr(self.llm, 'ainvoke'):
                    result = await self.llm.ainvoke(prompt)
                    # Handle different return types
                    return result.content if hasattr(result, 'content') else str(result)
                elif hasattr(self.llm, 'predict'):
                    # Fallback for sync LLMs
                    return self.llm.predict(prompt)
                else:
                    # Generic fallback
                    result = await self.llm(prompt)
                    return result.content if hasattr(result, 'content') else str(result)
                    
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)  # Brief delay before retry
    
    def calculate_confidence(self, reasoning: str, evidence: List[str], context: Dict[str, Any] = None) -> float:
        """Enhanced confidence calculation with context awareness"""
        if not evidence:
            return 0.3
        
        # Base evidence score
        evidence_score = min(0.6, len(evidence) * 0.15)
        
        # Reasoning quality - updated keywords for compliance domain
        high_confidence_keywords = [
            "specific", "clearly", "explicitly", "documented", "regulation", "law",
            "required", "mandated", "prohibited", "violates", "complies", "article"
        ]
        medium_confidence_keywords = [
            "likely", "appears", "suggests", "indicates", "probably", "seems",
            "potential", "may require", "could involve"
        ]
        low_confidence_keywords = [
            "possibly", "maybe", "unclear", "ambiguous", "uncertain", "might",
            "hard to determine", "difficult to assess", "unknown"
        ]
        
        reasoning_lower = reasoning.lower()
        
        high_matches = sum(1 for kw in high_confidence_keywords if kw in reasoning_lower)
        medium_matches = sum(1 for kw in medium_confidence_keywords if kw in reasoning_lower)
        low_matches = sum(1 for kw in low_confidence_keywords if kw in reasoning_lower)
        
        reasoning_score = (high_matches * 0.1) + (medium_matches * 0.05) - (low_matches * 0.1)
        
        # Context-based adjustments
        context_bonus = 0.0
        if context:
            if context.get("has_legal_keywords", False):
                context_bonus += 0.1
            if context.get("geographic_specificity", False):
                context_bonus += 0.1
            if context.get("regulatory_citations", False):
                context_bonus += 0.15
        
        # Final calculation
        total_confidence = min(1.0, evidence_score + reasoning_score + context_bonus + 0.2)
        return round(total_confidence, 2)
    
    def _parse_llm_output(self, raw_output: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with multiple extraction strategies"""
        try:
            # Strategy 1: Direct JSON parse
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            try:
                # Strategy 2: Extract JSON from code blocks
                json_patterns = [
                    r'``````',  # ``````
                    r'``````',      # ``````
                    r'<json>(.*?)</json>',         # <json>{ }</json>
                    r'<output>(.*?)</output>',     # <output>{ }</output>
                ]
                
                for pattern in json_patterns:
                    match = re.search(pattern, raw_output, re.DOTALL)
                    if match:
                        return json.loads(match.group(1))
                
                # Strategy 3: Find any JSON-like structure
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                    
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Fallback parsing
        return self._fallback_parse(raw_output)
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Enhanced fallback parsing when JSON fails"""
        # Try to extract key information using regex
        fallback_result = {
            "error": "Failed to parse LLM output",
            "raw_output": text,
            "confidence": 0.1,
            "needs_human_review": True
        }
        
        # Attempt to extract common fields
        text_lower = text.lower()
        
        # Risk level extraction
        if "high" in text_lower and "risk" in text_lower:
            fallback_result["risk_level"] = "HIGH"
        elif "medium" in text_lower and "risk" in text_lower:
            fallback_result["risk_level"] = "MEDIUM"
        elif "low" in text_lower and "risk" in text_lower:
            fallback_result["risk_level"] = "LOW"
        
        # Compliance required extraction
        if any(word in text_lower for word in ["compliance required", "needs compliance", "requires compliance"]):
            fallback_result["compliance_required"] = True
        elif any(word in text_lower for word in ["no compliance", "not required", "no requirement"]):
            fallback_result["compliance_required"] = False
        
        return fallback_result
    
    def _log_interaction(self, input_data: Dict, output_data: Dict):
        """Log agent interaction for audit trail"""
        self.interaction_count += 1
        self.logger.info(f"Agent {self.name} - Interaction {self.interaction_count}")
        self.logger.debug(f"Input: {input_data}")
        self.logger.debug(f"Output: {output_data}")
    
    # Configuration management methods
    @classmethod
    def from_config(cls, config: Dict[str, Any], llm):
        """Create agent from configuration dictionary"""
        return cls(
            name=config.get("name", cls.__name__),
            llm=llm,
            tools=config.get("tools", [])
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get current agent configuration"""
        return {
            "name": self.name,
            "confidence_threshold": self.confidence_threshold,
            "tools": [getattr(tool, 'name', str(tool)) for tool in self.tools],
            "created_at": self.created_at.isoformat(),
            "interaction_count": self.interaction_count
        }
    
    # Async context manager support
    async def __aenter__(self):
        """Async context manager entry"""
        self.logger.info(f"Agent {self.name} starting session")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type:
            self.logger.error(f"Agent {self.name} session failed: {exc_val}")
        else:
            self.logger.info(f"Agent {self.name} session completed successfully")
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"{self.__class__.__name__}(name='{self.name}', interactions={self.interaction_count})"
