from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import asyncio
import json
from datetime import datetime
from logs.logging_config import get_logger


class BaseComplianceAgent:
    def __init__(self, name: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0, memory_overlay: str = ""):
        self.name = name
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt_template: Optional[PromptTemplate] = None
        self.output_parser: Optional[JsonOutputParser] = None
        self.chain = None
        self.memory_overlay = memory_overlay
        
        # Enhanced logging setup using the centralized logging configuration
        self.logger = get_logger(f"agent.{name}")
        
        # Track agent execution metrics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        
        # Log agent initialization with memory overlay status
        if self.memory_overlay:
            self.logger.info(f"Agent {name} initialized with memory overlay ({len(self.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info(f"Agent {name} has access to TikTok terminology context")
            else:
                self.logger.warning(f"Agent {name} memory overlay missing TikTok terminology")
        else:
            self.logger.warning(f"Agent {name} initialized with NO memory overlay - will lack TikTok context")
    
    def create_chain(self, prompt_template: PromptTemplate, output_model: Optional[BaseModel] = None):
        """Standard LangChain setup with enhanced error checking and memory overlay integration"""
        if not isinstance(prompt_template, PromptTemplate):
            raise ValueError(f"Expected PromptTemplate, got {type(prompt_template)}")
        
        self.prompt_template = prompt_template
        
        # Log the prompt template variables and memory integration
        self.logger.info(f"Creating chain for {self.name} with variables: {prompt_template.input_variables}")
        
        if output_model:
            self.output_parser = JsonOutputParser(pydantic_object=output_model)
            self.chain = self.prompt_template | self.llm | self.output_parser
            self.logger.info(f"Chain created with output model: {output_model.__name__}")
        else:
            self.chain = self.prompt_template | self.llm
            self.logger.info("Chain created without output model")
        
        # Verify memory overlay integration
        if self.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
            self.logger.info(f"Chain for {self.name} includes TikTok terminology context")
        else:
            self.logger.warning(f"Chain for {self.name} missing TikTok terminology context")
    
    def update_memory(self, new_memory_overlay: str):
        """Update memory overlay and rebuild chain with new context"""
        old_overlay_length = len(self.memory_overlay) if self.memory_overlay else 0
        new_overlay_length = len(new_memory_overlay) if new_memory_overlay else 0
        
        self.logger.info(f"Updating memory overlay for {self.name}: {old_overlay_length} -> {new_overlay_length} characters")
        
        self.memory_overlay = new_memory_overlay
        
        # Verify TikTok terminology is present
        if new_memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in new_memory_overlay:
            self.logger.info(f"Updated memory overlay for {self.name} includes TikTok terminology")
        else:
            self.logger.warning(f"Updated memory overlay for {self.name} missing TikTok terminology")
        
        # Rebuild chain if prompt template exists
        if self.prompt_template:
            self.create_chain(self.prompt_template, self.output_parser.pydantic_object if self.output_parser else None)
            self.logger.info(f"Chain rebuilt for {self.name} with updated memory overlay")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Every agent must implement this - enhanced with timing and error tracking"""
        raise NotImplementedError("Each agent must implement the process method")
    
    def log_interaction(self, input_data: Dict, output_data: Dict, execution_time: float = None):
        """Enhanced logging for all agents with TikTok terminology context awareness"""
        self.logger.info(f"Agent {self.name} processed request successfully")
        
        if execution_time:
            self.logger.info(f"Execution time: {execution_time:.2f}s")
        
        # Log memory overlay status
        if self.memory_overlay:
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info(f"Agent {self.name} used TikTok terminology context")
            else:
                self.logger.warning(f"Agent {self.name} memory overlay missing TikTok terminology")
        else:
            self.logger.warning(f"Agent {self.name} has no memory overlay")
        
        # Log key metrics without sensitive data
        input_summary = {
            "input_keys": list(input_data.keys()),
            "input_sizes": {k: len(str(v)) for k, v in input_data.items()}
        }
        
        output_summary = {
            "output_keys": list(output_data.keys()) if isinstance(output_data, dict) else "non-dict",
            "agent": output_data.get("agent") if isinstance(output_data, dict) else None,
            "confidence_score": output_data.get("confidence_score") if isinstance(output_data, dict) else None
        }
        
        self.logger.debug(f"Input summary: {input_summary}")
        self.logger.debug(f"Output summary: {output_summary}")
    
    def log_error(self, error: Exception, input_data: Dict, context: str = ""):
        """Standardized error logging with memory context"""
        error_details = {
            "agent": self.name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "input_keys": list(input_data.keys()) if input_data else [],
            "timestamp": datetime.now().isoformat(),
            "memory_overlay_present": bool(self.memory_overlay),
            "tiktok_terminology_present": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
        }
        
        self.logger.error(f"Agent {self.name} failed: {error_details}")
        self.failure_count += 1
    
    def log_search_query(self, query: str, context: str = "", results_count: int = 0):
        """Log search queries for compliance analysis tracking"""
        query_details = {
            "agent": self.name,
            "query": query,
            "context": context,
            "results_count": results_count,
            "timestamp": datetime.now().isoformat(),
            "memory_overlay_present": bool(self.memory_overlay),
            "tiktok_terminology_present": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
        }
        
        self.logger.info(f"Search query executed by {self.name}: {query_details}")
        
        # Also log the actual query content for debugging
        self.logger.debug(f"Raw search query: '{query}'")
    
    async def safe_llm_call(self, prompt_data: Dict, max_retries: int = 3, retry_delay: float = 1.0) -> Dict:
        """Enhanced safe LLM call with better error handling, metrics, and memory context logging"""
        start_time = datetime.now()
        self.execution_count += 1
        
        # Log the LLM call with memory context
        self.logger.info(f"Agent {self.name} making LLM call with {len(prompt_data)} input variables")
        
        if self.memory_overlay:
            if "TIKTOK TERMINOLOGY REFERENCE" in self.memory_overlay:
                self.logger.info(f"LLM call includes TikTok terminology context")
            else:
                self.logger.warning(f"LLM call missing TikTok terminology context")
        else:
            self.logger.warning(f"LLM call has no memory overlay")
        
        for attempt in range(max_retries):
            try:
                if self.chain is None:
                    raise RuntimeError(f"Chain not set up for {self.name}. Call create_chain() first.")
                
                # Validate input data
                self._validate_prompt_data(prompt_data)
                
                # Make the LLM call
                result = await self.chain.ainvoke(prompt_data)
                
                # Validate output
                validated_result = self._validate_llm_output(result)
                
                # Track success metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                self.total_execution_time += execution_time
                self.success_count += 1
                
                self.logger.debug(f"LLM call succeeded on attempt {attempt + 1}, time: {execution_time:.2f}s")
                
                # Log successful LLM call with memory context
                self.logger.info(f"LLM call successful for {self.name} in {execution_time:.2f}s")
                
                return validated_result
                
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt == max_retries - 1:
                    self.log_error(e, prompt_data, f"Final LLM call failure after {max_retries} attempts")
                    raise
                
                # Exponential backoff
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
    
    def _validate_prompt_data(self, prompt_data: Dict):
        """Validate input data for LLM calls"""
        if not isinstance(prompt_data, dict):
            raise ValueError(f"Expected dict for prompt_data, got {type(prompt_data)}")
        
        if not prompt_data:
            raise ValueError("Empty prompt_data provided")
        
        # Check if required template variables are present
        if self.prompt_template:
            required_vars = self.prompt_template.input_variables
            missing_vars = [var for var in required_vars if var not in prompt_data]
            if missing_vars:
                raise ValueError(f"Missing required prompt variables: {missing_vars}")
    
    def _validate_llm_output(self, result: Any) -> Dict:
        """Validate and clean LLM output"""
        if result is None:
            raise ValueError("LLM returned None result")
        
        # If it's already a dict, return it
        if isinstance(result, dict):
            return result
        
        # Try to parse as JSON if it's a string
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    raise ValueError(f"LLM output parsed but not a dict: {type(parsed)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM output is not valid JSON: {e}")
        
        # Try to convert to dict if it has dict-like attributes
        if hasattr(result, '__dict__'):
            return result.__dict__
        
        raise ValueError(f"Cannot convert LLM output to dict: {type(result)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent execution metrics with memory context"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        success_rate = (
            self.success_count / self.execution_count 
            if self.execution_count > 0 else 0
        )
        
        return {
            "agent_name": self.name,
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "failed_executions": self.failure_count,
            "success_rate": round(success_rate * 100, 2),
            "average_execution_time": round(avg_execution_time, 2),
            "total_execution_time": round(self.total_execution_time, 2),
            "memory_overlay_present": bool(self.memory_overlay),
            "tiktok_terminology_present": "TIKTOK TERMINOLOGY REFERENCE" in (self.memory_overlay or "")
        }
    
    def reset_metrics(self):
        """Reset agent metrics (useful for testing)"""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.logger.info(f"Metrics reset for agent {self.name}")
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', executions={self.execution_count})>"
    
    def __repr__(self) -> str:
        return self.__str__()