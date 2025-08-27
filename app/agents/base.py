from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Dict, Any
import logging
import asyncio
from datetime import datetime

class BaseComplianceAgent:
    def __init__(self, name: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.name = name
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt_template: PromptTemplate = None
        self.output_parser: JsonOutputParser = None
        self.chain = None
        
        # Logging setup
        self.logger = logging.getLogger(f"agent.{name}")
        
    def create_chain(self, prompt_template: PromptTemplate, output_model: BaseModel = None):
        """Standard LangChain setup"""
        self.prompt_template = prompt_template
        if output_model:
            self.output_parser = JsonOutputParser(pydantic_object=output_model)
            self.chain = self.prompt_template | self.llm | self.output_parser
        else:
            self.chain = self.prompt_template | self.llm
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Every agent must implement this"""
        raise NotImplementedError("Each agent must implement the process method")
    
    def log_interaction(self, input_data: Dict, output_data: Dict):
        """Standard logging for all agents"""
        self.logger.info(f"Agent {self.name} processed request")
        self.logger.debug(f"Input: {input_data}")
        self.logger.debug(f"Output: {output_data}")
    
    async def safe_llm_call(self, prompt_data: Dict, max_retries: int = 3) -> Dict:
        """Safe LLM call with retries"""
        for attempt in range(max_retries):
            try:
                if self.chain is None:
                    raise RuntimeError("Chain not set up. Call create_chain() first.")
                
                result = await self.chain.ainvoke(prompt_data)
                return result
                
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)