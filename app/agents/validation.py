from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, HttpUrl
from .base import BaseComplianceAgent
from .prompts.validation_prompt import build_validation_prompt
from .memory.memory_pg import PostgresMemoryStore
from typing import Dict, Any, List, Literal
from datetime import datetime
import json, os, sys

class RelatedRegulation(BaseModel):
    name: str = Field(description="Regulation name")
    jurisdiction: str = Field(description="Jurisdiction")
    section: str = Field(description="Specific section")
    url: HttpUrl = Field(description="URL to regulation")
    evidence_excerpt: str = Field(description="Supporting evidence text")

class ValidationOutput(BaseModel):
    needs_geo_logic: Literal["YES", "NO", "REVIEW"] = Field(description="Final compliance decision")
    reasoning: Dict[str, Any] = Field(description="Detailed reasoning for decision")
    related_regulations: List[RelatedRegulation] = Field(description="Regulations that support the decision")
    confidence_score: float = Field(description="Confidence in the decision 0.0-1.0")

class ValidationAgent(BaseComplianceAgent):
    """Final decision-maker agent - validates compliance requirements"""
    
    def __init__(self, memory_overlay: str = ""):
        super().__init__("ValidationAgent", temperature=0.0)

        if memory_overlay:
            # user provided overlay -> skip DB
            self.store = None
            self.memory_overlay = memory_overlay
        else:
            # lazy import and robust init; disable vectors for tests to avoid API key deps
            from .memory.memory_pg import PostgresMemoryStore
            try:
                self.store = PostgresMemoryStore(os.getenv("PG_CONN_STRING"), use_vectors=False)
                self.memory_overlay = self.store.render_overlay_for("validation")
            except Exception as e:
                # fall back cleanly
                self.store = None
                self.memory_overlay = ""
                self.logger.warning(f"ValidationAgent: memory store unavailable, continuing without overlay ({e})")

        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain prompt and parser"""
        validation_prompt = build_validation_prompt(self.memory_overlay)
        self.create_chain(validation_prompt, ValidationOutput)

    def update_memory(self, new_memory_overlay: str):
        """Allow runtime updates to the prompt for learning"""
        self.memory_overlay = new_memory_overlay
        self._setup_chain()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible process method"""
        try:
            # Extract data from state
            feature_name = state.get("feature_name", "")
            feature_description = state.get("feature_description", "")
            screening_analysis = state.get("screening_analysis", {})
            research_analysis = state.get("research_analysis", {})
            
            if not feature_name or not feature_description:
                raise ValueError("Missing feature name or description")
            
            if not screening_analysis:
                raise ValueError("Missing screening analysis from previous agent")
            
            if not research_analysis or not research_analysis.get("regulations"):
                raise ValueError("Missing research analysis from previous agent")

            # Prepare input for validation
            validation_input = {
                "feature_name": feature_name,
                "feature_description": feature_description,
                "screening_analysis": json.dumps(screening_analysis, indent=2),
                "research_analysis": json.dumps(research_analysis, indent=2)
            }
            
            # Get LLM decision
            result = await self.safe_llm_call(validation_input)
            
            self.log_interaction(validation_input, result)
            
            # Return final decision for LangGraph in enhanced format
            return {
                "validation_analysis": {
                    "needs_geo_logic": result.get("needs_geo_logic", "REVIEW"),
                    "reasoning": result.get("reasoning", ""),
                    "related_regulations": result.get("related_regulations", []),
                    "confidence": result.get("confidence", 0.5),
                    "agent": "ValidationAgent",
                    "validation_metadata": {
                        "agent": "ValidationAgent",
                        "evidence_pieces_reviewed": len(research_analysis.get("regulations", [])),
                        "regulations_cited": len(result.get("related_regulations", [])),
                        "timestamp": datetime.now().isoformat()
                    }
                },
                "validation_completed": True,
                "validation_timestamp": datetime.now().isoformat(),
                "workflow_completed": True
            }
            
        except Exception as e:
            self.logger.error(f"Validation agent failed: {e}")
            return {
                "validation_analysis": {
                    "needs_geo_logic": "REVIEW",
                    "reasoning": f"Validation failed due to processing error: {str(e)}. Human review required.",
                    "related_regulations": [],
                    "confidence": 0.0,
                    "agent": "ValidationAgent",
                    "error": str(e)
                },
                "validation_completed": True,
                "validation_timestamp": datetime.now().isoformat(),
                "validation_error": str(e),
                "workflow_completed": True
            }

