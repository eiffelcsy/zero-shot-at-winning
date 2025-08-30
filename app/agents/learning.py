# app/agents/learning.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal, Callable
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from langchain_core.output_parsers import JsonOutputParser
from .base import BaseComplianceAgent
from .prompts.learning_prompt import build_learning_prompt
from .memory.memory_pg import PostgresMemoryStore, ApplyResult
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # → .../app
DEFAULT_FEEDBACK = ROOT / "data" / "feedback.jsonl"

# ---------- Pydantic models for LLM output ----------

class GlossaryItem(BaseModel):
    term: str
    expansion: str
    hints: List[str] = Field(default_factory=list)

class FewShotExample(BaseModel):
    agent: Literal["screening", "research", "validation"]
    input_fields: Dict[str, Any]
    expected_output: Dict[str, Any]
    rationale: str

# Kept for compatibility, but will be ignored
class RuleOverlay(BaseModel):
    agent: Literal["screening", "validation"]
    rule_text: str  # short imperative rule

class LearningPlan(BaseModel):
    agent: Literal["LearningAgent"] = "LearningAgent"
    summary: str = Field(description="One-paragraph summary of what will be updated")
    glossary: List[GlossaryItem] = Field(default_factory=list)
    # kb_snippets and rules are tolerated if the LLM returns them, but not applied
    kb_snippets: List[KBSnippet] = Field(default_factory=list)
    few_shots: List[FewShotExample] = Field(default_factory=list)
    rules: List[RuleOverlay] = Field(default_factory=list)

# ---------- Agent ----------

class LearningAgent(BaseComplianceAgent):
    """
    Learns from user feedback using ONLY the new ValidationOutput schema.
    Applies updates to the Postgres-backed memory: **glossary** and **few-shots only**.
    (KB snippets and rules are ignored.)
    """

    def __init__(self, feedback_file: str = str(DEFAULT_FEEDBACK), pg_conn: Optional[str] = None):
        super().__init__("LearningAgent", temperature=0.0)
        self.feedback_file = feedback_file
        self._ensure_feedback_file()

        # No overlay needed for LearningAgent; it just generates updates.
        self.memory_overlay = " "
        self.memory = PostgresMemoryStore(conn_string=pg_conn or os.getenv("PG_CONN_STRING"), use_vectors=False)

        self.refresh_prompts_callback: Optional[Callable[[], None]] = None
        self._setup_chain()

    def _setup_chain(self):
        """prompt -> llm -> JsonOutputParser(LearningPlan)"""
        learning_prompt = build_learning_prompt(self.memory_overlay)
        self.output_parser = JsonOutputParser(pydantic_object=LearningPlan)
        self.chain = learning_prompt | self.llm | self.output_parser
        self.create_chain(learning_prompt, LearningPlan)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Requires in `state`:
          - validation_analysis: dict (NEW ValidationOutput schema)
          - screening_analysis: dict
          - research_analysis: dict
          - feature_name, feature_description
          - user_feedback: {"is_correct": "yes"|"no", "notes": "string"}
        """
        try:
            if "validation_analysis" not in state or not isinstance(state["validation_analysis"], dict):
                raise ValueError("LearningAgent requires `validation_analysis` (new ValidationOutput schema).")

            feature = {
                "name": state.get("feature_name", ""),
                "description": state.get("feature_description", ""),
            }
            screening = state.get("screening_analysis", {}) or {}
            research = state.get("research_analysis", {}) or {}
            decision = state["validation_analysis"]  # new schema
            user_feedback = state.get("user_feedback", {"is_correct": "yes", "notes": ""})

            # Inputs to the LLM must match the variables used in build_learning_prompt
            llm_input = {
                "feature": json.dumps(feature, ensure_ascii=False, indent=2),
                "screening": json.dumps(screening, ensure_ascii=False, indent=2),
                "research": json.dumps(research, ensure_ascii=False, indent=2),
                "decision": json.dumps(decision, ensure_ascii=False, indent=2),
                "feedback": json.dumps(user_feedback, ensure_ascii=False, indent=2),
            }

            plan = await self.safe_llm_call(llm_input)
            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()

            # ---- Apply ONLY glossary and few-shots ----
            applied_glossary: ApplyResult = self.memory.update_glossary(plan.get("glossary", []))
            applied_few: ApplyResult = self.memory.add_few_shots(plan.get("few_shots", []))

            with open("app/data/glossary.jsonl", "a", encoding="utf-8") as f:
                for g in plan.get("glossary", []):
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")

            with open("app/data/fewshots.jsonl", "a", encoding="utf-8") as f:
                for ex in plan.get("few_shots", []):
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            # Explicitly ignore kb_snippets and rules (even if provided)
            applied = {
                "glossary": applied_glossary,
                "few_shots": applied_few,
            }

            # Optional hot-reload of prompts for running agents
            if getattr(self, "refresh_prompts_callback", None):
                try:
                    self.refresh_prompts_callback()
                except Exception as e:
                    self.logger.warning(f"Prompt refresh failed: {e}")

            # Append raw feedback for audit
            self._append_feedback_line(
                {
                    "timestamp": datetime.now().isoformat() + "Z",
                    "feature": feature,
                    "screening": screening,
                    "research_count": len(research.get("regulations", [])) if isinstance(research, dict) else 0,
                    "decision": decision,
                    "user_feedback": user_feedback,
                    "plan_summary": plan.get("summary", ""),
                    "plan_counts": {k: v.applied if hasattr(v, "applied") else v.get("applied")
                                    for k, v in applied.items()},
                }
            )

            report = {
                "learning_summary": plan.get("summary", ""),
                "learning_counts": {k: v.applied if hasattr(v, "applied") else v.get("applied")
                                    for k, v in applied.items()},
                "learning_timestamp": datetime.now().isoformat() + "Z",
            }
            self.log_interaction({"inputs": llm_input}, report)

            return {
                "learning_report": report,
                "workflow_completed": datetime.now().isoformat() + "Z",
                "next_step": "complete",
            }

        except Exception as e:
            self.logger.error(f"LearningAgent failed: {e}")
            return {
                "learning_report": {"error": str(e)},
                "next_step": "complete",
            }

    # ---------- feedback audit helpers ----------

    def _append_feedback_line(self, entry: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _ensure_feedback_file(self):
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file(), "w", encoding="utf-8"):
                pass
