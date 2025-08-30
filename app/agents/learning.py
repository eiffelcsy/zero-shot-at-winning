# app/agents/learning.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal, Callable
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .base import BaseComplianceAgent
from .prompts import build_learning_prompt
from .memory_pg import PostgresMemoryStore, ApplyResult
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

class KBSnippet(BaseModel):
    jurisdiction: str
    reg_code: str
    name: str
    section: str
    url: HttpUrl
    excerpt: str

class FewShotExample(BaseModel):
    agent: Literal["screening", "research", "validation"]
    input_fields: Dict[str, Any]
    expected_output: Dict[str, Any]
    rationale: str

class RuleOverlay(BaseModel):
    agent: Literal["screening", "validation"]
    rule_text: str  # short imperative rule

class LearningPlan(BaseModel):
    agent: Literal["LearningAgent"] = "LearningAgent"
    summary: str = Field(description="One-paragraph summary of what will be updated")
    glossary: List[GlossaryItem] = Field(default_factory=list)
    kb_snippets: List[KBSnippet] = Field(default_factory=list)
    few_shots: List[FewShotExample] = Field(default_factory=list)
    rules: List[RuleOverlay] = Field(default_factory=list)

# ---------- util ----------

def _escape_braces(s: str) -> str:
    # Protect literal JSON braces inside prompts
    return (s or "").replace("{", "{{").replace("}", "}}")

# ---------- Agent ----------

class LearningAgent(BaseComplianceAgent):
    """
    Learns from user feedback to improve system behavior by producing a structured LearningPlan
    and applying Postgres-backed memory updates (glossary, rules, few-shots, KB snippets).
    """

    def __init__(self, feedback_file: str = str(DEFAULT_FEEDBACK), pg_conn: Optional[str] = None):
        super().__init__("LearningAgent", temperature=0.0)
        self.feedback_file = feedback_file
        self._ensure_feedback_file()
        # Postgres-backed store (gracefully falls back to in-memory per your PostgresMemoryStore)
        self.memory_overlay = " " #PostgresMemoryStore(conn_string=pg_conn or os.getenv("PG_CONN_STRING"), use_vectors=False)
        self.memory = PostgresMemoryStore(conn_string=pg_conn or os.getenv("PG_CONN_STRING"), use_vectors=False)
        self.refresh_prompts_callback: Optional[Callable[[], None]] = None
        self._setup_chain()

    def _setup_chain(self):
        """
        prompt -> llm -> JsonOutputParser(LearningPlan)
        """
        # Compose chain
        learning_prompt = build_learning_prompt(self.memory_overlay)
        self.prompt_template = learning_prompt
        self.output_parser = JsonOutputParser(pydantic_object=LearningPlan)
        self.chain = self.prompt_template | self.llm | self.output_parser
        self.create_chain(learning_prompt, LearningPlan)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects in `state`:
          - final_decision: dict from ValidationAgent
          - screening_analysis: dict from ScreeningAgent
          - research_evidence: list from ResearchAgent (may be empty)
          - feature_name, feature_description
          - user_feedback: {"is_correct": "yes"|"no", "notes": "string"}
        """
        try:
            feature = {
                "name": state.get("feature_name"),
                "description": state.get("feature_description"),
            }
            screening = state.get("screening_analysis", {})
            research = state.get("research_evidence", [])
            decision = state.get("final_decision", {})
            user_feedback = state.get("user_feedback", {"is_correct": "yes", "notes": ""})

            # Inputs to the LLM must match the variables used in build_learning_prompt
            llm_input = {
                "feature": json.dumps(feature, ensure_ascii=False, indent=2),
                "screening": json.dumps(screening, ensure_ascii=False, indent=2),
                "research": json.dumps(research[:8], ensure_ascii=False, indent=2),
                "decision": json.dumps(decision, ensure_ascii=False, indent=2),
                "feedback": json.dumps(user_feedback, ensure_ascii=False, indent=2),
            }

            # Prefer the base class safe call (it should call self.chain under the hood)
            plan = await self.safe_llm_call(llm_input)
            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()

            # Apply memory updates to Postgres
            applied_glossary: ApplyResult = self.memory.update_glossary(plan.get("glossary", []))
            applied_kb: ApplyResult = self.memory.add_kb_snippets(plan.get("kb_snippets", []))
            applied_few: ApplyResult = self.memory.add_few_shots(plan.get("few_shots", []))
            applied_rules: ApplyResult = self.memory.update_rules(plan.get("rules", []))

            applied = {
                "glossary": applied_glossary,
                "kb_snippets": applied_kb,
                "few_shots": applied_few,
                "rules": applied_rules,
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
                    "research_count": len(research),
                    "decision": decision,
                    "user_feedback": user_feedback,
                    "plan_summary": plan.get("summary", ""),
                    "plan_counts": {k: v.applied if hasattr(v, "applied") else v.get("applied") for k, v in applied.items()},
                }
            )

            report = {
                "learning_summary": plan.get("summary", ""),
                "learning_counts": {k: v.applied if hasattr(v, "applied") else v.get("applied") for k, v in applied.items()},
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
            with open(self.feedback_file, "w", encoding="utf-8"):
                pass

    # (Optional) legacy helpers kept as-is
    def collect_feedback(self, analysis_result: Dict, user_feedback: Dict):
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "analysis": analysis_result,
            "user_correction": user_feedback.get("correction", ""),
            "feedback_type": user_feedback.get("type", ""),
            "user_comments": user_feedback.get("comments", ""),
            "session_id": analysis_result.get("session_id", ""),
        }
        self._store_feedback(feedback_entry)
        self.logger.info(f"Feedback collected: {feedback_entry.get('feedback_type','')}")

    def _store_feedback(self, feedback_entry: Dict):
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")

    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        feedback_data: List[Dict[str, Any]] = self._load_all_feedback()
        if not feedback_data:
            return {"message": "No feedback data available"}

        feedback_types: Dict[str, int] = {}
        common_errors: List[Dict[str, Any]] = []
        for entry in feedback_data:
            feedback_type = entry.get("feedback_type", "unknown")
            feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
            if feedback_type == "incorrect":
                common_errors.append(
                    {
                        "original_analysis": entry.get("analysis", {}),
                        "user_correction": entry.get("user_correction", ""),
                        "timestamp": entry.get("timestamp", ""),
                    }
                )

        total = len(feedback_data)
        accurate = feedback_types.get("accurate", 0)
        accuracy_rate = (accurate / total) if total else 0.0

        return {
            "total_feedback": total,
            "feedback_breakdown": feedback_types,
            "accuracy_rate": accuracy_rate,
            "common_errors": common_errors[:5],
            "improvement_suggestions": self._generate_improvements(common_errors),
        }

    def _load_all_feedback(self) -> List[Dict]:
        feedback_data: List[Dict[str, Any]] = []
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line))
        except FileNotFoundError:
            pass
        return feedback_data

    def _generate_improvements(self, errors: List[Dict]) -> List[str]:
        suggestions: List[str] = []
        if errors:
            suggestions.append("Consider adding more examples to agent prompts.")
            suggestions.append("Review geographic scope detection logic.")
            suggestions.append("Enhance age-sensitivity keyword matching.")
        return suggestions
