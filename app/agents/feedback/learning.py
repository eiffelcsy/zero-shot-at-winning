# app/agents/learning.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal, Callable, Union
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from ..base import BaseComplianceAgent
from ..prompts.learning_prompt import build_learning_prompt
from ..memory.memory_pg import PostgresMemoryStore, ApplyResult
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # → .../app
DEFAULT_FEEDBACK = ROOT / "data" / "feedback.jsonl"

# ---------- Pydantic models for LLM output ----------

class GlossaryItem(BaseModel):
    term: str
    expansion: str
    hints: List[str] = Field(default_factory=list)

class AgentLearning(BaseModel):
    evaluation: Literal["correct", "incorrect"]
    learning: str
    # Few-shot may be a string prompt or a structured object; accept both.
    few_shot: Optional[Union[str, Dict[str, Any]]] = None

class AgentLearnings(BaseModel):
    ScreeningAgent: AgentLearning
    ResearchAgent: AgentLearning
    ValidationAgent: AgentLearning

class LearningPlan(BaseModel):
    agent_learnings: AgentLearnings
    glossary: List[GlossaryItem] = Field(default_factory=list)
    key_lesson: str
    tags: List[str] = Field(default_factory=list)

# ---------- Agent ----------

class LearningAgent(BaseComplianceAgent):
    """
    Consumes state (including new ValidationOutput + user feedback),
    asks LLM for a LearningPlan, and applies updates:
      - Postgres: glossary + per-agent few-shots
      - Files:
          * data/feedback.jsonl (full plan, one line per run)
          * data/memory/glossary_overrides.jsonl (each glossary item)
          * data/memory/few_shots/screening.jsonl
          * data/memory/few_shots/research.jsonl
          * data/memory/few_shots/validation.jsonl
    """

    def __init__(self, feedback_file: str = str(DEFAULT_FEEDBACK), pg_conn: Optional[str] = None):
        super().__init__("LearningAgent", temperature=0.0)
        self.feedback_file = feedback_file
        self._ensure_parent_dir(self.feedback_file)

        # No overlay needed for LearningAgent; it only generates updates.
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
          - validation_analysis (new ValidationOutput schema)
          - screening_analysis, research_analysis
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
            decision = state["validation_analysis"]
            user_feedback = state.get("user_feedback", {"is_correct": "yes", "notes": ""})

            # Inputs must match the variables used in build_learning_prompt
            llm_input = {
                "feature": json.dumps(feature, ensure_ascii=False, indent=2),
                "screening": json.dumps(screening, ensure_ascii=False, indent=2),
                "research": json.dumps(research, ensure_ascii=False, indent=2),
                "decision": json.dumps(decision, ensure_ascii=False, indent=2),
                "feedback": json.dumps(user_feedback, ensure_ascii=False, indent=2),
            }

            plan = await self.safe_llm_call(llm_input)
            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()  # Pydantic v2 model -> dict

            # ---------- Persist full plan to feedback.jsonl ----------
            self._append_jsonl(self.feedback_file, plan)

            # ---------- Apply glossary updates (DB + JSONL) ----------
            glossary_items: List[Dict[str, Any]] = plan.get("glossary", []) or []
            applied_glossary: ApplyResult = self.memory.update_glossary(glossary_items)

            self._append_many_jsonl(
                ROOT / "data" / "memory" / "glossary_overrides.jsonl",
                glossary_items,
            )

            # ---------- Apply per-agent few-shot updates ----------
            # Convert to DB schema expected by PostgresMemoryStore.add_few_shots()
            # NS keys are lowercased: "screening" | "research" | "validation"
            ag_map = {
                "ScreeningAgent": "screening",
                "ResearchAgent": "research",
                "ValidationAgent": "validation",
            }

            agent_learnings: Dict[str, Dict[str, Any]] = plan.get("agent_learnings", {}) or {}
            applied_counts = {"screening": 0, "research": 0, "validation": 0}

            for agent_key, ns in ag_map.items():
                agent_block = agent_learnings.get(agent_key)
                if not agent_block:
                    continue
                few = agent_block.get("few_shot")
                if not few:
                    continue

                # Normalize a few-shot example object for DB and JSONL.
                # Accept either string or dict. Store minimally and consistently.
                if isinstance(few, str):
                    ex = {"agent": ns, "example": few}
                elif isinstance(few, dict):
                    ex = {"agent": ns, **few}
                else:
                    # Skip unknown shapes
                    continue

                # DB write
                res = self.memory.add_few_shots([ex])
                applied_counts[ns] += getattr(res, "applied", 0) or 0

                # JSONL write
                fewshots_dir = ROOT / "data" / "memory" / "few_shots"
                fewshots_dir.mkdir(parents=True, exist_ok=True)
                per_agent_path = fewshots_dir / f"{ns}.jsonl"
                self._append_jsonl(per_agent_path, ex)

            report = {
                "learning_summary": plan.get("key_lesson", ""),
                "learning_counts": {
                    "glossary": getattr(applied_glossary, "applied", 0) or 0,
                    "few_shots_screening": applied_counts["screening"],
                    "few_shots_research": applied_counts["research"],
                    "few_shots_validation": applied_counts["validation"],
                },
                "tags": plan.get("tags", []),
                "learning_timestamp": datetime.now().isoformat() + "Z",
            }

            # Optional hot-reload of prompts for running agents
            if getattr(self, "refresh_prompts_callback", None):
                try:
                    self.refresh_prompts_callback()
                except Exception as e:
                    self.logger.warning(f"Prompt refresh failed: {e}")

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

    # ---------- file helpers ----------

    def _ensure_parent_dir(self, pathlike: os.PathLike | str):
        p = Path(pathlike)
        p.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, pathlike: os.PathLike | str, item: Any):
        p = Path(pathlike)
        self._ensure_parent_dir(p)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _append_many_jsonl(self, pathlike: os.PathLike | str, items: List[Any]):
        if not items:
            return
        p = Path(pathlike)
        self._ensure_parent_dir(p)
        with p.open("a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
