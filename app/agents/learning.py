# from __future__ import annotations
# from typing import Dict, Any, List, Optional, Literal, Callable
# from datetime import datetime
# from pydantic import BaseModel, Field, HttpUrl
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from .base import BaseComplianceAgent
# # Use the local memory implementation in this repo
# from .memory import MemoryStore
# from .prompts.templates import build_learning_prompt
# import json
# import os


# class GlossaryItem(BaseModel):
#     term: str
#     expansion: str
#     hints: List[str] = Field(default_factory=list)


# class KBSnippet(BaseModel):
#     jurisdiction: str
#     reg_code: str
#     name: str
#     section: str
#     url: HttpUrl
#     excerpt: str


# class FewShotExample(BaseModel):
#     agent: Literal["screening", "research", "validation"]
#     input_fields: Dict[str, Any]
#     expected_output: Dict[str, Any]
#     rationale: str


# class RuleOverlay(BaseModel):
#     agent: Literal["screening", "validation"]
#     rule_text: str  # short imperative rule (“Treat business-only geofencing as NO.”)


# class LearningPlan(BaseModel):
#     agent: Literal["LearningAgent"] = "LearningAgent"
#     summary: str = Field(description="One-paragraph summary of what will be updated")
#     glossary: List[GlossaryItem] = Field(default_factory=list)
#     kb_snippets: List[KBSnippet] = Field(default_factory=list)
#     few_shots: List[FewShotExample] = Field(default_factory=list)
#     rules: List[RuleOverlay] = Field(default_factory=list)


# class LearningAgent(BaseComplianceAgent):
#     """
#     Learns from user feedback to improve system behavior by producing a structured LearningPlan
#     and applying file-backed memory updates (glossary, rules, few-shots, KB snippets).
#     """

#     def __init__(self, feedback_file: str = "data/feedback.jsonl"):
#         super().__init__("LearningAgent")
#         self.feedback_file = feedback_file
#         self._ensure_feedback_file()
#         self.memory = MemoryStore()
#         self.refresh_prompts_callback: Optional[Callable[[], None]] = None
#         self._setup_chain()

#     def _setup_chain(self):
#         """
#         Build an LC chain that:
#           prompt -> llm -> JsonOutputParser(pydantic=LearningPlan)
#         The prompt MUST include format instructions so the model returns strict JSON.
#         """
#         # Parser that validates/loads into our Pydantic model
#         self.parser = JsonOutputParser(pydantic_object=LearningPlan)

#         # Build the prompt with explicit format instructions
#         # `build_learning_prompt` should accept a `format_instructions` kwarg.
#         prompt: ChatPromptTemplate | PromptTemplate = build_learning_prompt(
#             format_instructions=self.parser.get_format_instructions()
#         )

#         # Compose the chain
#         # BaseComplianceAgent is expected to expose self.llm
#         self.chain = prompt | self.llm | self.parser

#     async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Inputs expected in state (from your graph):
#           - final_decision: dict from ValidationAgent
#           - screening_analysis: dict from ScreeningAgent
#           - research_evidence: list from ResearchAgent (may be empty if skipped)
#           - feature_name, feature_description
#           - user_feedback: {"is_correct": "yes"|"no", "notes": "string (empty if yes)"}

#         Returns a learning report + counts of applied updates.
#         """
#         try:
#             # 0) pull inputs
#             feature = {
#                 "name": state.get("feature_name"),
#                 "description": state.get("feature_description"),
#             }
#             screening = state.get("screening_analysis", {})
#             research = state.get("research_evidence", [])
#             decision = state.get("final_decision", {})
#             user_feedback = state.get("user_feedback", {"is_correct": "yes", "notes": ""})

#             # 1) ask LLM for a LearningPlan
#             llm_input = {
#                 "feature": json.dumps(feature, ensure_ascii=False, indent=2),
#                 "screening": json.dumps(screening, ensure_ascii=False, indent=2),
#                 "research": json.dumps(research[:8], ensure_ascii=False, indent=2),
#                 "decision": json.dumps(decision, ensure_ascii=False, indent=2),
#                 "feedback": json.dumps(user_feedback, ensure_ascii=False, indent=2),
#             }

#             # Prefer BaseComplianceAgent's safer call if present; otherwise call the chain directly
#             # plan = None
#             # if hasattr(self, "safe_llm_call") and callable(getattr(self, "safe_llm_call")):
#                 # BaseComplianceAgent.safe_llm_call typically uses self.chain internally.
#             plan = await self.safe_llm_call(llm_input)
#             # if plan is None:
#             #     plan = await self.chain.ainvoke(llm_input)

#             # Normalize to dict (JsonOutputParser returns a Pydantic model instance in some LC versions)
#             if hasattr(plan, "model_dump"):
#                 plan = plan.model_dump()

#             # 2) apply memory updates
#             applied = {
#                 "glossary": self.memory.update_glossary(plan.get("glossary", [])),
#                 "kb_snippets": self.memory.add_kb_snippets(plan.get("kb_snippets", [])),
#                 "few_shots": self.memory.add_few_shots(plan.get("few_shots", [])),
#                 "rules": self.memory.update_rules(plan.get("rules", [])),
#             }

#             # 3) optional hot-reload of prompts (immediate effect in running app)
#             if getattr(self, "refresh_prompts_callback", None):
#                 try:
#                     self.refresh_prompts_callback()  # e.g., orchestrator re-builds agent chains from memory overlays
#                 except Exception as e:
#                     self.logger.warning(f"Prompt refresh failed: {e}")

#             # 4) append raw feedback to JSONL for audit
#             self._append_feedback_line(
#                 {
#                     "timestamp": datetime.now().isoformat() + "Z",
#                     "feature": feature,
#                     "screening": screening,
#                     "research_count": len(research),
#                     "decision": decision,
#                     "user_feedback": user_feedback,
#                     "plan_summary": plan.get("summary", ""),
#                     "plan_counts": {k: v.applied if hasattr(v, "applied") else v.get("applied")
#                                     for k, v in applied.items()},
#                 }
#             )

#             report = {
#                 "learning_summary": plan.get("summary", ""),
#                 "learning_counts": {k: v.applied if hasattr(v, "applied") else v.get("applied")
#                                     for k, v in applied.items()},
#                 "learning_files": {k: v.file if hasattr(v, "file") else v.get("file")
#                                    for k, v in applied.items()},
#                 "learning_timestamp": datetime.now().isoformat() + "Z",
#             }
#             self.log_interaction({"inputs": llm_input}, report)

#             # LangGraph state update
#             return {
#                 "learning_report": report,
#                 "workflow_completed": datetime.now().isoformat() + "Z",
#                 "next_step": "complete",
#             }

#         except Exception as e:
#             self.logger.error(f"LearningAgent failed: {e}")
#             return {
#                 "learning_report": {"error": str(e)},
#                 "next_step": "complete",
#             }

#     def _append_feedback_line(self, entry: Dict[str, Any]):
#         with open(self.feedback_file, "a", encoding="utf-8") as f:
#             f.write(json.dumps(entry) + "\n")

#     def _ensure_feedback_file(self):
#         """Create feedback file if it doesn't exist"""
#         os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
#         if not os.path.exists(self.feedback_file):
#             with open(self.feedback_file, "w", encoding="utf-8"):
#                 pass  # Create empty file

#     # ----- Optional additional feedback analytics helpers (kept from your draft) -----

#     def collect_feedback(self, analysis_result: Dict, user_feedback: Dict):
#         """Store feedback for future learning"""
#         feedback_entry = {
#             "timestamp": datetime.utcnow().isoformat() + "Z",
#             "analysis": analysis_result,
#             "user_correction": user_feedback.get("correction", ""),
#             "feedback_type": user_feedback.get("type", ""),  # "accurate", "incorrect", "needs_context"
#             "user_comments": user_feedback.get("comments", ""),
#             "session_id": analysis_result.get("session_id", ""),
#         }

#         self._store_feedback(feedback_entry)
#         self.logger.info(f"Feedback collected: {feedback_entry.get('feedback_type','')}")

#     def _store_feedback(self, feedback_entry: Dict):
#         """Store feedback entry to file"""
#         with open(self.feedback_file, "a", encoding="utf-8") as f:
#             f.write(json.dumps(feedback_entry) + "\n")

#     def analyze_feedback_patterns(self) -> Dict[str, Any]:
#         """Find common mistakes and improvement areas"""
#         feedback_data = self._load_all_feedback()

#         if not feedback_data:
#             return {"message": "No feedback data available"}

#         # Analyze patterns
#         feedback_types: Dict[str, int] = {}
#         common_errors: List[Dict[str, Any]] = []

#         for entry in feedback_data:
#             feedback_type = entry.get("feedback_type", "unknown")
#             feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1

#             if feedback_type == "incorrect":
#                 common_errors.append(
#                     {
#                         "original_analysis": entry.get("analysis", {}),
#                         "user_correction": entry.get("user_correction", ""),
#                         "timestamp": entry.get("timestamp", ""),
#                     }
#                 )

#         total = len(feedback_data)
#         accurate = feedback_types.get("accurate", 0)
#         accuracy_rate = (accurate / total) if total else 0.0

#         return {
#             "total_feedback": total,
#             "feedback_breakdown": feedback_types,
#             "accuracy_rate": accuracy_rate,
#             "common_errors": common_errors[:5],  # Top 5 errors
#             "improvement_suggestions": self._generate_improvements(common_errors),
#         }

#     def _load_all_feedback(self) -> List[Dict]:
#         """Load all feedback from file"""
#         feedback_data: List[Dict[str, Any]] = []
#         try:
#             with open(self.feedback_file, "r", encoding="utf-8") as f:
#                 for line in f:
#                     if line.strip():
#                         feedback_data.append(json.loads(line))
#         except FileNotFoundError:
#             pass
#         return feedback_data

#     def _generate_improvements(self, errors: List[Dict]) -> List[str]:
#         """Generate improvement suggestions from errors"""
#         suggestions: List[str] = []

#         if len(errors) > 0:
#             suggestions.append("Consider adding more examples to agent prompts.")
#             suggestions.append("Review geographic scope detection logic.")
#             suggestions.append("Enhance age-sensitivity keyword matching.")

#         return suggestions
