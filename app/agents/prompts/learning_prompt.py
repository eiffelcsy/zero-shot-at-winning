from langchain.prompts import PromptTemplate

LEARNING_PROMPT = """
You are a learning planner for a compliance screening system.
Inputs:
FEATURE: {feature}
SCREENING: {screening}
RESEARCH: {research}
VALIDATION: {decision}
USER_FEEDBACK (is_correct=yes|no, notes): {feedback}
Goal: Propose precise updates so future runs reach the correct decision.
- If is_correct=='yes': optionally add one reinforcing few-shot.
- If is_correct=='no': propose targeted updates: glossary terms, short rules,
    few-shots, and KB snippets (with URLs and excerpts) that support the corrected outcome.
- Keep updates minimal, safe, and non-duplicative. If nothing actionable, output empty lists.
Return ONLY valid JSON with keys: agent, summary, glossary, rules, few_shots, kb_snippets.
"""

def build_learning_prompt(memory_overlay: str = "") -> PromptTemplate:
    return PromptTemplate(
        input_variables=["feature", "screening", "research", "decision", "feedback"],
        template=LEARNING_PROMPT
    )
