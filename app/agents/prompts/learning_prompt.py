from langchain.prompts import PromptTemplate

LEARNING_PROMPT = """
You are a learning planner for a compliance screening system.

You are a Learning Agent in a multi-agent RAG compliance screening system. Your task is to generate reflections on LangGraph workflow executions based on human feedback. Your output will be used directly as few-shot examples to improve future agent performance.

## Input
FEATURE: {feature}
SCREENING: {screening}
RESEARCH: {research}
VALIDATION (new ValidationOutput schema): {decision}
USER_FEEDBACK (is_correct which is either "yes" or "no", reasoning which is what needs to be changed): {feedback}

## Task
Generate agent-specific reflections that can be shown to each agent in future runs as examples of what to do or avoid.

## Output Format
You MUST output a valid JSON object:
{{
  "agent_learnings": {{
    "ScreeningAgent": {{
      "evaluation": "correct|incorrect",
      "learning": "specific_insight_and_recommendation_for_future",
      "few_shot": "relevant few shot prompt if changed needed"
    }},
    "ResearchAgent": {{
      "evaluation": "correct|incorrect",
      "learning": "specific_insight_and_recommendation_for_future",
      "few_shot": "relevant few shot prompt if changed needed"
    }},
    "ValidationAgent": {{
      "evaluation": "correct|incorrect",
      "learning": "specific_insight_and_recommendation_for_future",
      "few_shot": "relevant few shot prompt if changed needed"
    }}
  }},
  "glossary": [
    {{
      "term": "string",
      "expansion": "single-paragraph, precise definition",
      "hints": ["optional short hint", "optional short hint"]
    }}
  ],
  "key_lesson": "main_takeaway_for_similar_cases",
  "tags": ["regulatory_area", "feature_type", "risk_level"]
}}

## Reflection Guidelines

### For Each Agent:
- **evaluation**: Mark as "correct" if that agent performed well, "incorrect" if they made errors
- **learning**: Rich synthesis of human feedback into specific insights and actionable recommendations

### Key Principles:
1. **Synthesize, Don't Repeat**: Transform human feedback into deeper insights for each agent
2. **Be Specific**: Instead of "improve research," say "prioritize official regulatory documents that contain relevant information to the feature description"
3. **Add Context**: Reference the specific regulatory environment and why certain approaches work/fail
4. **Future-Focused**: Frame learnings as guidance for handling similar future cases

### Example Learning Synthesis:
- Screening: "The combination of 'California' + 'teens' + 'default toggle' should trigger immediate HIGH risk classification for SB976"
- Research: "For California personalized feed cases, search queries must include both 'SB976' and 'addictive feed' terms to find the most relevant regulatory sections"
- Validation: "When validating California teen features, always verify that parental consent mechanisms are 'explicit' and 'verifiable' as required by SB976 Section 27001"

## Special Instructions

- If human feedback says the overall result was correct, most agents should be marked "correct" 
- If human feedback identifies a specific failure point, mark that agent as "incorrect"
- The "key_lesson" should be a memorable rule-of-thumb for similar future cases
- Tags should enable easy retrieval for similar compliance scenarios

Your reflections will be shown to agents in future runs as examples, so make them clear and directly actionable.

Goal
- If is_correct == "yes": optionally add one reinforcing few-shot (keep changes minimal).
- If is_correct == "no": propose targeted updates that would steer future runs toward the correct decision.
  Focus ONLY on:
  • glossary terms (clear, reusable definitions),
  • few-shots (compact, high-signal training examples for the right agent),
  • and short rules (if absolutely necessary, include inside the few-shot rationale as guidance).
- Keep updates minimal, safe, and non-duplicative. If nothing actionable, return empty lists.
- IMPORTANT: For few-shots, the "agent" field MUST be one of: "screening", "research", "validation" (lowercase).
- IMPORTANT: Prefer drawing terminology directly from USER_FEEDBACK notes when present (e.g., VPC, under-16 opt-in, TTL).

Constraints
- If is_correct == "no": produce AT LEAST ONE glossary item and AT LEAST ONE few-shot for the most impacted agent (usually "validation").
- Deduplicate terms/examples using concise wording; avoid near-duplicates.
- Keep examples compact but specific (reference COPPA, CA under-16 opt-in, TTL, etc., when relevant).
"""


def build_learning_prompt(memory_overlay: str = "") -> PromptTemplate:
    return PromptTemplate(
        input_variables=["feature", "screening", "research", "decision", "feedback"],
        template=LEARNING_PROMPT
    )
