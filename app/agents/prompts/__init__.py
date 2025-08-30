from .screening_prompt import SCREENING_PROMPT
from .research_prompt import RESEARCH_PROMPT, SEARCH_QUERY_GENERATION, build_search_query_prompt
from .validation_prompt import VALIDATION_PROMPT
from .learning_prompt import LEARNING_PROMPT, build_learning_prompt

def build_screening_prompt(memory_overlay: str = "") -> str:
    """Build screening prompt with memory overlay"""
    if memory_overlay:
        return f"{memory_overlay}\n\n{SCREENING_PROMPT}"
    return SCREENING_PROMPT

def build_research_prompt(memory_overlay: str = "") -> str:
    """Build research prompt with memory overlay"""
    if memory_overlay:
        return f"{memory_overlay}\n\n{RESEARCH_PROMPT}"
    return RESEARCH_PROMPT

def build_validation_prompt(memory_overlay: str = "") -> str:
    """Build validation prompt with memory overlay"""
    if memory_overlay:
        return f"{memory_overlay}\n\n{VALIDATION_PROMPT}"
    return VALIDATION_PROMPT

__all__ = [
    "build_screening_prompt",
    "build_research_prompt", 
    "build_validation_prompt",
    "build_search_query_prompt",
    "build_learning_prompt",
    "SCREENING_PROMPT",
    "RESEARCH_PROMPT",
    "SEARCH_QUERY_GENERATION",
    "VALIDATION_PROMPT",
    "LEARNING_PROMPT"
]