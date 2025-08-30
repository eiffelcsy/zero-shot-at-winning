#!/usr/bin/env python3
"""
Combined memory system that integrates TikTok terminology and few-shot examples.
Each agent gets its own combined overlay with only its relevant few-shot examples.
"""

from pathlib import Path
from typing import Dict, Any
from .tiktok_memory import TikTokMemory, get_tiktok_memory
from .fewshot_memory import FewShotMemory, get_fewshot_memory


def get_combined_agent_overlay(agent_type: str) -> str:
    """
    Get a combined memory overlay for a specific agent that includes both
    TikTok terminology and only that agent's few-shot examples.
    
    Args:
        agent_type: Type of agent (screening, research, validation)
        
    Returns:
        Combined memory overlay string for the specific agent
    """
    try:
        # Get TikTok memory
        tiktok_memory = get_tiktok_memory()
        if not tiktok_memory:
            tiktok_section = "TikTok terminology not available."
        else:
            tiktok_section = tiktok_memory.get_terminology_summary()
        
        # Get few-shot memory
        fewshot_memory = get_fewshot_memory()
        if not fewshot_memory:
            fewshot_section = f"No few-shot examples available for {agent_type} agent."
        else:
            fewshot_section = fewshot_memory.get_examples_reference(agent_type)
        
        # Build combined overlay for the specific agent
        overlay = f"""MEMORY OVERLAY FOR {agent_type.upper()} AGENT
PRIORITY INSTRUCTIONS:

PRIMARY FOCUS: Reference the few-shot examples below for task execution patterns
CONDITIONAL REFERENCE: Use TikTok terminology only when input contains platform-specific terms

{fewshot_section}
{tiktok_section}
AGENT OPERATION GUIDELINES:

Agent Type: {agent_type}
Primary Memory: FewShotMemory (LangChain SimpleMemory) - USE THESE EXAMPLES FIRST
Secondary Memory: TikTokMemory - Apply when detecting TikTok terminology in input
Memory Architecture: Multi-Agent RAG System Integration
Example Priority: Agent-specific few-shot examples take precedence
Terminology Activation: TikTok terms triggered by context detection

EXECUTION STRATEGY:

Analyze input for task patterns matching few-shot examples
Apply {agent_type}-specific approaches from examples above
If TikTok terminology detected in input, reference platform definitions
Maintain consistency with demonstrated few-shot patterns
Integrate seamlessly within multi-agent RAG workflow
"""
        
        return overlay
        
    except Exception:
        # Return minimal fallback overlay
        return f"""MEMORY OVERLAY FOR {agent_type.upper()} AGENT

Memory systems temporarily unavailable.

AGENT CONTEXT:
- Agent Type: {agent_type}
- Status: Fallback mode
"""


def get_all_combined_overlays() -> Dict[str, str]:
    """
    Get combined memory overlays for all agent types.
    Each agent gets only its specific few-shot examples.
    
    Returns:
        Dictionary mapping agent type to combined overlay
    """
    agent_types = ["screening", "research", "validation"]
    
    overlays = {}
    for agent_type in agent_types:
        overlays[agent_type] = get_combined_agent_overlay(agent_type)
    
    return overlays


def initialize_combined_memory_system():
    """
    Initialize the combined memory system with both TikTok terminology
    and agent-specific few-shot examples.
    
    Returns:
        Dictionary containing initialized components and overlays
    """
    try:
        # Initialize both memory systems
        tiktok_memory = get_tiktok_memory()
        fewshot_memory = get_fewshot_memory()
        
        # Generate combined overlays for each agent
        overlays = get_all_combined_overlays()
        
        return {
            "tiktok_memory": tiktok_memory,
            "fewshot_memory": fewshot_memory,
            "combined_overlays": overlays,
            "screening_overlay": overlays["screening"],
            "research_overlay": overlays["research"], 
            "validation_overlay": overlays["validation"]
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize combined memory system: {e}")


# Convenience functions for easy agent integration
def get_screening_overlay() -> str:
    """Get combined memory overlay for screening agent."""
    return get_combined_agent_overlay("screening")


def get_research_overlay() -> str:
    """Get combined memory overlay for research agent."""
    return get_combined_agent_overlay("research")


def get_validation_overlay() -> str:
    """Get combined memory overlay for validation agent."""
    return get_combined_agent_overlay("validation")


# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================

if __name__ == "__main__":
    try:
        print("=== Combined Memory System Test ===")
        print("Initializing combined memory system...")
        
        result = initialize_combined_memory_system()
        
        print("\n=== SYSTEM READY ===")
        print("Combined memory overlays generated:")
        
        for agent_type in ["screening", "research", "validation"]:
            overlay = result["combined_overlays"][agent_type]
            print(f"\n{agent_type.title()} Agent:")
            print(f"  - Overlay length: {len(overlay)} characters")
            print(f"  - Has TikTok terminology: {'TIKTOK TERMINOLOGY REFERENCE' in overlay}")
            print(f"  - Has few-shot examples: {'FEW-SHOT EXAMPLES' in overlay}")
            
            # Check for agent-specific content
            examples_section = f"FEW-SHOT EXAMPLES FOR {agent_type.upper()} AGENT"
            print(f"  - Agent-specific examples: {examples_section in overlay}")
            
            # Check for other agents' examples (should not be present)
            other_agents = [a for a in ["screening", "research", "validation"] if a != agent_type]
            for other_agent in other_agents:
                other_examples = f"FEW-SHOT EXAMPLES FOR {other_agent.upper()} AGENT"
                if other_examples in overlay:
                    print(f"  - WARNING: Contains {other_agent} examples (should not!)")
        
        print("\n=== Usage Instructions ===")
        print("Import and use in your agents:")
        print("from app.agents.memory.combined_memory import get_screening_overlay, get_research_overlay, get_validation_overlay")
        print("screening_overlay = get_screening_overlay()")
        print("research_overlay = get_research_overlay()")
        print("validation_overlay = get_validation_overlay()")
        
        print("\nEach agent now has access to:")
        print("1. TikTok terminology (shared)")
        print("2. Agent-specific few-shot examples only")
        
    except Exception as e:
        print(f"System test failed: {e}")
        import sys
        sys.exit(1)
