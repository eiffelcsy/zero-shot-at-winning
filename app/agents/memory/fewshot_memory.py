#!/usr/bin/env python3
"""
Few-shot memory implementation using LangChain SimpleMemory.
Loads few-shot examples for each agent type and provides them as memory overlays.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.memory import BaseMemory
from typing_extensions import override


class FewShotMemory(BaseMemory):
    """
    Simple memory for storing few-shot examples that shouldn't
    ever change between prompts.
    """

    def __init__(self, few_shots_dir: str = None):
        """
        Initialize few-shot memory with examples from JSONL files.
        
        Args:
            few_shots_dir: Path to few-shots directory. If None, uses default.
        """
        super().__init__()
        
        if few_shots_dir is None:
            # Default to the few_shots directory in the data directory
            app_dir = Path(__file__).parent.parent.parent
            few_shots_dir = app_dir / "data" / "memory" / "few_shots"
        
        # Store the few_shots directory path and load memories
        self._few_shots_dir = Path(few_shots_dir)
        self._memories = self._load_few_shots()
    
    def _load_few_shots(self) -> Dict[str, Any]:
        """
        Load few-shot examples from JSONL files.
        
        Returns:
            Dictionary containing few-shot data organized by agent type
        """
        few_shots_dict = {}
        
        # Define the agent types and their corresponding files
        agent_types = ["screening", "research", "validation"]
        
        for agent_type in agent_types:
            file_path = self._few_shots_dir / f"{agent_type}.jsonl"
            examples = self._load_jsonl_file(file_path)
            
            # Store individual examples
            few_shots_dict[f"fewshot_{agent_type}_examples"] = examples
            
            # Store formatted reference for the agent
            few_shots_dict[f"fewshot_{agent_type}_reference"] = self._build_examples_reference(examples, agent_type)
        
        return few_shots_dict
    
    def _load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load examples from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of example dictionaries
        """
        examples = []
        
        try:
            if not file_path.exists():
                return examples
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            example = json.loads(line)
                            examples.append(example)
                        except json.JSONDecodeError:
                            continue  # Skip invalid JSON lines
                            
        except Exception:
            pass  # Return empty list on any error
            
        return examples
    
    def _build_examples_reference(self, examples: List[Dict[str, Any]], agent_type: str) -> str:
        """
        Build a formatted reference string for few-shot examples.
        
        Args:
            examples: List of example dictionaries
            agent_type: Type of agent (screening, research, validation)
            
        Returns:
            Formatted examples reference string
        """
        if not examples:
            return f"No few-shot examples available for {agent_type} agent."
        
        reference = f"FEW-SHOT EXAMPLES FOR {agent_type.upper()} AGENT:\n\n"
        
        for i, example in enumerate(examples, 1):
            reference += f"Example {i}:\n"
            
            # Handle different possible fields in the example
            if "prompt" in example:
                reference += f"  Prompt: {example['prompt']}\n"
            if "input" in example:
                reference += f"  Input: {example['input']}\n"
            if "output" in example:
                reference += f"  Output: {example['output']}\n"
            if "reasoning" in example:
                reference += f"  Reasoning: {example['reasoning']}\n"
            if "context" in example:
                reference += f"  Context: {example['context']}\n"
            
            # Add any other fields that might be present
            for key, value in example.items():
                if key not in ["prompt", "input", "output", "reasoning", "context", "agent"]:
                    reference += f"  {key.title()}: {value}\n"
            
            reference += "\n"
        
        return reference.strip()
    
    @property
    @override
    def memory_variables(self) -> List[str]:
        """Return list of memory variable names."""
        return list(self._memories.keys())
    
    @override
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Load memory variables for the given inputs.
        
        Args:
            inputs: Input dictionary (unused in this implementation)
            
        Returns:
            Dictionary of memory variables
        """
        # Convert all memory values to strings for compatibility
        return {key: str(value) for key, value in self._memories.items()}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Nothing should be saved or changed, my memory is set in stone.
        
        Args:
            inputs: Input dictionary (unused)
            outputs: Output dictionary (unused)
        """
        pass
    
    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
    
    def get_examples_for_agent(self, agent_type: str) -> List[Dict[str, Any]]:
        """
        Get few-shot examples for a specific agent type.
        
        Args:
            agent_type: Type of agent (screening, research, validation)
            
        Returns:
            List of examples for the agent
        """
        key = f"fewshot_{agent_type}_examples"
        if key in self._memories:
            return self._memories[key]
        return []
    
    def get_examples_reference(self, agent_type: str) -> str:
        """
        Get formatted examples reference for a specific agent type.
        
        Args:
            agent_type: Type of agent (screening, research, validation)
            
        Returns:
            Formatted examples reference string
        """
        key = f"fewshot_{agent_type}_reference"
        if key in self._memories:
            return self._memories[key]
        return f"No few-shot examples available for {agent_type} agent."


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _build_agent_overlay(fewshot_memory: FewShotMemory, agent_type: str) -> str:
    """
    Build a memory overlay for a specific agent type.
    
    Args:
        fewshot_memory: FewShotMemory instance
        agent_type: Type of agent (screening, research, validation)
        
    Returns:
        Formatted memory overlay string
    """
    # Get the examples reference
    examples_ref = fewshot_memory.get_examples_reference(agent_type)
    
    # Build agent-specific overlay
    overlay = f"""MEMORY OVERLAY FOR {agent_type.upper()} AGENT

{examples_ref}

AGENT CONTEXT:
- Agent Type: {agent_type}
- Memory System: FewShotMemory (LangChain SimpleMemory)
- Examples Source: JSONL files
- Memory Persistence: Immutable (set in stone)

This overlay provides the {agent_type} agent with access to all few-shot examples
needed for proper operation and learning from past successful interactions.
"""
    
    return overlay


def get_agent_overlays():
    """Get the memory overlays for each agent type."""
    try:
        fewshot_memory = FewShotMemory()
        
        overlays = {
            "screening": _build_agent_overlay(fewshot_memory, "screening"),
            "research": _build_agent_overlay(fewshot_memory, "research"),
            "validation": _build_agent_overlay(fewshot_memory, "validation")
        }
        
        return overlays
        
    except Exception:
        return {}


def get_fewshot_memory():
    """Get a FewShotMemory instance."""
    try:
        memory = FewShotMemory()
        return memory
    except Exception:
        return None


def initialize_fewshot_system():
    """Initialize the few-shot memory system."""
    try:
        # Initialize few-shot memory
        fewshot_memory = FewShotMemory()
        
        # Generate memory overlays for each agent
        screening_overlay = _build_agent_overlay(fewshot_memory, "screening")
        research_overlay = _build_agent_overlay(fewshot_memory, "research")
        validation_overlay = _build_agent_overlay(fewshot_memory, "validation")
        
        return {
            "fewshot_memory": fewshot_memory,
            "screening_overlay": screening_overlay,
            "research_overlay": research_overlay,
            "validation_overlay": validation_overlay
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize few-shot system: {e}")


# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================

if __name__ == "__main__":
    try:
        print("=== Few-Shot Memory System Initialization ===")
        print("Setting up few-shot memory system...")
        
        result = initialize_fewshot_system()
        
        print("\n=== SYSTEM READY ===")
        print("To use in your application:")
        print("1. Import: from app.agents.memory.fewshot_memory import FewShotMemory, get_agent_overlays, get_fewshot_memory")
        print("2. Get overlays: overlays = get_agent_overlays()")
        print("3. Get memory: memory = get_fewshot_memory()")
        print("4. Initialize orchestrator with few-shot context")
        print("\nAll agents will now have access to few-shot examples!")
        
        # Show some stats
        memory = result["fewshot_memory"]
        for agent_type in ["screening", "research", "validation"]:
            examples = memory.get_examples_for_agent(agent_type)
            print(f"{agent_type.title()} agent: {len(examples)} examples loaded")
        
    except Exception as e:
        print(f"System initialization failed: {e}")
        import sys
        sys.exit(1)
