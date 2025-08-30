#!/usr/bin/env python3
"""
TikTok-specific memory implementation using LangChain SimpleMemory.
This file contains both the TikTokMemory class and system initialization utilities.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.memory import BaseMemory
from typing_extensions import override

from app.logs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

class TikTokMemory(BaseMemory):
    """
    Simple memory for storing TikTok terminology and context that shouldn't
    ever change between prompts.
    """

    def __init__(self, terminology_file: str = None):
        """
        Initialize TikTok memory with terminology.
        
        Args:
            terminology_file: Path to terminology JSON file. If None, uses default.
        """
        super().__init__()
        
        if terminology_file is None:
            # Default to the terminology file in the data directory
            app_dir = Path(__file__).parent.parent.parent
            terminology_file = app_dir / "data" / "tiktok_terminology" / "terminology.json"
        
        # Store the terminology file path and load memories
        self._terminology_file = Path(terminology_file)
        self._memories = self._load_terminology()
        
        logger.info(f"TikTokMemory initialized with {len(self._memories)} terminology entries")
    
    def _load_terminology(self) -> Dict[str, Any]:
        """
        Load TikTok terminology from JSON file.
        
        Returns:
            Dictionary containing terminology data
        """
        try:
            if not self._terminology_file.exists():
                logger.warning(f"Terminology file not found: {self._terminology_file}")
                return {}
            
            with open(self._terminology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert terminology to a more usable format
            terminology_dict = {}
            for item in data.get('terminology', []):
                acronym = item.get('acronym', '')
                if acronym:
                    terminology_dict[f"tiktok_{acronym.lower()}"] = {
                        "acronym": acronym,
                        "meaning": item.get('meaning', ''),
                        "category": item.get('category', '')
                    }
            
            # Add a comprehensive terminology reference
            terminology_dict["tiktok_terminology_reference"] = self._build_terminology_reference(data.get('terminology', []))
            
            logger.info(f"Successfully loaded {len(data.get('terminology', []))} terminology entries")
            return terminology_dict
            
        except Exception as e:
            logger.error(f"Failed to load terminology: {e}")
            return {}
    
    def _build_terminology_reference(self, terminology: List[Dict[str, Any]]) -> str:
        """
        Build a comprehensive terminology reference string.
        
        Args:
            terminology: List of terminology items
            
        Returns:
            Formatted terminology reference string
        """
        if not terminology:
            return "No TikTok terminology available."
        
        reference = "TIKTOK TERMINOLOGY REFERENCE:\n\n"
        
        # Group by category
        categories = {}
        for item in terminology:
            category = item.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        
        # Build reference by category
        for category, items in categories.items():
            reference += f"{category.upper().replace('_', ' ')}:\n"
            for item in items:
                reference += f"  {item['acronym']}: {item['meaning']}\n"
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
        logger.debug("TikTokMemory: No context to save - memory is immutable")
        pass
    
    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        logger.debug("TikTokMemory: No memory to clear - memory is immutable")
        pass
    
    def get_terminology_summary(self) -> str:
        """
        Get a summary of all loaded terminology.
        
        Returns:
            Summary string of terminology
        """
        if "tiktok_terminology_reference" in self._memories:
            return self._memories["tiktok_terminology_reference"]
        return "No TikTok terminology available."
    
    def get_acronym_meaning(self, acronym: str) -> str:
        """
        Get the meaning of a specific acronym.
        
        Args:
            acronym: The acronym to look up
            
        Returns:
            Meaning of the acronym or "Not found" if not available
        """
        key = f"tiktok_{acronym.lower()}"
        if key in self._memories:
            return self._memories[key]["meaning"]
        return f"Acronym '{acronym}' not found in TikTok terminology"
    
    def get_category_terms(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all terms in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of terms in the category
        """
        terms = []
        for key, value in self._memories.items():
            if key.startswith("tiktok_") and key != "tiktok_terminology_reference":
                if value.get("category") == category:
                    terms.append(value)
        return terms

# =============================================================================
# SYSTEM INITIALIZATION AND UTILITY FUNCTIONS
# =============================================================================

def initialize_tiktok_system():
    """Initialize the complete TikTok compliance system with proper memory overlays."""
    
    # Set up logging
    logger = setup_logging(log_level="INFO")
    logger.info("=== Initializing TikTok Compliance System ===")
    
    try:
        # 1. Initialize TikTok memory
        tiktok_memory = TikTokMemory()
        logger.info("TikTok memory initialized successfully")
        
        # 2. Verify terminology loading
        terminology_count = len([k for k in tiktok_memory.memory_variables if k.startswith("tiktok_") and k != "tiktok_terminology_reference"])
        logger.info(f"Loaded {terminology_count} TikTok terminology entries")
        
        # 3. Generate memory overlays for each agent
        screening_overlay = _build_agent_overlay(tiktok_memory, "screening")
        research_overlay = _build_agent_overlay(tiktok_memory, "research")
        validation_overlay = _build_agent_overlay(tiktok_memory, "validation")
        
        logger.info("Generated memory overlays:")
        logger.info(f"  - Screening: {len(screening_overlay)} characters")
        logger.info(f"  - Research: {len(research_overlay)} characters")
        logger.info(f"  - Validation: {len(validation_overlay)} characters")
        
        # 4. Verify terminology is included in overlays
        logger.info("=== VERIFICATION ===")
        
        # Check screening overlay
        if "TIKTOK TERMINOLOGY REFERENCE" in screening_overlay:
            logger.info("Screening overlay contains TikTok terminology")
        else:
            logger.warning("Screening overlay missing TikTok terminology")
        
        # Check research overlay  
        if "TIKTOK TERMINOLOGY REFERENCE" in research_overlay:
            logger.info("Research overlay contains TikTok terminology")
        else:
            logger.warning("Research overlay missing TikTok terminology")
        
        # Check validation overlay
        if "TIKTOK TERMINOLOGY REFERENCE" in validation_overlay:
            logger.info("Validation overlay contains TikTok terminology")
        else:
            logger.warning("Validation overlay missing TikTok terminology")
        
        # 5. Test prompt building
        logger.info("=== PROMPT BUILDING TEST ===")
        
        try:
            # Import prompt builders
            from agents.prompts.screening_prompt import build_screening_prompt
            from agents.prompts.research_prompt import build_research_prompt
            from agents.prompts.validation_prompt import build_validation_prompt
            
            screening_prompt = build_screening_prompt(screening_overlay)
            logger.info("Screening prompt built successfully")
            
            research_prompt = build_research_prompt(research_overlay)
            logger.info("Research prompt built successfully")
            
            validation_prompt = build_validation_prompt(validation_overlay)
            logger.info("Validation prompt built successfully")
            
        except ImportError as e:
            logger.warning(f"Some prompt modules not available: {e}")
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
        
        # 6. Show overlay content preview
        logger.info("=== OVERLAY CONTENT PREVIEW ===")
        logger.info("Screening overlay preview:")
        preview = screening_overlay[:300] + "..." if len(screening_overlay) > 300 else screening_overlay
        logger.info(preview)
        
        return {
            "tiktok_memory": tiktok_memory,
            "screening_overlay": screening_overlay,
            "research_overlay": research_overlay,
            "validation_overlay": validation_overlay
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize TikTok system: {e}")
        raise

def _build_agent_overlay(tiktok_memory: TikTokMemory, agent_type: str) -> str:
    """
    Build a memory overlay for a specific agent type.
    
    Args:
        tiktok_memory: TikTokMemory instance
        agent_type: Type of agent (screening, research, validation)
        
    Returns:
        Formatted memory overlay string
    """
    # Get the terminology reference
    terminology_ref = tiktok_memory.get_terminology_summary()
    
    # Build agent-specific overlay
    overlay = f"""MEMORY OVERLAY FOR {agent_type.upper()} AGENT

{terminology_ref}

AGENT CONTEXT:
- Agent Type: {agent_type}
- Memory System: TikTokMemory (LangChain SimpleMemory)
- Terminology Source: JSON file
- Memory Persistence: Immutable (set in stone)

This overlay provides the {agent_type} agent with access to all TikTok terminology
and compliance context needed for proper operation.
"""
    
    return overlay

def get_agent_overlays():
    """Get the memory overlays for each agent type."""
    try:
        tiktok_memory = TikTokMemory()
        
        return {
            "screening": _build_agent_overlay(tiktok_memory, "screening"),
            "research": _build_agent_overlay(tiktok_memory, "research"),
            "validation": _build_agent_overlay(tiktok_memory, "validation")
        }
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to get agent overlays: {e}")
        return {}

def get_tiktok_memory():
    """Get a TikTokMemory instance."""
    try:
        return TikTokMemory()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to get TikTok memory: {e}")
        return None

# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================

if __name__ == "__main__":
    try:
        result = initialize_tiktok_system()
        
        print("\n=== SYSTEM READY ===")
        print("To use in your application:")
        print("1. Import: from app.agents.memory.tiktok_memory import TikTokMemory, get_agent_overlays, get_tiktok_memory")
        print("2. Get overlays: overlays = get_agent_overlays()")
        print("3. Get memory: memory = get_tiktok_memory()")
        print("4. Initialize orchestrator: orchestrator = ComplianceOrchestrator(memory=memory)")
        print("\nAll agents will now have access to TikTok terminology context!")
        
    except Exception as e:
        print(f"System initialization failed: {e}")
        sys.exit(1)
