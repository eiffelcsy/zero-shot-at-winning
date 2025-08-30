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

from logs.logging_config import setup_logging, get_logger, ensure_logging_setup, get_current_log_file

# Ensure logging is set up before creating the logger
ensure_logging_setup()
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
        
        # Log the current log file being used
        current_log_file = get_current_log_file()
        logger.info(f"TikTokMemory initialized with {len(self._memories)} terminology entries")
        logger.info(f"Logging to file: {current_log_file}")
    
    def _load_terminology(self) -> Dict[str, Any]:
        """
        Load TikTok terminology from JSON file, with fallback to embedded data.
        
        Returns:
            Dictionary containing terminology data
        """
        try:
            logger.info(f"Attempting to load terminology from: {self._terminology_file}")
            
            if not self._terminology_file.exists():
                logger.warning(f"Terminology file not found: {self._terminology_file}")
                logger.info("Falling back to embedded terminology data")
                return self._get_embedded_terminology()
            
            # Try to read the JSON file
            with open(self._terminology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded {len(data.get('terminology', []))} terminology entries from JSON file")
            
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
            
            logger.info(f"JSON file terminology loaded successfully")
            return terminology_dict
            
        except PermissionError as e:
            logger.warning(f"Permission denied accessing terminology file: {e}")
            logger.info("This is likely a Docker container permission issue. Using embedded terminology as fallback.")
            return self._get_embedded_terminology()
        except FileNotFoundError as e:
            logger.warning(f"Terminology file not found: {e}")
            logger.info("Using embedded terminology as fallback.")
            return self._get_embedded_terminology()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in terminology file: {e}")
            logger.info("Using embedded terminology as fallback.")
            return self._get_embedded_terminology()
        except Exception as e:
            logger.error(f"Unexpected error loading terminology: {e}")
            logger.info("Using embedded terminology as fallback.")
            return self._get_embedded_terminology()
    
    def _get_embedded_terminology(self) -> Dict[str, Any]:
        """
        Get embedded terminology data as a fallback when JSON file cannot be accessed.
        
        Returns:
            Dictionary containing embedded terminology data
        """
        logger.info("Loading embedded TikTok terminology data as fallback")
        
        # Embedded terminology data - always available
        embedded_data = [
            {"acronym": "NR", "meaning": "Not recommended (restriction/limitation level)", "category": "restriction"},
            {"acronym": "PF", "meaning": "Personalized feed (recommendation algorithm)", "category": "algorithm"},
            {"acronym": "GH", "meaning": "Geo-handler (region-based routing and enforcement)", "category": "geographic"},
            {"acronym": "CDS", "meaning": "Compliance Detection System (automated compliance monitoring)", "category": "compliance"},
            {"acronym": "DRT", "meaning": "Data retention threshold (how long data can be stored)", "category": "data"},
            {"acronym": "LCP", "meaning": "Local compliance policy (region-specific rules)", "category": "compliance"},
            {"acronym": "Redline", "meaning": "Flag for legal review (not financial loss context)", "category": "legal"},
            {"acronym": "Softblock", "meaning": "Silent user limitation without notifications", "category": "user_management"},
            {"acronym": "Spanner", "meaning": "Rule engine (not Google Spanner database)", "category": "system"},
            {"acronym": "ShadowMode", "meaning": "Deploy feature without user impact for analytics collection", "category": "deployment"},
            {"acronym": "T5", "meaning": "Tier 5 data (highest sensitivity level - more critical than T1-T4)", "category": "data_sensitivity"},
            {"acronym": "ASL", "meaning": "Age-sensitive logic (age verification/restrictions for minors)", "category": "age_verification"},
            {"acronym": "Glow", "meaning": "Compliance-flagging status for geo-based alerts", "category": "compliance"},
            {"acronym": "NSP", "meaning": "Non-shareable policy (content sharing restrictions)", "category": "content"},
            {"acronym": "Jellybean", "meaning": "Internal parental control system", "category": "parental_controls"},
            {"acronym": "EchoTrace", "meaning": "Log tracing mode for compliance verification", "category": "logging"},
            {"acronym": "BB", "meaning": "Baseline Behavior (standard user behavior for anomaly detection)", "category": "behavior"},
            {"acronym": "Snowcap", "meaning": "Child safety policy framework", "category": "child_safety"},
            {"acronym": "FR", "meaning": "Feature rollout status", "category": "deployment"},
            {"acronym": "IMT", "meaning": "Internal monitoring trigger", "category": "monitoring"}
        ]
        
        # Convert terminology to a more usable format
        terminology_dict = {}
        for item in embedded_data:
            acronym = item.get('acronym', '')
            if acronym:
                terminology_dict[f"tiktok_{acronym.lower()}"] = {
                    "acronym": acronym,
                    "meaning": item.get('meaning', ''),
                    "category": item.get('category', '')
                }
        
        # Add a comprehensive terminology reference
        terminology_dict["tiktok_terminology_reference"] = self._build_terminology_reference(embedded_data)
        
        logger.info(f"Successfully loaded {len(embedded_data)} embedded terminology entries as fallback")
        return terminology_dict
    
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
    
    # Ensure logging is set up and get current log file info
    ensure_logging_setup()
    current_log_file = get_current_log_file()
    
    # Set up logging with enhanced configuration
    logger = setup_logging(log_level="INFO")
    logger.info("=== Initializing TikTok Compliance System ===")
    logger.info(f"Logging to file: {current_log_file}")
    
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
        
        # 7. Log final status
        logger.info("=== SYSTEM INITIALIZATION COMPLETE ===")
        logger.info(f"All logs are being saved to: {current_log_file}")
        logger.info("TikTok terminology context is available to all agents")
        
        return {
            "tiktok_memory": tiktok_memory,
            "screening_overlay": screening_overlay,
            "research_overlay": research_overlay,
            "validation_overlay": validation_overlay,
            "log_file": current_log_file
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
    logger.info(f"Building agent overlay for {agent_type} agent")
    
    # Get the terminology reference
    terminology_ref = tiktok_memory.get_terminology_summary()
    logger.info(f"Terminology reference length: {len(terminology_ref)} characters")
    logger.info(f"Terminology reference contains 'TIKTOK TERMINOLOGY REFERENCE': {'TIKTOK TERMINOLOGY REFERENCE' in terminology_ref}")
    
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
    
    logger.info(f"Generated overlay for {agent_type} agent: {len(overlay)} characters")
    logger.info(f"Overlay contains 'TIKTOK TERMINOLOGY REFERENCE': {'TIKTOK TERMINOLOGY REFERENCE' in overlay}")
    
    return overlay

def get_agent_overlays():
    """Get the memory overlays for each agent type."""
    try:
        logger.info("=== Getting Agent Overlays ===")
        
        # Ensure logging is set up
        ensure_logging_setup()
        
        tiktok_memory = TikTokMemory()
        logger.info(f"TikTokMemory created with {len(tiktok_memory.memory_variables)} variables")
        
        # Check if TikTok terminology reference is present
        if "tiktok_terminology_reference" in tiktok_memory.memory_variables:
            logger.info("TikTok terminology reference found in memory variables")
            ref_content = tiktok_memory.get_terminology_summary()
            logger.info(f"Reference content length: {len(ref_content)} characters")
            logger.info(f"Reference contains 'TIKTOK TERMINOLOGY REFERENCE': {'TIKTOK TERMINOLOGY REFERENCE' in ref_content}")
        else:
            logger.error("TikTok terminology reference NOT found in memory variables")
            logger.error(f"Available variables: {list(tiktok_memory.memory_variables.keys())}")
        
        overlays = {
            "screening": _build_agent_overlay(tiktok_memory, "screening"),
            "research": _build_agent_overlay(tiktok_memory, "research"),
            "validation": _build_agent_overlay(tiktok_memory, "validation")
        }
        
        # Log the overlay generation
        logger.info(f"Generated {len(overlays)} agent overlays")
        for agent_type, overlay in overlays.items():
            logger.info(f"  - {agent_type}: {len(overlay)} characters")
            logger.info(f"    Contains 'TIKTOK TERMINOLOGY REFERENCE': {'TIKTOK TERMINOLOGY REFERENCE' in overlay}")
        
        return overlays
        
    except Exception as e:
        logger.error(f"Failed to get agent overlays: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

def get_tiktok_memory():
    """Get a TikTokMemory instance."""
    try:
        # Ensure logging is set up
        ensure_logging_setup()
        
        memory = TikTokMemory()
        logger = get_logger(__name__)
        logger.info("TikTokMemory instance created successfully")
        
        return memory
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to get TikTok memory: {e}")
        return None

# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================

if __name__ == "__main__":
    try:
        print("=== TikTok Compliance System Initialization ===")
        print("Setting up logging and memory system...")
        
        result = initialize_tiktok_system()
        
        print("\n=== SYSTEM READY ===")
        print(f"Log file: {result.get('log_file', 'Unknown')}")
        print("To use in your application:")
        print("1. Import: from app.agents.memory.tiktok_memory import TikTokMemory, get_agent_overlays, get_tiktok_memory")
        print("2. Get overlays: overlays = get_agent_overlays()")
        print("3. Get memory: memory = get_tiktok_memory()")
        print("4. Initialize orchestrator: orchestrator = ComplianceOrchestrator(memory=memory)")
        print("\nAll agents will now have access to TikTok terminology context!")
        print("All application logs are being saved to log files.")
        
    except Exception as e:
        print(f"System initialization failed: {e}")
        sys.exit(1)
