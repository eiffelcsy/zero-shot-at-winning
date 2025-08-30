#!/usr/bin/env python3
"""
System initialization script to properly set up TikTok terminology and memory overlays.
This ensures all agents have access to the TikTok context they need.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any

# Add the app directory to Python path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any
from langchain.memory import SimpleMemory
from langchain_core.memory import BaseMemory
from typing_extensions import override
from ..prompts import build_screening_prompt, build_research_prompt, build_validation_prompt

# Configure logging
log_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "tiktok_system.log"
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    ]
)

# Get logger
logger = logging.getLogger(__name__)

class SimpleMemory(BaseMemory):
    """Simple memory for storing context or other information that shouldn't
    ever change between prompts.
    """

    memories: dict[str, Any] = {}

    @property
    @override
    def memory_variables(self) -> list[str]:
        return list(self.memories.keys())

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return self.memories

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Nothing should be saved or changed, my memory is set in stone."""

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""


class TikTokMemoryStore:
    """Simple memory store using LangChain's SimpleMemory for TikTok terminology and overlays."""
    
    def __init__(self):
        self.memory = SimpleMemory()
        self.terminology: List[Dict[str, str]] = []
        self.overlays: Dict[str, str] = {}
        
    def load_terminology_from_file(self, file_path: str = None) -> bool:
        """Load TikTok terminology from JSON file."""
        if file_path is None:
            # Default path relative to app directory
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "tiktok_terminology.json"
            )
        
        logger.info(f"Attempting to load terminology from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.terminology = json.load(file)
            
            logger.info(f"Successfully loaded {len(self.terminology)} terminology entries from {file_path}")
            
            # Store in SimpleMemory for easy access
            self.memory.memories["tiktok_terminology"] = str(self.terminology)
            
            return True
        except FileNotFoundError as e:
            error_msg = f"Terminology file not found at {file_path}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return False
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in terminology file {file_path}: {e}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return False
        except PermissionError as e:
            error_msg = f"Permission denied accessing terminology file {file_path}: {e}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Unexpected error loading terminology from {file_path}: {e}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return False
        
    def update_terminology(self, terminology: List[Dict[str, str]]) -> Any:
        """Store TikTok terminology in memory."""
        self.terminology = terminology
        
        # Store in SimpleMemory for easy access
        self.memory.memories["tiktok_terminology"] = str(terminology)
        
        # Create a simple result object to maintain compatibility
        class Result:
            def __init__(self, applied: int):
                self.applied = applied
        
        return Result(len(terminology))
    
    def render_overlay_for(self, agent_type: str) -> str:
        """Generate memory overlay for a specific agent type."""
        if agent_type not in self.overlays:
            # Build the overlay content
            overlay_content = self._build_overlay_content(agent_type)
            self.overlays[agent_type] = overlay_content
            
        return self.overlays[agent_type]
    
    def _build_overlay_content(self, agent_type: str) -> str:
        """Build the overlay content for a specific agent type."""
        # Base overlay content
        overlay = f"""TIKTOK COMPLIANCE SYSTEM - {agent_type.upper()} AGENT

TIKTOK TERMINOLOGY REFERENCE:
"""
        
        # Add terminology
        for term in self.terminology:
            overlay += f"- {term['acronym']}: {term['meaning']} ({term['category']})\n"
        
        # Add agent-specific context
        if agent_type == "screening":
            overlay += """
SCREENING AGENT CONTEXT:
- Focus on content moderation and initial compliance checks
- Use terminology to identify potential violations
- Apply age-sensitive logic (ASL) for minor-related content
"""
        elif agent_type == "research":
            overlay += """
RESEARCH AGENT CONTEXT:
- Deep dive into compliance issues and policy research
- Understand data sensitivity tiers (T1-T5, with T5 being highest)
- Analyze geographic compliance policies (LCP) and geo-handlers (GH)
"""
        elif agent_type == "validation":
            overlay += """
VALIDATION AGENT CONTEXT:
- Final compliance verification and legal review preparation
- Handle redline flags and softblock decisions
- Ensure proper data retention thresholds (DRT) are met
"""
        
        overlay += """
MEMORY CONTEXT:
- All terminology is stored in SimpleMemory for fast access
- Overlays are generated dynamically based on agent type
- System maintains consistency across all agent interactions
"""
        
        return overlay
    
    def get_terminology(self) -> List[Dict[str, str]]:
        """Retrieve stored terminology."""
        return self.terminology
    
    def get_memory(self) -> SimpleMemory:
        """Get the underlying SimpleMemory instance."""
        return self.memory
    
    def get_terminology_from_memory(self) -> str:
        """Get the terminology stored in SimpleMemory."""
        return self.memory.memories.get("tiktok_terminology", "No terminology stored")
    
    def get_memory_variables(self) -> list[str]:
        """Get the memory variables available in SimpleMemory."""
        return self.memory.memory_variables

def initialize_tiktok_system():
    """Initialize the complete TikTok compliance system with proper memory overlays."""
    
    print("=== Initializing TikTok Compliance System ===\n")
    logger.info("Starting TikTok compliance system initialization")
    
    # Log the current working directory and file paths
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # Check if terminology file exists
    terminology_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "tiktok_terminology.json"
    )
    logger.info(f"Terminology file path: {terminology_file}")
    logger.info(f"Terminology file exists: {os.path.exists(terminology_file)}")
    

    
    # 1. Initialize memory store
    store = TikTokMemoryStore()
    print("Memory store initialized (using SimpleMemory)")
    logger.info("TikTokMemoryStore initialized successfully")
    
    # 2. Load TikTok terminology from file
    logger.info("Attempting to load TikTok terminology...")
    if store.load_terminology_from_file():
        print(f"Loaded {len(store.terminology)} TikTok terminology entries from file")
        logger.info(f"System initialized with {len(store.terminology)} terminology entries")
        
        # Log some sample terminology for verification
        sample_terms = store.terminology[:3]
        logger.info(f"Sample terminology: {sample_terms}")
    else:
        print("Failed to load terminology from file, using fallback")
        logger.warning("Using fallback terminology due to file loading failure")
        
        # Fallback terminology if file loading fails
        fallback_terminology = [
            {"acronym": "NR", "meaning": "Not recommended (restriction/limitation level)", "category": "restriction"},
            {"acronym": "PF", "meaning": "Personalized feed (recommendation algorithm)", "category": "algorithm"},
            {"acronym": "GH", "meaning": "Geo-handler (region-based routing and enforcement)", "category": "geographic"}
        ]
        store.update_terminology(fallback_terminology)
        print(f"✓ Added {len(fallback_terminology)} fallback terminology entries")
        logger.info(f"Fallback terminology loaded with {len(fallback_terminology)} entries")
    
    # 3. Generate memory overlays for each agent
    logger.info("Generating memory overlays for all agent types")
    screening_overlay = store.render_overlay_for("screening")
    research_overlay = store.render_overlay_for("research")
    validation_overlay = store.render_overlay_for("validation")
    
    print(f"Generated memory overlays:")
    print(f"  - Screening: {len(screening_overlay)} characters")
    print(f"  - Research: {len(research_overlay)} characters")
    print(f"  - Validation: {len(validation_overlay)} characters")
    
    logger.info(f"Memory overlays generated - Screening: {len(screening_overlay)}, Research: {len(research_overlay)}, Validation: {len(validation_overlay)} characters")
    
    # 4. Verify terminology is included in overlays
    print("\n=== VERIFICATION ===")
    logger.info("Verifying terminology inclusion in memory overlays")
    
    # Check screening overlay
    if "TIKTOK TERMINOLOGY REFERENCE" in screening_overlay:
        print("Screening overlay contains TikTok terminology")
        logger.info("Screening overlay verification passed")
    else:
        print("Screening overlay missing TikTok terminology")
        logger.error("Screening overlay verification failed")
    
    # Check research overlay  
    if "TIKTOK TERMINOLOGY REFERENCE" in research_overlay:
        print("Research overlay contains TikTok terminology")
        logger.info("Research overlay verification passed")
    else:
        print("Research overlay missing TikTok terminology")
        logger.error("Research overlay verification failed")
    
    # Check validation overlay
    if "TIKTOK TERMINOLOGY REFERENCE" in validation_overlay:
        print("Validation overlay contains TikTok terminology")
        logger.info("Validation overlay verification passed")
    else:
        print("Validation overlay missing TikTok terminology")
        logger.error("Validation overlay verification failed")
    
    # 5. Test prompt building
    print("\n=== PROMPT BUILDING TEST ===")
    logger.info("Testing prompt building with memory overlays")
    
    try:
        screening_prompt = build_screening_prompt(screening_overlay)
        print("Screening prompt built successfully")
        logger.info("Screening prompt built successfully")
        
        research_prompt = build_research_prompt(research_overlay)
        print("Research prompt built successfully")
        logger.info("Research prompt built successfully")
        
        validation_prompt = build_validation_prompt(validation_overlay)
        print("Validation prompt built successfully")
        logger.info("Validation prompt built successfully")
        
    except Exception as e:
        error_msg = f"Prompt building failed: {e}"
        print(f"{error_msg}")
        logger.error(error_msg)
    
    # 6. Show overlay content preview
    print("\n=== OVERLAY CONTENT PREVIEW ===")
    print("Screening overlay preview:")
    preview = screening_overlay[:300] + "..." if len(screening_overlay) > 300 else screening_overlay
    print(preview)
    
    # 7. Test SimpleMemory functionality
    print("\n=== SIMPLEMEMORY TEST ===")
    memory_vars = store.get_memory_variables()
    terminology_from_memory = store.get_terminology_from_memory()
    
    print(f"Memory variables: {memory_vars}")
    print(f"Terminology from memory: {terminology_from_memory[:100]}...")
    
    # Log SimpleMemory test results
    logger.info(f"SimpleMemory test - Variables: {memory_vars}")
    logger.info(f"SimpleMemory test - Terminology length: {len(terminology_from_memory)}")
    logger.info(f"SimpleMemory test - Contains NR: {'NR' in terminology_from_memory}")
    logger.info(f"SimpleMemory test - Contains T5: {'T5' in terminology_from_memory}")
    
    logger.info("System initialization completed successfully")
    
    return {
        "store": store,
        "screening_overlay": screening_overlay,
        "research_overlay": research_overlay,
        "validation_overlay": validation_overlay
    }

def get_agent_overlays():
    """Get the memory overlays for each agent type."""
    store = TikTokMemoryStore()
    
    # Initialize with terminology from file if not already done
    if not store.get_terminology():
        if not store.load_terminology_from_file():
            logger.warning("File loading failed in get_agent_overlays, using fallback terminology")
            # Fallback to minimal terminology if file loading fails
            fallback_terminology = [
                {"acronym": "NR", "meaning": "Not recommended (restriction/limitation level)", "category": "restriction"},
                {"acronym": "PF", "meaning": "Personalized feed (recommendation algorithm)", "category": "algorithm"},
                {"acronym": "GH", "meaning": "Geo-handler (region-based routing and enforcement)", "category": "geographic"}
            ]
            store.update_terminology(fallback_terminology)
            logger.info(f"Fallback terminology loaded in get_agent_overlays with {len(fallback_terminology)} entries")
    
    return {
        "screening": store.render_overlay_for("screening"),
        "research": store.render_overlay_for("research"),
        "validation": store.render_overlay_for("validation")
    }

if __name__ == "__main__":
    try:
        result = initialize_tiktok_system()
        
        print("\n=== SYSTEM READY ===")
        print("To use in your application:")
        print("1. Import: from app.agents.initialize_system import get_agent_overlays")
        print("2. Get overlays: overlays = get_agent_overlays()")
        print("3. Initialize orchestrator: orchestrator = ComplianceOrchestrator(memory_overlay=overlays['screening'])")
        print("\nAll agents will now have access to TikTok terminology context!")
        print("Note: Using SimpleMemory - data persists only during runtime")
        print("\nSimpleMemory Features:")
        print("- Terminology stored in memory.memories['tiktok_terminology']")
        print("- Access via store.get_terminology_from_memory()")
        print("- Memory variables: store.get_memory_variables()")
        
        logger.info("TikTok compliance system initialization completed successfully")
        
    except Exception as e:
        error_msg = f"System initialization failed: {e}"
        print(f"{error_msg}")
        logger.error(error_msg)
        sys.exit(1)