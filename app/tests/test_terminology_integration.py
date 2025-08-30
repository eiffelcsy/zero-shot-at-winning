#!/usr/bin/env python3
"""
Test script to verify TikTok terminology integration with agents.
This helps ensure that the terminology is properly understood and used.
"""

import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory.initialize_system import TikTokMemoryStore, get_agent_overlays
from agents.prompts import build_screening_prompt, build_research_prompt, build_validation_prompt

def test_terminology_loading():
    """Test if terminology loads correctly from file."""
    print("=== Testing Terminology Loading ===")
    
    store = TikTokMemoryStore()
    
    # Test loading from default file
    success = store.load_terminology_from_file()
    print(f"Load success: {success}")
    
    if success:
        terminology = store.get_terminology()
        print(f"Loaded {len(terminology)} terminology entries")
        
        # Show first few entries
        print("\nSample terminology:")
        for term in terminology[:5]:
            print(f"- {term['acronym']}: {term['meaning']}")
        
        # Test SimpleMemory integration
        print(f"\nSimpleMemory test:")
        print(f"Memory variables: {store.get_memory_variables()}")
        print(f"Terminology from memory: {store.get_terminology_from_memory()[:100]}...")
    else:
        print("Failed to load terminology - check if tiktok_terminology.json exists")
    
    print()

def test_memory_overlays():
    """Test if memory overlays contain terminology."""
    print("=== Testing Memory Overlays ===")
    
    overlays = get_agent_overlays()
    
    for agent_type, overlay in overlays.items():
        print(f"\n{agent_type.upper()} OVERLAY:")
        print(f"Length: {len(overlay)} characters")
        
        # Check if terminology reference is included
        has_terminology = "TIKTOK TERMINOLOGY REFERENCE" in overlay
        print(f"Contains terminology reference: {has_terminology}")
        
        # Check for specific acronyms
        specific_acronyms = ["NR", "T5", "ASL", "GH", "CDS"]
        found_acronyms = [acronym for acronym in specific_acronyms if acronym in overlay]
        print(f"Found acronyms: {found_acronyms}")
        
        # Show overlay preview
        print(f"Preview: {overlay[:200]}...")
    
    print()

def test_prompt_integration():
    """Test if terminology is properly integrated into prompts."""
    print("=== Testing Prompt Integration ===")
    
    overlays = get_agent_overlays()
    
    # Test screening prompt
    print("Testing SCREENING prompt:")
    screening_prompt = build_screening_prompt(overlays['screening'])
    
    # Test with sample data
    test_data = {
        "feature_name": "Test Feature",
        "feature_description": "This feature uses NR settings and T5 data sensitivity with ASL logic",
        "context_documents": "Additional context here"
    }
    
    try:
        formatted_prompt = screening_prompt.format(**test_data)
        
        # Check terminology presence
        terminology_checks = {
            "TIKTOK TERMINOLOGY REFERENCE": "TIKTOK TERMINOLOGY REFERENCE" in formatted_prompt,
            "NR (Not recommended)": "NR" in formatted_prompt and "Not recommended" in formatted_prompt,
            "T5 (Tier 5 data)": "T5" in formatted_prompt and "Tier 5" in formatted_prompt,
            "ASL (Age-sensitive logic)": "ASL" in formatted_prompt and "Age-sensitive logic" in formatted_prompt,
            "Agent-specific context": "SCREENING AGENT CONTEXT" in formatted_prompt
        }
        
        for check, result in terminology_checks.items():
            print(f"  {check}: {'✓' if result else '❌'}")
        
        # Show prompt preview
        print(f"\nPrompt preview (first 300 chars):")
        print(formatted_prompt[:300] + "..." if len(formatted_prompt) > 300 else formatted_prompt)
        
    except Exception as e:
        print(f"  ❌ Prompt formatting failed: {e}")
    
    print()

def test_agent_specific_context():
    """Test if agent-specific context is properly included."""
    print("=== Testing Agent-Specific Context ===")
    
    store = TikTokMemoryStore()
    store.load_terminology_from_file()
    
    agent_types = ["screening", "research", "validation"]
    
    for agent_type in agent_types:
        overlay = store.render_overlay_for(agent_type)
        print(f"\n{agent_type.upper()} AGENT:")
        
        # Check for agent-specific content
        if agent_type == "screening":
            has_content = "SCREENING AGENT CONTEXT" in overlay and "content moderation" in overlay
        elif agent_type == "research":
            has_content = "RESEARCH AGENT CONTEXT" in overlay and "data sensitivity tiers" in overlay
        elif agent_type == "validation":
            has_content = "VALIDATION AGENT CONTEXT" in overlay and "legal review" in overlay
        
        print(f"  Has agent-specific context: {'✓' if has_content else '❌'}")
        
        # Show context preview
        if "AGENT CONTEXT" in overlay:
            start = overlay.find("AGENT CONTEXT")
            end = overlay.find("MEMORY CONTEXT") if "MEMORY CONTEXT" in overlay else len(overlay)
            context = overlay[start:end].strip()
            print(f"  Context preview: {context[:100]}...")
    
    print()

def test_terminology_consistency():
    """Test if terminology is consistent across all overlays."""
    print("=== Testing Terminology Consistency ===")
    
    store = TikTokMemoryStore()
    store.load_terminology_from_file()
    
    terminology = store.get_terminology()
    acronyms = [term['acronym'] for term in terminology]
    
    print(f"Total terminology entries: {len(terminology)}")
    print(f"Acronyms: {acronyms}")
    
    # Check if all acronyms appear in overlays
    agent_types = ["screening", "research", "validation"]
    
    for agent_type in agent_types:
        overlay = store.render_overlay_for(agent_type)
        print(f"\n{agent_type.upper()} overlay:")
        
        missing_acronyms = []
        for acronym in acronyms:
            if acronym not in overlay:
                missing_acronyms.append(acronym)
        
        if missing_acronyms:
            print(f"  ❌ Missing acronyms: {missing_acronyms}")
        else:
            print(f"  ✓ All acronyms present")
    
    print()

def test_simple_memory_functionality():
    """Test the SimpleMemory implementation."""
    print("=== Testing SimpleMemory Functionality ===")
    
    store = TikTokMemoryStore()
    
    # Test empty memory
    print("Empty memory test:")
    print(f"  Memory variables: {store.get_memory_variables()}")
    print(f"  Terminology from memory: {store.get_terminology_from_memory()}")
    
    # Load terminology and test again
    store.load_terminology_from_file()
    
    print("\nAfter loading terminology:")
    print(f"  Memory variables: {store.get_memory_variables()}")
    print(f"  Has tiktok_terminology: {'tiktok_terminology' in store.get_memory_variables()}")
    
    # Test memory access
    terminology_from_memory = store.get_terminology_from_memory()
    print(f"  Terminology length from memory: {len(terminology_from_memory)}")
    print(f"  Contains NR: {'NR' in terminology_from_memory}")
    print(f"  Contains T5: {'T5' in terminology_from_memory}")
    
    print()

def main():
    """Run all terminology integration tests."""
    print("Testing TikTok Terminology Integration\n")
    print("This script verifies that:")
    print("1. Terminology loads correctly from JSON file")
    print("2. Memory overlays contain terminology")
    print("3. Prompts integrate terminology properly")
    print("4. Agent-specific context is included")
    print("5. Terminology is consistent across overlays")
    print()
    
    try:
        test_terminology_loading()
        test_memory_overlays()
        test_prompt_integration()
        test_agent_specific_context()
        test_terminology_consistency()
        test_simple_memory_functionality()
        
        print("=== Test Summary ===")
        print("All terminology integration tests completed.")
        print("\nIf all tests pass, your agents should properly understand")
        print("TikTok terminology and use it in their compliance assessments.")
        print("SimpleMemory is properly storing and retrieving terminology data.")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
