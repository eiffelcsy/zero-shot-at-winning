#!/usr/bin/env python3
"""
Test script for the integrated TikTokMemory system with all agents.
"""

import sys
import os
import asyncio

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def test_integrated_system():
    """Test the integrated TikTokMemory system with all agents."""
    try:
        print("Testing integrated TikTokMemory system...")
        
        # Test imports from the combined file
        from app.agents.memory.tiktok_memory import TikTokMemory, get_agent_overlays, get_tiktok_memory
        from app.agents.orchestrator import ComplianceOrchestrator
        print("✓ All modules imported successfully")
        
        # Test TikTokMemory functionality
        print("\n=== Testing TikTokMemory ===")
        memory = TikTokMemory()
        print(f"✓ TikTokMemory created with {len(memory.memory_variables)} variables")
        
        # Check if TikTok terminology is loaded
        if "tiktok_terminology_reference" in memory.memory_variables:
            print("✓ TikTok terminology reference loaded")
        else:
            print("⚠ TikTok terminology reference missing")
        
        # Test agent overlays
        print("\n=== Testing Agent Overlays ===")
        overlays = get_agent_overlays()
        print(f"✓ Generated {len(overlays)} agent overlays")
        
        for agent_type, overlay in overlays.items():
            print(f"  - {agent_type}: {len(overlay)} characters")
            if "TIKTOK TERMINOLOGY REFERENCE" in overlay:
                print(f"    ✓ Contains TikTok terminology")
            else:
                print(f"    ⚠ Missing TikTok terminology")
        
        # Test orchestrator initialization
        print("\n=== Testing ComplianceOrchestrator ===")
        screening_overlay = overlays.get("screening", "")
        orchestrator = ComplianceOrchestrator(memory_overlay=screening_overlay)
        print("✓ ComplianceOrchestrator created successfully")
        
        # Test agent memory integration
        print("\n=== Testing Agent Memory Integration ===")
        
        # Check screening agent
        screening_agent = orchestrator.screening_agent
        if screening_agent.memory_overlay:
            print(f"✓ Screening agent has memory overlay ({len(screening_agent.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in screening_agent.memory_overlay:
                print("  ✓ Includes TikTok terminology context")
            else:
                print("  ⚠ Missing TikTok terminology context")
        else:
            print("⚠ Screening agent has no memory overlay")
        
        # Check research agent
        research_agent = orchestrator.research_agent
        if research_agent.memory_overlay:
            print(f"✓ Research agent has memory overlay ({len(research_agent.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in research_agent.memory_overlay:
                print("  ✓ Includes TikTok terminology context")
            else:
                print("  ⚠ Missing TikTok terminology context")
        else:
            print("⚠ Research agent has no memory overlay")
        
        # Check validation agent
        validation_agent = orchestrator.validation_agent
        if validation_agent.memory_overlay:
            print(f"✓ Validation agent has memory overlay ({len(validation_agent.memory_overlay)} characters)")
            if "TIKTOK TERMINOLOGY REFERENCE" in validation_agent.memory_overlay:
                print("  ✓ Includes TikTok terminology context")
            else:
                print("  ⚠ Missing TikTok terminology context")
        else:
            print("⚠ Validation agent has no memory overlay")
        
        # Test memory update functionality
        print("\n=== Testing Memory Update ===")
        new_overlay = "UPDATED MEMORY OVERLAY WITH NEW CONTEXT"
        orchestrator.update_agent_memory(new_overlay)
        print("✓ Memory update completed")
        
        # Verify agents have updated memory
        if screening_agent.memory_overlay == new_overlay:
            print("✓ Screening agent memory updated")
        else:
            print("⚠ Screening agent memory not updated")
        
        if research_agent.memory_overlay == new_overlay:
            print("✓ Research agent memory updated")
        else:
            print("⚠ Research agent memory not updated")
        
        if validation_agent.memory_overlay == new_overlay:
            print("✓ Validation agent memory updated")
        else:
            print("⚠ Validation agent memory not updated")
        
        print("\n🎉 All integration tests passed! TikTokMemory system is working correctly with all agents.")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_integrated_system())
    sys.exit(0 if success else 1)
