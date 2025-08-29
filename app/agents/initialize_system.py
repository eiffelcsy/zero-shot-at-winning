#!/usr/bin/env python3
"""
System initialization script to properly set up TikTok terminology and memory overlays.
This ensures all agents have access to the TikTok context they need.
"""

import os
import sys

# Add the app directory to Python path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory_pg import PostgresMemoryStore
from agents.prompts.templates import build_screening_prompt, build_research_prompt, build_validation_prompt

def initialize_tiktok_system():
    """Initialize the complete TikTok compliance system with proper memory overlays."""
    
    print("=== Initializing TikTok Compliance System ===\n")
    
    # 1. Initialize memory store
    store = PostgresMemoryStore()
    print("Memory store initialized")
    
    # 2. Populate TikTok terminology
    terminology = [
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
    
    result = store.update_terminology(terminology)
    print(f"Added {result.applied} TikTok terminology entries")
    
    # 3. Generate memory overlays for each agent
    screening_overlay = store.render_overlay_for("screening")
    research_overlay = store.render_overlay_for("research")
    validation_overlay = store.render_overlay_for("validation")
    
    print(f"Generated memory overlays:")
    print(f"  - Screening: {len(screening_overlay)} characters")
    print(f"  - Research: {len(research_overlay)} characters")
    print(f"  - Validation: {len(validation_overlay)} characters")
    
    # 4. Verify terminology is included in overlays
    print("\n=== VERIFICATION ===")
    
    # Check screening overlay
    if "TIKTOK TERMINOLOGY REFERENCE" in screening_overlay:
        print("Screening overlay contains TikTok terminology")
    else:
        print("Screening overlay missing TikTok terminology")
    
    # Check research overlay  
    if "TIKTOK TERMINOLOGY REFERENCE" in research_overlay:
        print("Research overlay contains TikTok terminology")
    else:
        print("Research overlay missing TikTok terminology")
    
    # Check validation overlay
    if "TIKTOK TERMINOLOGY REFERENCE" in validation_overlay:
        print("Validation overlay contains TikTok terminology")
    else:
        print("Validation overlay missing TikTok terminology")
    
    # 5. Test prompt building
    print("\n=== PROMPT BUILDING TEST ===")
    
    try:
        screening_prompt = build_screening_prompt(screening_overlay)
        print("Screening prompt built successfully")
        
        research_prompt = build_research_prompt(research_overlay)
        print("Research prompt built successfully")
        
        validation_prompt = build_validation_prompt(validation_overlay)
        print("Validation prompt built successfully")
        
    except Exception as e:
        print(f"Prompt building failed: {e}")
    
    # 6. Show overlay content preview
    print("\n=== OVERLAY CONTENT PREVIEW ===")
    print("Screening overlay preview:")
    print(screening_overlay[:300] + "..." if len(screening_overlay) > 300 else screening_overlay)
    
    return {
        "store": store,
        "screening_overlay": screening_overlay,
        "research_overlay": research_overlay,
        "validation_overlay": validation_overlay
    }

def get_agent_overlays():
    """Get the memory overlays for each agent type."""
    store = PostgresMemoryStore()
    
    return {
        "screening": store.render_overlay_for("screening"),
        "research": store.render_overlay_for("research"),
        "validation": store.render_overlay_for("validation")
    }

if __name__ == "__main__":
    result = initialize_tiktok_system()
    
    print("\n=== SYSTEM READY ===")
    print("To use in your application:")
    print("1. Import: from app.agents.initialize_system import get_agent_overlays")
    print("2. Get overlays: overlays = get_agent_overlays()")
    print("3. Initialize orchestrator: orchestrator = ComplianceOrchestrator(memory_overlay=overlays['screening'])")
    print("\nAll agents will now have access to TikTok terminology context!")
