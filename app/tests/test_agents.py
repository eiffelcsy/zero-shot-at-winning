import pytest
from app.agents.orchestrator import ComplianceOrchestrator

# Test data from TikTok challenge
test_cases = [
    {
        "name": "Utah Curfew Feature",
        "feature_description": "ASL-based curfew restrictions for Utah minors using GH with T5 data processing and Jellybean parental controls",
        "expected_result": {
            "needs_geo_logic": "YES",
            "risk_level": "HIGH",
            "geographic_scope": ["Utah"],
            "age_sensitivity": True
        }
    },
    {
        "name": "California Feed Defaults",
        "feature_description": "PF default toggle with NR enforcement for California teens under Snowcap framework",
        "expected_result": {
            "needs_geo_logic": "YES",
            "risk_level": "HIGH",
            "geographic_scope": ["California"]
        }
    },
    {
        "name": "Business Feature Rollout",
        "feature_description": "New chat UI rollout in US for market testing with standard analytics",
        "expected_result": {
            "needs_geo_logic": "NO",
            "risk_level": "LOW"
        }
    },
    {
        "name": "EU DSA Content Moderation",
        "feature_description": "Content visibility lock with NSP for EU DSA transparency requirements using Redline review",
        "expected_result": {
            "needs_geo_logic": "YES",
            "risk_level": "HIGH",
            "geographic_scope": ["EU"]
        }
    }
]

@pytest.mark.asyncio
async def test_agent_pipeline():
    """Test the complete multi-agent pipeline"""
    orchestrator = ComplianceOrchestrator(kb_dir="data/kb")
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        result = await orchestrator.analyze_feature(
            test_case["name"],
            test_case["feature_description"]
        )
        
        # Validate results
        assert result["needs_geo_logic"] in ["YES", "NO", "REVIEW"], f"Invalid decision: {result['needs_geo_logic']}"
        assert "reasoning" in result, "Missing reasoning in result"
        assert "confidence_score" in result, "Missing confidence score"
        
        print(f"✅ {test_case['name']}: {result['needs_geo_logic']}")
        print(f"   Confidence: {result['confidence_score']}")
        print(f"   Reasoning: {result['reasoning'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test_agent_pipeline())