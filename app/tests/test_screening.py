#!/usr/bin/env python3
"""
Test cases for ScreeningAgent
Run with: python test_screening_agent.py
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify the key is loaded
print("API Key loaded:", "✅" if os.getenv("OPENAI_API_KEY") else "❌")

# Add the app directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.agents.screening import ScreeningAgent
from app.agents.state import create_initial_state

class TestScreeningAgent:
    
    def __init__(self):
        self.agent = ScreeningAgent()
        self.test_cases = [
            {
                "name": "Age-sensitive feature with location tracking",
                "feature_name": "Teen Location Sharing",
                "feature_description": "Allow users under 18 to share their real-time location with friends using GH for geographic enforcement and ASL for age verification. Uses T5 data processing.",
                "context_documents": "Feature uses Jellybean parental controls and requires Snowcap compliance.",
                "expected_risk": "HIGH",
                "expected_age_sensitivity": True,
                "expected_data_sensitivity": "T5"
            },
            {
                "name": "Basic content recommendation",
                "feature_description": "Standard PF algorithm that shows users recommended content based on their viewing history. Uses T2 data processing.",
                "context_documents": "Standard recommendation engine with BB analysis.",
                "expected_risk": "LOW",
                "expected_age_sensitivity": False,
                "expected_data_sensitivity": "T2"
            },
            {
                "name": "Content moderation system",
                "feature_description": "Automated content filtering using CDS for compliance monitoring. Implements Softblock for policy violations and uses Spanner rule engine.",
                "context_documents": "Content moderation with EchoTrace logging for compliance verification.",
                "expected_risk": "MEDIUM",
                "expected_age_sensitivity": False,
                "expected_data_sensitivity": "T3"
            }
        ]

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"📝 Description: {test_case['feature_description'][:100]}...")
        
        # Create test state
        state = create_initial_state(
            feature_name=test_case.get("feature_name", test_case["name"]),
            feature_description=test_case["feature_description"],
            context_documents=test_case.get("context_documents", "")
        )
        
        try:
            # Run the screening agent
            result = await self.agent.process(state)
            
            if result.get("screening_completed"):
                analysis = result["screening_analysis"]
                print(f"✅ Test completed successfully")
                print(f"📊 Risk Level: {analysis.get('risk_level')}")
                print(f"🎯 Confidence: {analysis.get('confidence')}")
                print(f"👶 Age Sensitive: {analysis.get('age_sensitivity')}")
                print(f"🔒 Data Sensitivity: {analysis.get('data_sensitivity')}")
                print(f"🌍 Geographic Scope: {analysis.get('geographic_scope')}")
                print(f"🔍 Trigger Keywords: {analysis.get('trigger_keywords')}")
                
                # Validate expected outcomes
                self._validate_result(test_case, analysis)
                
                return result
            else:
                print(f"❌ Test failed: Screening not completed")
                if result.get("screening_analysis", {}).get("error"):
                    print(f"Error: {result['screening_analysis']['error']}")
                return result
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            return {"error": str(e)}

    def _validate_result(self, test_case: Dict[str, Any], analysis: Dict[str, Any]):
        """Validate the result against expected outcomes"""
        print(f"\n📋 Validation Results:")
        
        # Check risk level
        expected_risk = test_case.get("expected_risk")
        actual_risk = analysis.get("risk_level")
        if expected_risk and actual_risk == expected_risk:
            print(f"✅ Risk level matches: {actual_risk}")
        elif expected_risk:
            print(f"⚠️  Risk level mismatch: expected {expected_risk}, got {actual_risk}")
        
        # Check age sensitivity
        expected_age = test_case.get("expected_age_sensitivity")
        actual_age = analysis.get("age_sensitivity")
        if expected_age is not None and actual_age == expected_age:
            print(f"✅ Age sensitivity matches: {actual_age}")
        elif expected_age is not None:
            print(f"⚠️  Age sensitivity mismatch: expected {expected_age}, got {actual_age}")
        
        # Check data sensitivity
        expected_data = test_case.get("expected_data_sensitivity")
        actual_data = analysis.get("data_sensitivity")
        if expected_data and actual_data == expected_data:
            print(f"✅ Data sensitivity matches: {actual_data}")
        elif expected_data:
            print(f"⚠️  Data sensitivity mismatch: expected {expected_data}, got {actual_data}")

    async def run_all_tests(self):
        """Run all test cases"""
        print("🚀 Starting ScreeningAgent Tests")
        print("="*50)
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[Test {i}/{len(self.test_cases)}]")
            result = await self.run_single_test(test_case)
            results.append({
                "test_name": test_case["name"],
                "result": result,
                "success": result.get("screening_completed", False)
            })
        
        # Summary
        print("\n" + "="*50)
        print("📊 Test Summary:")
        successful = sum(1 for r in results if r["success"])
        print(f"✅ Successful: {successful}/{len(results)}")
        print(f"❌ Failed: {len(results) - successful}/{len(results)}")
        
        return results

def main():
    """Main test runner"""
    tester = TestScreeningAgent()
    results = asyncio.run(tester.run_all_tests())
    
    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(failed_count)

if __name__ == "__main__":
    main()
