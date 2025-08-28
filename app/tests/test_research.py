#!/usr/bin/env python3
"""
Test cases for ResearchAgent
Run with: python test_research_agent.py
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agents.research import ResearchAgent
from app.agents.state import create_initial_state

class TestResearchAgent:
    
    def __init__(self):
        self.agent = ResearchAgent()
        self.test_cases = [
            {
                "name": "California minors feature",
                "feature_description": "Social media feature for users under 18 in California",
                "screening_analysis": {
                    "agent": "ScreeningAgent",
                    "risk_level": "HIGH",
                    "compliance_required": True,
                    "confidence": 0.9,
                    "trigger_keywords": ["minors", "california", "social media", "age verification"],
                    "reasoning": "Feature involves minors in California jurisdiction",
                    "needs_research": True,
                    "geographic_scope": ["California", "US"],
                    "age_sensitivity": True,
                    "data_sensitivity": "T5"
                },
                "expected_candidates": ["CA_SB976", "COPPA"],
                "expected_jurisdictions": ["California", "US"]
            },
            {
                "name": "EU data processing feature",
                "feature_description": "Personal data processing feature for European users",
                "screening_analysis": {
                    "agent": "ScreeningAgent",
                    "risk_level": "MEDIUM",
                    "compliance_required": True,
                    "confidence": 0.8,
                    "trigger_keywords": ["personal data", "processing", "european", "privacy"],
                    "reasoning": "Feature processes personal data in EU jurisdiction",
                    "needs_research": True,
                    "geographic_scope": ["EU", "Europe"],
                    "age_sensitivity": False,
                    "data_sensitivity": "T4"
                },
                "expected_candidates": ["GDPR", "EU_DSA"],
                "expected_jurisdictions": ["EU"]
            },
            {
                "name": "Basic recommendation system",
                "feature_description": "Simple content recommendation algorithm",
                "screening_analysis": {
                    "agent": "ScreeningAgent",
                    "risk_level": "LOW",
                    "compliance_required": False,
                    "confidence": 0.7,
                    "trigger_keywords": ["recommendation", "algorithm", "content"],
                    "reasoning": "Basic algorithmic recommendation with low compliance risk",
                    "needs_research": False,
                    "geographic_scope": ["unknown"],
                    "age_sensitivity": False,
                    "data_sensitivity": "T2"
                },
                "expected_candidates": [],
                "expected_jurisdictions": []
            }
        ]

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n🔍 Testing: {test_case['name']}")
        print(f"📝 Description: {test_case['feature_description'][:100]}...")
        
        # Create test state with screening analysis
        state = create_initial_state(
            feature_name=test_case["name"],
            feature_description=test_case["feature_description"]
        )
        
        # Add screening analysis to state
        state["screening_analysis"] = test_case["screening_analysis"]
        state["screening_completed"] = True
        
        try:
            # Run the research agent
            result = await self.agent.process(state)
            
            if result.get("research_completed"):
                analysis = result["research_analysis"]
                print(f"✅ Test completed successfully")
                print(f"🎯 Confidence Score: {analysis.get('confidence_score')}")
                print(f"📊 Candidates Found: {len(analysis.get('candidates', []))}")
                print(f"📋 Evidence Pieces: {len(analysis.get('evidence', []))}")
                print(f"🔍 Query Used: {analysis.get('query_used', 'N/A')[:100]}")
                
                # Show candidates
                if analysis.get('candidates'):
                    print(f"🏛️  Regulation Candidates:")
                    for candidate in analysis['candidates'][:3]:  # Show top 3
                        print(f"   - {candidate.get('reg', 'N/A')}: {candidate.get('why', 'N/A')} (Score: {candidate.get('score', 0)})")
                
                # Show evidence
                if analysis.get('evidence'):
                    print(f"📄 Evidence Found:")
                    for evidence in analysis['evidence'][:2]:  # Show top 2
                        print(f"   - {evidence.get('name', 'N/A')} ({evidence.get('jurisdiction', 'N/A')})")
                        print(f"     Score: {evidence.get('score', 0)}")
                
                # Show risk assessment if available
                if result.get('research_risk_assessment'):
                    risk_assessment = result['research_risk_assessment']
                    print(f"⚠️  Risk Assessment:")
                    print(f"   - Overall Risk Score: {risk_assessment.get('overall_risk_score', 'N/A')}")
                    print(f"   - Risk Level: {risk_assessment.get('risk_level', 'N/A')}")
                
                # Validate expected outcomes
                self._validate_result(test_case, analysis, result)
                
                return result
            else:
                print(f"❌ Test failed: Research not completed")
                if result.get("research_error"):
                    print(f"Error: {result['research_error']}")
                return result
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            return {"error": str(e)}

    def _validate_result(self, test_case: Dict[str, Any], analysis: Dict[str, Any], result: Dict[str, Any]):
        """Validate the result against expected outcomes"""
        print(f"\n📋 Validation Results:")
        
        # Check output format compliance
        required_fields = ["agent", "candidates", "evidence", "query_used", "confidence_score"]
        missing_fields = [field for field in required_fields if field not in analysis]
        
        if not missing_fields:
            print(f"✅ Output format compliance: All required fields present")
        else:
            print(f"❌ Output format compliance: Missing fields {missing_fields}")
        
        # Check agent field
        if analysis.get("agent") == "ResearchAgent":
            print(f"✅ Agent field correct: {analysis['agent']}")
        else:
            print(f"❌ Agent field incorrect: {analysis.get('agent')}")
        
        # Validate candidates structure
        candidates = analysis.get("candidates", [])
        if candidates:
            valid_candidates = all(
                isinstance(c, dict) and "reg" in c and "why" in c and "score" in c
                for c in candidates
            )
            if valid_candidates:
                print(f"✅ Candidates structure valid: {len(candidates)} candidates")
            else:
                print(f"❌ Candidates structure invalid")
        
        # Validate evidence structure
        evidence = analysis.get("evidence", [])
        if evidence:
            required_evidence_fields = ["reg", "jurisdiction", "name", "section", "url", "excerpt", "score"]
            valid_evidence = all(
                isinstance(e, dict) and all(field in e for field in required_evidence_fields)
                for e in evidence
            )
            if valid_evidence:
                print(f"✅ Evidence structure valid: {len(evidence)} pieces")
            else:
                print(f"❌ Evidence structure invalid")
        
        # Check confidence score
        confidence = analysis.get("confidence_score")
        if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
            print(f"✅ Confidence score valid: {confidence}")
        else:
            print(f"❌ Confidence score invalid: {confidence}")

    async def run_all_tests(self):
        """Run all test cases"""
        print("🚀 Starting ResearchAgent Tests")
        print("="*50)
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[Test {i}/{len(self.test_cases)}]")
            result = await self.run_single_test(test_case)
            results.append({
                "test_name": test_case["name"],
                "result": result,
                "success": result.get("research_completed", False)
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
    tester = TestResearchAgent()
    results = asyncio.run(tester.run_all_tests())
    
    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(failed_count)

if __name__ == "__main__":
    main()
