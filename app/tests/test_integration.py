#!/usr/bin/env python3
"""
Integration test for ScreeningAgent -> ResearchAgent workflow
Run with: python test_integration.py
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agents.screening import ScreeningAgent
from app.agents.research import ResearchAgent
from app.agents.state import create_initial_state

class TestIntegration:
    
    def __init__(self):
        self.screening_agent = ScreeningAgent()
        self.research_agent = ResearchAgent()
        self.test_case = {
            "feature_name": "California Teen Safety Feature",
            "feature_description": "A new feature that uses ASL for age verification and GH for geographic targeting to provide enhanced safety controls for users under 18 in California. The feature processes T5 data including location information and uses Jellybean parental controls.",
            "context_documents": "Feature implements Snowcap child safety framework and uses EchoTrace for compliance logging. Requires Redline legal review before deployment."
        }

    async def run_integration_test(self):
        """Run full screening -> research workflow"""
        print("🚀 Starting Integration Test: Screening -> Research")
        print("="*60)
        
        # Step 1: Initialize state
        print("\n📝 Step 1: Initialize State")
        state = create_initial_state(
            feature_name=self.test_case["feature_name"],
            feature_description=self.test_case["feature_description"],
            context_documents=self.test_case["context_documents"]
        )
        print(f"✅ Initial state created for: {self.test_case['feature_name']}")
        
        # Step 2: Run Screening Agent
        print("\n🔍 Step 2: Run Screening Agent")
        try:
            screening_result = await self.screening_agent.process(state)
            
            if screening_result.get("screening_completed"):
                analysis = screening_result["screening_analysis"]
                print(f"✅ Screening completed successfully")
                print(f"📊 Risk Level: {analysis.get('risk_level')}")
                print(f"🎯 Confidence: {analysis.get('confidence')}")
                print(f"🔍 Needs Research: {analysis.get('needs_research')}")
                
                # Update state with screening results
                state.update(screening_result)
            else:
                print(f"❌ Screening failed")
                return False
                
        except Exception as e:
            print(f"❌ Screening failed with exception: {str(e)}")
            return False
        
        # Step 3: Run Research Agent (only if screening says research is needed)
        if analysis.get("needs_research"):
            print("\n🔬 Step 3: Run Research Agent")
            try:
                research_result = await self.research_agent.process(state)
                
                if research_result.get("research_completed"):
                    research_analysis = research_result["research_analysis"]
                    print(f"✅ Research completed successfully")
                    print(f"🎯 Confidence Score: {research_analysis.get('confidence_score')}")
                    print(f"📊 Candidates: {len(research_analysis.get('candidates', []))}")
                    print(f"📋 Evidence: {len(research_analysis.get('evidence', []))}")
                    
                    # Show key findings
                    print(f"\n🏛️  Top Regulation Candidates:")
                    for candidate in research_analysis.get('candidates', [])[:3]:
                        print(f"   - {candidate.get('reg')}: {candidate.get('why')} (Score: {candidate.get('score')})")
                    
                    print(f"\n📄 Top Evidence:")
                    for evidence in research_analysis.get('evidence', [])[:2]:
                        print(f"   - {evidence.get('name')} ({evidence.get('jurisdiction')})")
                        print(f"     {evidence.get('excerpt', '')[:100]}...")
                    
                    # Update state with research results
                    state.update(research_result)
                    
                    # Final validation
                    self._validate_integration(state)
                    
                    return True
                else:
                    print(f"❌ Research failed")
                    if research_result.get("research_error"):
                        print(f"Error: {research_result['research_error']}")
                    return False
                    
            except Exception as e:
                print(f"❌ Research failed with exception: {str(e)}")
                return False
        else:
            print("\n⏭️  Step 3: Skipped - Research not needed according to screening")
            return True

    def _validate_integration(self, state: Dict[str, Any]):
        """Validate the complete integration workflow"""
        print(f"\n📋 Integration Validation:")
        
        # Check state completeness
        required_fields = [
            "screening_analysis", "screening_completed",
            "research_analysis", "research_completed"
        ]
        
        missing_fields = [field for field in required_fields if not state.get(field)]
        if not missing_fields:
            print(f"✅ State completeness: All workflow steps completed")
        else:
            print(f"❌ State completeness: Missing {missing_fields}")
        
        # Check data flow continuity
        screening_risk = state["screening_analysis"].get("risk_level")
        research_confidence = state["research_analysis"].get("confidence_score")
        
        print(f"📊 Data Flow:")
        print(f"   - Screening Risk Level: {screening_risk}")
        print(f"   - Research Confidence: {research_confidence}")
        
        # Check workflow consistency
        screening_geo = state["screening_analysis"].get("geographic_scope", [])
        research_candidates = state["research_analysis"].get("candidates", [])
        
        if screening_geo and research_candidates:
            print(f"✅ Workflow consistency: Geographic scope led to regulation candidates")
        elif not screening_geo and not research_candidates:
            print(f"✅ Workflow consistency: No geographic scope, no specific candidates")
        else:
            print(f"⚠️  Workflow consistency: Potential mismatch between screening and research")

async def main():
    """Main test runner"""
    tester = TestIntegration()
    success = await tester.run_integration_test()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Integration Test PASSED")
        sys.exit(0)
    else:
        print("❌ Integration Test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
