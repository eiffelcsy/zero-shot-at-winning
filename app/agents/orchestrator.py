from typing import Dict, Any, List
from datetime import datetime
import asyncio
from .screening import ScreeningAgent

class ComplianceOrchestrator:
    """Orchestrates the multi-agent compliance analysis workflow"""
    
    def __init__(self, screening_agent: ScreeningAgent, research_agent=None, validation_agent=None):
        self.screening_agent = screening_agent
        self.research_agent = research_agent
        self.validation_agent = validation_agent
        
        # Workflow state
        self.analysis_history = []
        
    async def analyze_feature(self, feature_name: str, feature_description: str) -> Dict[str, Any]:
        """Main entry point for feature compliance analysis"""
        
        analysis_session = {
            "session_id": f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "feature_name": feature_name,
            "feature_description": feature_description,
            "started_at": datetime.utcnow(),
            "agents_used": [],
            "results": {}
        }
        
        try:
            # Step 1: Screening
            screening_input = {
                "feature_description": feature_description,
                "feature_name": feature_name
            }
            
            screening_result = await self.screening_agent.process(screening_input)
            analysis_session["agents_used"].append("screening")
            analysis_session["results"]["screening"] = screening_result
            
            # Step 2: Determine next steps based on screening
            next_step = screening_result.get("next_step", "validation")
            
            if next_step == "research" and self.research_agent:
                # Add research step when you implement research agent
                pass
            elif next_step == "validation" and self.validation_agent:
                # Add validation step when you implement validation agent  
                pass
            
            # Final assessment
            final_assessment = self._create_final_assessment(analysis_session)
            analysis_session["final_assessment"] = final_assessment
            analysis_session["completed_at"] = datetime.utcnow()
            
            # Store for learning
            self.analysis_history.append(analysis_session)
            
            return final_assessment
            
        except Exception as e:
            return self._create_error_assessment(str(e), analysis_session)
    
    def _create_final_assessment(self, session: Dict) -> Dict[str, Any]:
        """Create final compliance assessment from agent results"""
        
        screening_analysis = session["results"]["screening"]["analysis"]
        
        return {
            "session_id": session["session_id"],
            "feature_name": session["feature_name"],
            "compliance_required": screening_analysis.get("compliance_required", False),
            "risk_level": screening_analysis.get("risk_level", "UNKNOWN"),
            "confidence_score": screening_analysis.get("confidence", 0.0),
            "applicable_regulations": [],  # Will be filled by research agent
            "reasoning": screening_analysis.get("reasoning", ""),
            "geographic_scope": screening_analysis.get("geographic_scope", ["unknown"]),
            "age_sensitive": screening_analysis.get("age_sensitivity", False),
            "human_review_needed": screening_analysis.get("confidence", 0) < 0.7,
            "agents_consulted": session["agents_used"],
            "analysis_timestamp": session["started_at"].isoformat()
        }
    
    def _create_error_assessment(self, error: str, session: Dict) -> Dict[str, Any]:
        """Create error assessment"""
        return {
            "session_id": session.get("session_id", "error_session"),
            "feature_name": session.get("feature_name", "unknown"),
            "compliance_required": None,
            "risk_level": "ERROR",
            "confidence_score": 0.0,
            "error": error,
            "human_review_needed": True,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    async def batch_analyze(self, features: List[Dict]) -> List[Dict]:
        """Analyze multiple features in batch"""
        tasks = []
        for feature in features:
            task = self.analyze_feature(
                feature.get("feature_name", ""),
                feature.get("feature_description", "")
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)