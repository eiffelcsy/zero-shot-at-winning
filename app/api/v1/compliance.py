from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.agents.orchestrator import ComplianceOrchestrator
from app.agents.feedback.learning import LearningAgent

router = APIRouter()

# Initialize orchestrator with combined memory system
orchestrator = ComplianceOrchestrator(use_combined_memory=True)
learning_agent = LearningAgent()

@router.post("/analyze-feature")
async def analyze_feature_compliance(
    feature_name: str,
    feature_description: str
) -> Dict[str, Any]:
    """Analyze feature using LangGraph multi-agent workflow"""
    
    try:
        result = await orchestrator.analyze_feature(feature_name, feature_description)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-feedback")
async def submit_feedback(
    session_id: str,
    analysis_result: Dict[str, Any],
    feedback_type: str,
    correction: str = "",
    comments: str = ""
):
    """Submit user feedback for learning"""
    
    try:
        learning_agent.collect_feedback(
            analysis_result=analysis_result,
            user_feedback={
                "type": feedback_type,
                "correction": correction,
                "comments": comments
            }
        )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback-analytics")
async def get_feedback_analytics():
    """Get feedback analytics and improvement suggestions"""
    
    try:
        analytics = learning_agent.analyze_feedback_patterns()
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))