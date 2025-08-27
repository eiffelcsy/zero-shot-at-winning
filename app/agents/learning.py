from .base import BaseComplianceAgent
from typing import Dict, Any, List
from datetime import datetime
import json
import os

class LearningAgent(BaseComplianceAgent):
    """Learns from user feedback to improve system"""
    
    def __init__(self, feedback_file: str = "data/feedback.jsonl"):
        super().__init__("LearningAgent")
        self.feedback_file = feedback_file
        self._ensure_feedback_file()
    
    def _ensure_feedback_file(self):
        """Create feedback file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                pass  # Create empty file
    
    def collect_feedback(self, analysis_result: Dict, user_feedback: Dict):
        """Store feedback for future learning"""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis_result,
            "user_correction": user_feedback.get("correction", ""),
            "feedback_type": user_feedback.get("type", ""),  # "accurate", "incorrect", "needs_context"
            "user_comments": user_feedback.get("comments", ""),
            "session_id": analysis_result.get("session_id", "")
        }
        
        self._store_feedback(feedback_entry)
        self.logger.info(f"Feedback collected: {feedback_entry['feedback_type']}")
    
    def _store_feedback(self, feedback_entry: Dict):
        """Store feedback entry to file"""
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Find common mistakes and improvement areas"""
        feedback_data = self._load_all_feedback()
        
        if not feedback_data:
            return {"message": "No feedback data available"}
        
        # Analyze patterns
        feedback_types = {}
        common_errors = []
        
        for entry in feedback_data:
            feedback_type = entry.get("feedback_type", "unknown")
            feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
            
            if feedback_type == "incorrect":
                common_errors.append({
                    "original_analysis": entry.get("analysis", {}),
                    "user_correction": entry.get("user_correction", ""),
                    "timestamp": entry.get("timestamp", "")
                })
        
        return {
            "total_feedback": len(feedback_data),
            "feedback_breakdown": feedback_types,
            "accuracy_rate": feedback_types.get("accurate", 0) / len(feedback_data) if feedback_data else 0,
            "common_errors": common_errors[:5],  # Top 5 errors
            "improvement_suggestions": self._generate_improvements(common_errors)
        }
    
    def _load_all_feedback(self) -> List[Dict]:
        """Load all feedback from file"""
        feedback_data = []
        try:
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line))
        except FileNotFoundError:
            pass
        return feedback_data
    
    def _generate_improvements(self, errors: List[Dict]) -> List[str]:
        """Generate improvement suggestions from errors"""
        suggestions = []
        
        if len(errors) > 0:
            suggestions.append("Consider adding more examples to agent prompts")
            suggestions.append("Review geographic scope detection logic")
            suggestions.append("Enhance age sensitivity keyword matching")
        
        return suggestions