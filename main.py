import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Your existing agent imports
from app.agents.screening import ScreeningAgent

# Load environment variables from .env file
load_dotenv()

async def main():
    """Main application entry point"""
    
    print("🚀 Initializing OpenAI LLM...")
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cheaper alternative: gpt-4o-mini
        temperature=0.0,      # For consistent compliance analysis
        max_tokens=1000
    )
    
    print("🤖 Creating ScreeningAgent...")
    
    # Create agents with real LLM
    screening_agent = ScreeningAgent(llm)
    
    # Test with a real TikTok feature
    test_feature = {
        "feature_name": "Utah Curfew Controls",
        "feature_description": "ASL-based curfew restrictions for Utah minors using GH with T5 data processing and Jellybean parental controls"
    }
    
    print("🔍 Analyzing feature for compliance...")
    print(f"Feature: {test_feature['feature_name']}")
    print(f"Description: {test_feature['feature_description']}")
    print("-" * 50)
    
    # Run the analysis
    result = await screening_agent.process(test_feature)
    
    # Display results
    print("📊 **COMPLIANCE ANALYSIS RESULTS:**")
    print(f"🎯 Risk Level: {result['analysis']['risk_level']}")
    print(f"✅ Compliance Required: {result['analysis']['compliance_required']}")
    print(f"🔢 Confidence Score: {result['analysis']['confidence']}")
    print(f"➡️ Next Step: {result['next_step']}")
    print(f"💭 AI Reasoning: {result['analysis']['reasoning']}")
    
    if 'trigger_keywords' in result['analysis']:
        print(f"🔑 Detected Keywords: {result['analysis']['trigger_keywords']}")
    
    if 'geographic_scope' in result['analysis']:
        print(f"🌍 Geographic Scope: {result['analysis']['geographic_scope']}")

if __name__ == "__main__":
    asyncio.run(main())