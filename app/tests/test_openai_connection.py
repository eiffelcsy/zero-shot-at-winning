import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def test_openai_connection():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment variables")
        return False
    
    print(f"🔑 API key loaded: {api_key[:8]}...")
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([{"role": "user", "content": "Hello, world!"}])
        print("✅ OpenAI API connection successful!")
        print(f"Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()