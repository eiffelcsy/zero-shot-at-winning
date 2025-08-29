# app/tests/smoke_validation_real.py
import sys, os, asyncio, json
from pathlib import Path

# Put project root on sys.path BEFORE importing app.*
ROOT = Path(__file__).resolve().parents[2]   # .../zero-shot-at-winning
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()  # optional

from app.agents.validation import ValidationAgent

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

async def main():
    agent = ValidationAgent(memory_overlay="")  # keep empty to avoid DB
    state = {
        "feature_name": "Teen Location Sharing",
        "feature_description": "Under-18 users can share precise location.",
        "screening_analysis": {"risk_level": "HIGH", "compliance_required": True},
        "research_evidence": [
            {
                "reg": "COPPA",
                "jurisdiction": "US",
                "name": "Children's Online Privacy Protection Rule",
                "section": "16 CFR Part 312",
                "url": "https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa",
                "excerpt": "Parental consent required.",
                "score": 8.9
            }
        ]
    }
    res = await agent.process(state)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    # requires: OPENAI_API_KEY in env
    asyncio.run(main())
