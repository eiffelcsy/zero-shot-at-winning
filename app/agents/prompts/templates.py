from langchain.prompts import PromptTemplate

# Internal TikTok context mapping
TIKTOK_CONTEXT = """
INTERNAL TIKTOK TERMINOLOGY:
- ASL: Age-sensitive logic (age verification/restrictions)
- GH: Geo-handler (region-based routing and enforcement)
- CDS: Compliance Detection System  
- T5: Tier 5 data (highest sensitivity level - more critical than T1-T4)
- Jellybean: Internal parental control system
- Snowcap: Child safety policy framework
- Spanner: Rule engine (not Google Spanner)
- EchoTrace: Log tracing mode for compliance verification
- ShadowMode: Deploy feature without user impact for analytics
- Redline: Flag for legal review (not financial loss)
- Softblock: Silent user limitation without notifications
- Glow: Compliance-flagging status for geo-based alerts
- NSP: Non-shareable policy (content restrictions)
- DRT: Data retention threshold
- LCP: Local compliance policy
- IMT: Internal monitoring trigger
- BB: Baseline Behavior (standard user behavior for anomaly detection)
- PF: Personalized feed
- FR: Feature rollout status
- NR: Not recommended (restriction level)
"""

BASE_COMPLIANCE_PROMPT = PromptTemplate(
    input_variables=["context", "feature_description", "analysis_type"],
    template="""
You are a TikTok geo-regulation compliance expert. Analyze this feature for regulatory requirements.

{context}

FEATURE TO ANALYZE:
{feature_description}

ANALYSIS TYPE: {analysis_type}

Provide your analysis in valid JSON format only.
"""
)