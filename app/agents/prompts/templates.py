from langchain.prompts import PromptTemplate

# Internal TikTok context mapping for all agents
TIKTOK_CONTEXT = """
INTERNAL TIKTOK TERMINOLOGY (Provides information about what each term/acronym means, critical for analysis):
- NR: Not recommended (restriction/limitation level)
- PF: Personalized feed (recommendation algorithm)
- GH: Geo-handler (region-based routing and enforcement)
- CDS: Compliance Detection System (automated compliance monitoring)
- DRT: Data retention threshold (how long data can be stored)
- LCP: Local compliance policy (region-specific rules)
- Redline: Flag for legal review (not financial loss context)
- Softblock: Silent user limitation without notifications
- Spanner: Rule engine (not Google Spanner database)
- ShadowMode: Deploy feature without user impact for analytics collection
- T5: Tier 5 data (highest sensitivity level - more critical than T1-T4)
- ASL: Age-sensitive logic (age verification/restrictions for minors)
- Glow: Compliance-flagging status for geo-based alerts
- NSP: Non-shareable policy (content sharing restrictions)
- Jellybean: Internal parental control system
- EchoTrace: Log tracing mode for compliance verification
- BB: Baseline Behavior (standard user behavior for anomaly detection)
- Snowcap: Child safety policy framework
- FR: Feature rollout status
- IMT: Internal monitoring trigger
"""

# Screening agent prompt template
SCREENING_PROMPT_TEMPLATE = TIKTOK_CONTEXT + """

FEATURE TO ANALYZE: {feature_description}

ADDITIONAL CONTEXT DOCUMENTS: {context_documents}

ANALYSIS FRAMEWORK:
Analyze this feature for potential regulatory requirements across multiple jurisdictions.

KEY COMPLIANCE PATTERNS TO EVALUATE:
1. **Data Protection & Privacy**: Personal data collection, processing, retention
2. **Age Restrictions & Child Safety**: Minor protection, parental controls, age verification  
3. **Content Governance**: Moderation obligations, transparency requirements
4. **Geographic Enforcement**: Location-based restrictions, jurisdiction-specific behaviors
5. **Platform Responsibilities**: Regulatory reporting, user safety obligations

DETECTION CRITERIA:
- Age-sensitive functionality (ASL) or data processing involving minors
- Geographic targeting (GH) or location-aware enforcement mechanisms  
- Personal data handling (T5) including collection, processing, or retention
- Content control mechanisms including filtering, blocking, or moderation

Return ONLY valid JSON matching this schema:
{{
    "agent": "ScreeningAgent",
    "risk_level": "LOW|MEDIUM|HIGH",
    "compliance_required": true/false,
    "confidence": 0.0-1.0,
    "trigger_keywords": ["keyword1", "keyword2"],
    "reasoning": "detailed explanation",
    "needs_research": true/false,
    "geographic_scope": ["region1", "region2"] or "global" or "unknown",
    "age_sensitivity": true/false,
    "data_sensitivity": "T5|T4|T3|T2|T1|none"
}}
"""

# Research agent prompt template
RESEARCH_PROMPT_TEMPLATE = TIKTOK_CONTEXT + """

SCREENING ANALYSIS FROM PREVIOUS AGENT:
{screening_analysis}

KNOWLEDGE BASE EVIDENCE FOUND:
{evidence_found}

TASK: Based on the screening analysis and knowledge base evidence, identify the most applicable regulations and provide your research assessment.

Focus on:
- Regulations that match the geographic scope identified
- Laws addressing the compliance patterns flagged  
- Standards relevant to the data sensitivity level
- Requirements triggered by age sensitivity or other risk factors

Return ONLY valid JSON:
{{
    "agent": "ResearchAgent",
    "candidates": [
        {{"reg": "regulation_code", "why": "reason_for_selection", "score": 0.85}}
    ],
    "evidence": [
        {{
            "reg": "regulation_code",
            "jurisdiction": "jurisdiction_name", 
            "name": "regulation_name",
            "section": "section_reference",
            "url": "regulation_url",
            "excerpt": "relevant_text_snippet",
            "score": 8.5
        }}
    ],
    "query_used": "search_query_constructed",
    "confidence_score": 0.85
}}
"""

# Create PromptTemplate objects
SCREENING_PROMPT = PromptTemplate(
    input_variables=["feature_name"", feature_description", "context_documents"],
    template=SCREENING_PROMPT_TEMPLATE
)

RESEARCH_PROMPT = PromptTemplate(
    input_variables=["screening_analysis", "evidence_found"],
    template=RESEARCH_PROMPT_TEMPLATE
)

# Validation prompt template (for future use)
VALIDATION_PROMPT_TEMPLATE = TIKTOK_CONTEXT + """

SCREENING ANALYSIS: {screening_result}
RESEARCH FINDINGS: {research_result}

VALIDATION TASK: Cross-verify the compliance assessment for consistency and accuracy.

Return ONLY valid JSON:
{{
    "validation_status": "CONFIRMED|DISPUTED|UNCLEAR",
    "final_risk_level": "LOW|MEDIUM|HIGH", 
    "final_compliance_required": true/false,
    "confidence": 0.0-1.0,
    "conflicts_found": ["conflict1", "conflict2"],
    "reasoning": "validation analysis and recommendations"
}}
"""

VALIDATION_PROMPT = PromptTemplate(
    input_variables=["screening_result", "research_result"],
    template=VALIDATION_PROMPT_TEMPLATE
)

# Common output format schema for reference
COMPLIANCE_OUTPUT_SCHEMA = {
    "risk_level": "LOW|MEDIUM|HIGH",
    "compliance_required": "boolean",
    "confidence": "float 0.0-1.0",
    "reasoning": "string",
    "applicable_regulations": "list of strings",
    "geographic_scope": "list of strings or 'global'",
    "needs_human_review": "boolean",
    "compliance_patterns": "list of compliance categories detected"
}