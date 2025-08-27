from langchain.prompts import PromptTemplate

# Internal TikTok context mapping for all agents
TIKTOK_CONTEXT = """
INTERNAL TIKTOK TERMINOLOGY (Critical for Analysis):
- ASL: Age-sensitive logic (age verification/restrictions for minors)
- GH: Geo-handler (region-based routing and enforcement)
- CDS: Compliance Detection System (automated compliance monitoring)
- T5: Tier 5 data (highest sensitivity level - more critical than T1-T4)
- Jellybean: Internal parental control system
- Snowcap: Child safety policy framework
- Spanner: Rule engine (not Google Spanner database)
- EchoTrace: Log tracing mode for compliance verification
- ShadowMode: Deploy feature without user impact for analytics collection
- Redline: Flag for legal review (not financial loss context)
- Softblock: Silent user limitation without notifications
- Glow: Compliance-flagging status for geo-based alerts
- NSP: Non-shareable policy (content sharing restrictions)
- DRT: Data retention threshold (how long data can be stored)
- LCP: Local compliance policy (region-specific rules)
- IMT: Internal monitoring trigger
- BB: Baseline Behavior (standard user behavior for anomaly detection)
- PF: Personalized feed (recommendation algorithm)
- FR: Feature rollout status
- NR: Not recommended (restriction/limitation level)
"""

# Base compliance analysis prompt template
BASE_COMPLIANCE_PROMPT = PromptTemplate(
    input_variables=["context", "feature_description", "analysis_type"],
    template="""
You are a global regulatory compliance expert analyzing features for compliance requirements.

{context}

FEATURE TO ANALYZE:
{feature_description}

ANALYSIS TYPE: {analysis_type}

Consider regulatory patterns across multiple jurisdictions and compliance domains.

Provide analysis in valid JSON format only.
"""
)

# Generalized screening prompt focusing on compliance patterns
SCREENING_PROMPT = PromptTemplate(
    input_variables=["feature_description"],
    template=TIKTOK_CONTEXT + """

FEATURE TO ANALYZE: {feature_description}

REGULATORY COMPLIANCE ANALYSIS FRAMEWORK:
Analyze this feature for potential regulatory requirements across multiple jurisdictions and compliance domains.

KEY COMPLIANCE PATTERNS TO EVALUATE:
1. **Data Protection & Privacy**: Personal data collection, processing, retention, cross-border transfers, user consent
2. **Age Restrictions & Child Safety**: Minor protection, parental controls, age verification, content filtering for children
3. **Content Governance**: Moderation obligations, transparency requirements, algorithmic accountability, user reporting
4. **Geographic Enforcement**: Location-based restrictions, jurisdiction-specific behaviors, regional compliance variations
5. **Platform Responsibilities**: Regulatory reporting, user safety obligations, accessibility requirements

DETECTION CRITERIA:
- Age-sensitive functionality (ASL) or data processing involving minors
- Geographic targeting (GH) or location-aware enforcement mechanisms
- Personal data handling (T5) including collection, processing, or retention
- Content control mechanisms including filtering, blocking, or moderation
- Compliance indicators such as legal terminology, regulatory references, or policy enforcement

RISK ASSESSMENT GUIDELINES:
- **HIGH RISK**: Clear regulatory compliance required (combines multiple patterns like age+location+data)
- **MEDIUM RISK**: Potential compliance needs (one or two patterns present)
- **LOW RISK**: Business functionality with minimal regulatory implications

Return ONLY valid JSON in this exact format:
{{{{
    "risk_level": "LOW|MEDIUM|HIGH",
    "compliance_required": true/false,
    "confidence": 0.0-1.0,
    "trigger_keywords": ["keyword1", "keyword2"],
    "regulatory_indicators": ["ASL", "GH", "T5", "other compliance signals"],
    "reasoning": "detailed explanation of regulatory analysis",
    "needs_research": true/false,
    "geographic_scope": ["region1", "region2"] or "global" or "unknown",
    "age_sensitivity": true/false,
    "data_sensitivity": "T5|T4|T3|T2|T1|none",
    "compliance_patterns": ["data_protection", "age_restrictions", "content_governance"]
}}}}
"""
)

# Research agent prompt (for future implementation)
RESEARCH_PROMPT = PromptTemplate(
    input_variables=["feature_analysis", "regulation_context"],
    template=f"""
{TIKTOK_CONTEXT}

SCREENING RESULTS: {{feature_analysis}}

REGULATORY CONTEXT: {{regulation_context}}

RESEARCH ANALYSIS: Match the feature against applicable regulatory frameworks and compliance requirements.

Return ONLY valid JSON:
{{{{
    "applicable_regulations": ["regulation1", "regulation2"],
    "compliance_requirements": ["requirement1", "requirement2"],
    "jurisdictions_affected": ["jurisdiction1", "jurisdiction2"],
    "confidence": 0.0-1.0,
    "reasoning": "detailed regulatory research analysis"
}}}}
"""
)

# Validation agent prompt (for future implementation)  
VALIDATION_PROMPT = PromptTemplate(
    input_variables=["screening_result", "research_result"],
    template=f"""
{TIKTOK_CONTEXT}

SCREENING ANALYSIS: {{screening_result}}
RESEARCH FINDINGS: {{research_result}}

VALIDATION TASK: Cross-verify the compliance assessment for consistency and accuracy.

Return ONLY valid JSON:
{{{{
    "validation_status": "CONFIRMED|DISPUTED|UNCLEAR",
    "final_risk_level": "LOW|MEDIUM|HIGH", 
    "final_compliance_required": true/false,
    "confidence": 0.0-1.0,
    "conflicts_found": ["conflict1", "conflict2"],
    "reasoning": "validation analysis and recommendations"
}}}}
"""
)

# Common output format for consistency
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