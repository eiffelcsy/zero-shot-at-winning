import pytest

from app.agents.prompts.templates import (
    SCREENING_PROMPT, 
    TIKTOK_CONTEXT,
    BASE_COMPLIANCE_PROMPT,
    COMPLIANCE_OUTPUT_SCHEMA,
    RESEARCH_PROMPT,
    RESEARCH_OUTPUT_SCHEMA
)

class TestPromptTemplates:
    
    def test_screening_prompt_formatting(self):
        """Test that screening prompt formats correctly with sample data"""
        feature_desc = "ASL-based curfew restrictions for Utah minors using GH"
        
        formatted_prompt = SCREENING_PROMPT.format(feature_description=feature_desc)
        
        assert isinstance(formatted_prompt, str)
        assert len(formatted_prompt) > 100  # Should be substantial
        assert feature_desc in formatted_prompt
        assert "JSON" in formatted_prompt
    
    @pytest.mark.parametrize("feature_description", [
        "Simple feature test",
        "ASL and GH for California users with T5 data",
        "Feature with Jellybean controls and Snowcap policies",
        "Complex feature with multiple regulatory indicators"
    ])
    def test_prompt_variable_substitution(self, feature_description):
        """Test prompt formatting with various inputs"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description=feature_description)
        
        assert feature_description in formatted_prompt
        assert len(formatted_prompt) > len(feature_description)  # Should add template content
    
    def test_prompt_contains_tiktok_context(self):
        """Test that prompts contain TikTok internal terminology"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description="test")
        
        # Should contain key TikTok terms
        tiktok_terms = ["ASL:", "GH:", "T5:", "Jellybean", "Snowcap"]
        found_terms = sum(1 for term in tiktok_terms if term in formatted_prompt)
        
        assert found_terms >= 3, f"Should contain at least 3 TikTok terms, found {found_terms}"
    
    def test_tiktok_context_completeness(self):
        """Test that TikTok context contains all required terms"""
        expected_terms = [
            "ASL", "GH", "CDS", "T5", "Jellybean", 
            "Snowcap", "EchoTrace", "ShadowMode", "Redline"
        ]
        
        for term in expected_terms:
            assert term in TIKTOK_CONTEXT, f"Missing TikTok term: {term}"
    
    def test_output_schema_present(self):
        """Test that the prompt specifies the required output schema"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description="test")
        
        required_fields = [
            "risk_level", "compliance_required", "confidence", 
            "reasoning", "trigger_keywords", "geographic_scope"
        ]
        
        for field in required_fields:
            assert field in formatted_prompt, f"Missing required field: {field}"
    
    def test_prompt_input_variables(self):
        """Test that prompt has correct input variables"""
        assert SCREENING_PROMPT.input_variables == ["feature_description"]
    
    def test_base_compliance_prompt_variables(self):
        """Test base compliance prompt input variables"""
        expected_vars = ["context", "feature_description", "analysis_type"]
        assert set(BASE_COMPLIANCE_PROMPT.input_variables) == set(expected_vars)
    
    def test_compliance_output_schema_structure(self):
        """Test that compliance output schema has required fields"""
        required_keys = [
            "risk_level", "compliance_required", "confidence", 
            "reasoning", "applicable_regulations"
        ]
        
        for key in required_keys:
            assert key in COMPLIANCE_OUTPUT_SCHEMA, f"Missing schema field: {key}"
    
    @pytest.mark.parametrize("risk_level", ["LOW", "MEDIUM", "HIGH"])
    def test_risk_level_options_in_prompt(self, risk_level):
        """Test that all risk levels are mentioned in prompt"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description="test")
        assert risk_level in formatted_prompt
    
    def test_prompt_contains_regulatory_concepts(self):
        """Test that prompt references regulatory concepts (FLEXIBLE VERSION)"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description="test")
        
        # Look for regulatory CONCEPTS, not specific regulation names
        regulatory_concepts = [
            "data protection", "privacy", "age restrictions", "child safety",
            "content governance", "moderation", "geographic enforcement", 
            "compliance", "regulatory", "jurisdiction", "platform responsibilities"
        ]
        
        found_concepts = sum(1 for concept in regulatory_concepts 
                            if concept.lower() in formatted_prompt.lower())
        
        assert found_concepts >= 5, f"Should reference multiple regulatory concepts, found {found_concepts}"
    
    def test_prompt_contains_compliance_patterns(self):
        """Test that prompt includes key compliance pattern categories"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description="test")
        
        compliance_patterns = [
            "Data Protection", "Age Restrictions", "Content Governance", 
            "Geographic Enforcement", "Platform Responsibilities"
        ]
        
        found_patterns = sum(1 for pattern in compliance_patterns 
                            if pattern in formatted_prompt)
        
        assert found_patterns >= 4, f"Should include major compliance patterns, found {found_patterns}"
    
    def test_prompt_encourages_pattern_recognition(self):
        """Test that prompt focuses on recognizing compliance patterns rather than specific laws"""
        formatted_prompt = SCREENING_PROMPT.format(feature_description="test")
        
        pattern_keywords = [
            "patterns", "compliance patterns", "regulatory patterns", 
            "framework", "analysis", "evaluate", "assess"
        ]
        
        found_keywords = sum(1 for keyword in pattern_keywords 
                            if keyword.lower() in formatted_prompt.lower())
        
        assert found_keywords >= 3, "Should emphasize pattern recognition approach"

    def test_research_prompt_formatting(self):
        """Test that research prompt formats correctly with sample data"""
        feature_name = "Test Feature"
        feature_description = "A test feature for compliance screening"
        screening_analysis = "Supplier X flagged for AML risks in APAC"
        evidence_found = "No documents retrieved in this test."
        
        formatted_prompt = RESEARCH_PROMPT.format(
            feature_name=feature_name,
            feature_description=feature_description,
            screening_analysis=screening_analysis,
            evidence_found=evidence_found
        )
        
        assert isinstance(formatted_prompt, str)
        assert len(formatted_prompt) > 100  # Should be substantial
        assert feature_name in formatted_prompt
        assert feature_description in formatted_prompt
        assert screening_analysis in formatted_prompt
        assert evidence_found in formatted_prompt
        assert "JSON" in formatted_prompt
    
    @pytest.mark.parametrize("screening_analysis", [
        "Supplier Y under investigation for child labor in Africa",
        "Supplier Z flagged for T5 data processing in EU",
        "Entity involved in multiple geographic jurisdictions",
        "High-risk supplier with unclear compliance requirements"
    ])
    def test_prompt_variable_substitution(self, screening_analysis):
        """Test research prompt formatting with various inputs"""
        formatted_prompt = RESEARCH_PROMPT.format(
            feature_name="Test Feature",
            feature_description="Test description",
            screening_analysis=screening_analysis,
            evidence_found="placeholder evidence"
        )
        
        assert screening_analysis in formatted_prompt
        assert len(formatted_prompt) > len(screening_analysis)
    
    def test_output_schema_present(self):
        """Test that the research prompt specifies the required output schema"""
        formatted_prompt = RESEARCH_PROMPT.format(
            feature_name="test",
            feature_description="test description",
            screening_analysis="test",
            evidence_found="test evidence"
        )
        
        required_fields = [
            "agent", "regulations",
            "query_used", "confidence_score"
        ]
        
        for field in required_fields:
            assert field in formatted_prompt, f"Missing required field: {field}"
    
    def test_prompt_input_variables(self):
        """Test that prompt has correct input variables"""
        assert set(RESEARCH_PROMPT.input_variables) == {"feature_name", "feature_description", "screening_analysis", "evidence_found"}
    
    def test_research_output_schema_structure(self):
        """Test that the research output schema has required fields"""
        required_keys = [
            "agent", "regulations", 
            "queries_used", "confidence_score"
        ]
        
        for key in required_keys:
            assert key in RESEARCH_OUTPUT_SCHEMA, f"Missing schema field: {key}"
    
    def test_prompt_references_regulatory_concepts(self):
        """Test that research prompt references regulatory/compliance concepts"""
        formatted_prompt = RESEARCH_PROMPT.format(
            feature_name="Test Feature",
            feature_description="Test description",
            screening_analysis="Supplier X AML risk",
            evidence_found="Evidence sample"
        )
        
        regulatory_concepts = [
            "regulation", "compliance", "jurisdiction", 
            "governance", "legal", "obligations"
        ]
        
        found_concepts = sum(1 for concept in regulatory_concepts 
                            if concept.lower() in formatted_prompt.lower())
        
        assert found_concepts >= 3, f"Should reference multiple regulatory concepts, found {found_concepts}"
    
    def test_prompt_emphasizes_evidence_synthesis(self):
        """Test that research prompt emphasizes evidence-based reasoning"""
        formatted_prompt = RESEARCH_PROMPT.format(
            feature_name="Test Feature",
            feature_description="Test description",
            screening_analysis="Supplier X flagged",
            evidence_found="Sample evidence"
        )
        
        keywords = [
            "evidence", "support", "justify", 
            "sources", "context", "retrieved"
        ]
        
        found_keywords = sum(1 for kw in keywords if kw.lower() in formatted_prompt.lower())
        
        assert found_keywords >= 3, "Should emphasize evidence-based synthesis"