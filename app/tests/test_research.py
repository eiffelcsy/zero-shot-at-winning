#!/usr/bin/env python3
"""
Unit tests for ResearchAgent with mocked RAG components
"""

import unittest
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from datetime import datetime
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()

import sys
from unittest.mock import MagicMock

# Mock the missing modules
sys.modules['chroma'] = MagicMock()
sys.modules['chroma.chroma_connection'] = MagicMock()
sys.modules['chroma.chroma_connection'].get_chroma_client = MagicMock(return_value=MagicMock())


# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.agents.research import ResearchAgent, ResearchOutput

class MockVectorStorage:
    """Mock RAG VectorStorage"""
    def __init__(self, embedding_model="text-embedding-3-large", collection_name="regulation_kb"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embeddings = MockEmbeddings()
        self.collection = MagicMock()

class MockEmbeddings:
    """Mock OpenAI embeddings"""
    def embed_query(self, text):
        """Return mock embedding vector"""
        return [0.1] * 1536  # Standard OpenAI embedding dimension

    def embed_documents(self, texts):
        """Return mock embedding vectors for multiple texts"""
        return [[0.1] * 1536 for _ in texts]

class MockQueryProcessor:
    """Mock RAG QueryProcessor"""
    def __init__(self, llm=None):
        self.llm = llm
    
    async def expand_query(self, query):
        """Return expanded query"""
        return f"{query} regulatory compliance legal requirements"
    
    async def generate_multiple_queries(self, query, count=5):
        """Return multiple query variations"""
        return [f"{query} variation {i}" for i in range(count)]

class MockRAGRetriever:
    """Mock RAG Retriever"""
    def __init__(self, collection):
        self.collection = collection
    
    def retrieve(self, query_embedding, n_results=10, include=None):
        """Return mock retrieved documents"""
        if "age" in str(query_embedding) or n_results > 5:
            return [
                {
                    'id': 'doc1',
                    'document': 'Children under 13 require parental consent for data collection under COPPA regulations.',
                    'metadata': {
                        'regulation_code': 'COPPA',
                        'geo_jurisdiction': 'US',
                        'regulation_name': 'Children\'s Online Privacy Protection Act',
                        'section': 'Section 1304'
                    },
                    'distance': 0.1
                },
                {
                    'id': 'doc2', 
                    'document': 'California requires default privacy settings for users under 18 per SB-976.',
                    'metadata': {
                        'regulation_code': 'CA_SB976',
                        'geo_jurisdiction': 'California',
                        'regulation_name': 'California Age-Appropriate Design Code',
                        'section': 'Default Settings'
                    },
                    'distance': 0.15
                }
            ]
        else:
            return [
                {
                    'id': 'doc3',
                    'document': 'General data protection requirements for user privacy.',
                    'metadata': {
                        'regulation_code': 'GDPR',
                        'geo_jurisdiction': 'EU',
                        'regulation_name': 'General Data Protection Regulation',
                        'section': 'Article 6'
                    },
                    'distance': 0.3
                }
            ]
    
    def retrieve_with_metadata_filter(self, query_embedding, metadata_filter, n_results=10, include=None):
        """Return filtered mock documents"""
        # Simulate filtering by jurisdiction
        if 'geo_jurisdiction' in metadata_filter:
            jurisdictions = metadata_filter['geo_jurisdiction'].get('$in', [])
            if 'US' in jurisdictions or 'California' in jurisdictions:
                return self.retrieve(query_embedding, n_results, include)[:2]  # US results
            elif 'EU' in jurisdictions:
                return self.retrieve(query_embedding, n_results, include)[-1:]  # EU results
        
        return self.retrieve(query_embedding, n_results, include)

class TestResearchAgent(unittest.TestCase):
    
    @patch('app.rag.ingestion.vector_storage.VectorStorage', MockVectorStorage)
    @patch('app.rag.retrieval.query_processor.QueryProcessor', MockQueryProcessor)
    @patch('app.rag.retrieval.retriever.RAGRetriever', MockRAGRetriever)
    def setUp(self):
        """Setup test fixtures with mocked RAG components"""
        self.agent = ResearchAgent()
    
    def test_init_with_memory_overlay(self):
        """Test agent initialization with memory overlay"""
        with patch('app.rag.ingestion.vector_storage.VectorStorage', MockVectorStorage), \
             patch('app.rag.retrieval.query_processor.QueryProcessor', MockQueryProcessor), \
             patch('app.rag.retrieval.retriever.RAGRetriever', MockRAGRetriever):
            
            memory = "TEST RESEARCH MEMORY"
            agent = ResearchAgent(memory_overlay=memory)
            
            self.assertEqual(agent.memory_overlay, memory)
            self.assertEqual(agent.name, "ResearchAgent")
    
    def test_build_search_query(self):
        """Test search query building from screening analysis"""
        feature_description = "Location sharing for teens"
        screening_analysis = {
            "age_sensitivity": True,
            "data_sensitivity": "T5", 
            "compliance_required": True,
            "geographic_scope": ["US", "California"]
        }
        trigger_keywords = ["minors", "location", "privacy"]
        
        query = self.agent._build_search_query(feature_description, screening_analysis, trigger_keywords)
        
        self.assertIn("Location sharing for teens", query)
        self.assertIn("children minors age verification", query)
        self.assertIn("personal data privacy protection", query)
        self.assertIn("regulatory compliance legal requirements", query)
        self.assertIn("US", query)
        self.assertIn("California", query)
        self.assertIn("minors", query)
    
    def test_extract_candidates(self):
        """Test candidate extraction from retrieved documents"""
        mock_documents = [
            {
                'metadata': {'regulation_code': 'COPPA', 'geo_jurisdiction': 'US'},
                'distance': 0.1
            },
            {
                'metadata': {'regulation_code': 'CA_SB976', 'geo_jurisdiction': 'California'},
                'distance': 0.15
            }
        ]
        
        candidates = self.agent._extract_candidates(mock_documents)
        
        self.assertEqual(len(candidates), 2)
        
        coppa_candidate = next(c for c in candidates if c["reg"] == "COPPA")
        self.assertIn("relevance score", coppa_candidate["why"])
        self.assertAlmostEqual(coppa_candidate["score"], 0.9, places=1)
        
        sb976_candidate = next(c for c in candidates if c["reg"] == "CA_SB976")
        self.assertIn("relevance score", sb976_candidate["why"])
        self.assertAlmostEqual(sb976_candidate["score"], 0.85, places=1)
    
    def test_format_evidence(self):
        """Test evidence formatting from retrieved documents"""
        mock_documents = [
            {
                'id': 'doc1',
                'document': 'Test regulation content about child privacy protection.',
                'metadata': {
                    'regulation_code': 'COPPA',
                    'geo_jurisdiction': 'US',
                    'regulation_name': 'Children\'s Online Privacy Protection Act',
                    'section': 'Section 1304'
                },
                'distance': 0.1
            }
        ]
        
        evidence = self.agent._format_evidence(mock_documents)
        
        self.assertEqual(len(evidence), 1)
        evidence_item = evidence[0]
        
        self.assertEqual(evidence_item["reg"], "COPPA")
        self.assertEqual(evidence_item["jurisdiction"], "US")
        self.assertEqual(evidence_item["name"], "Children's Online Privacy Protection Act")
        self.assertEqual(evidence_item["section"], "Section 1304")
        self.assertIn("child privacy protection", evidence_item["excerpt"])
        self.assertGreater(evidence_item["score"], 8.0)
    
    def test_calculate_confidence(self):
        """Test confidence calculation from retrieval quality"""
        # High quality documents (low distance)
        high_quality_docs = [
            {'distance': 0.1, 'metadata': {'geo_jurisdiction': 'US'}},
            {'distance': 0.15, 'metadata': {'geo_jurisdiction': 'California'}}
        ]
        
        screening_analysis = {"geographic_scope": ["US", "California"]}
        
        confidence = self.agent._calculate_confidence(high_quality_docs, screening_analysis)
        self.assertGreater(confidence, 0.8)
        
        # Lower quality documents (higher distance)
        low_quality_docs = [
            {'distance': 0.8, 'metadata': {'geo_jurisdiction': 'Unknown'}},
            {'distance': 0.9, 'metadata': {'geo_jurisdiction': 'Unknown'}}
        ]
        
        confidence = self.agent._calculate_confidence(low_quality_docs, screening_analysis)
        self.assertLess(confidence, 0.5)
    
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    async def test_process_success_high_risk(self, mock_llm):
        """Test successful processing of high-risk research"""
        # Mock LLM response
        mock_llm.return_value = {
            "agent": "ResearchAgent",
            "candidates": [
                {"reg": "COPPA", "why": "Applies to US minors data collection", "score": 0.9}
            ],
            "evidence": [
                {
                    "reg": "COPPA",
                    "jurisdiction": "US", 
                    "name": "Children's Online Privacy Protection Act",
                    "section": "Section 1304",
                    "url": "https://www.coppa.gov",
                    "excerpt": "Requires parental consent for children under 13",
                    "score": 9.2
                }
            ],
            "query_used": "minors age verification regulatory compliance legal requirements",
            "confidence_score": 0.88
        }
        
        # Test state with high-risk screening
        state = {
            "feature_name": "Teen Location Sharing",
            "feature_description": "Location sharing feature for users under 18",
            "screening_analysis": {
                "agent": "ScreeningAgent",
                "risk_level": "HIGH",
                "compliance_required": True,
                "age_sensitivity": True,
                "data_sensitivity": "T5",
                "geographic_scope": ["US", "California"],
                "trigger_keywords": ["minors", "location", "T5"]
            }
        }
        
        # Execute
        result = await self.agent.process(state)
        
        # Assertions
        self.assertTrue(result["research_completed"])
        self.assertEqual(result["next_step"], "validation")
        
        analysis = result["research_analysis"]
        self.assertEqual(analysis["agent"], "ResearchAgent")
        self.assertGreater(len(analysis["candidates"]), 0)
        self.assertGreater(len(analysis["evidence"]), 0)
        self.assertGreater(analysis["confidence_score"], 0.0)
    
    async def test_process_missing_screening(self):
        """Test error handling when screening analysis is missing"""
        state = {
            "feature_name": "Test Feature",
            "feature_description": "Test description"
            # Missing screening_analysis
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["research_completed"])
        self.assertEqual(result["next_step"], "validation")
        self.assertIn("research_error", result)
        self.assertIn("Missing screening analysis", result["research_error"])
    
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    async def test_process_llm_failure(self, mock_llm):
        """Test error handling when LLM call fails"""
        mock_llm.side_effect = Exception("LLM API Error")
        
        state = {
            "feature_description": "Test feature",
            "screening_analysis": {
                "risk_level": "MEDIUM",
                "geographic_scope": ["US"],
                "trigger_keywords": ["test"],
                "age_sensitivity": False,
                "data_sensitivity": "T3"
            }
        }
        
        result = await self.agent.process(state)
        
        self.assertTrue(result["research_completed"])
        analysis = result["research_analysis"] 
        self.assertEqual(analysis["confidence_score"], 0.0)
        self.assertIn("LLM API Error", analysis["error"])
    
    @patch('app.agents.research.ResearchAgent._expand_query')
    async def test_query_expansion(self, mock_expand):
        """Test query expansion functionality"""
        mock_expand.return_value = "expanded query with regulatory terms"
        
        state = {
            "feature_description": "basic feature",
            "screening_analysis": {
                "geographic_scope": ["US"],
                "trigger_keywords": ["data"],
                "age_sensitivity": False,
                "data_sensitivity": "T3"
            }
        }
        
        await self.agent.process(state)
        
        # Verify query expansion was called
        mock_expand.assert_called_once()

class AsyncTestCase(unittest.TestCase):
    """Base class for async test methods"""
    
    def run_async(self, coro):
        """Helper to run async tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Sync versions of async tests for unittest compatibility
class TestResearchAgentSync(AsyncTestCase):
    
    @patch('app.rag.ingestion.vector_storage.VectorStorage', MockVectorStorage)
    @patch('app.rag.retrieval.query_processor.QueryProcessor', MockQueryProcessor)
    @patch('app.rag.retrieval.retriever.RAGRetriever', MockRAGRetriever)
    def setUp(self):
        """Setup with mocked RAG components"""
        self.agent = ResearchAgent()
    
    @patch('app.agents.research.ResearchAgent.safe_llm_call')
    def test_process_success_sync(self, mock_llm):
        """Sync version of successful process test"""
        mock_llm.return_value = {
            "agent": "ResearchAgent",
            "candidates": [{"reg": "COPPA", "why": "US minors", "score": 0.9}],
            "evidence": [{
                "reg": "COPPA",
                "jurisdiction": "US",
                "name": "COPPA",
                "section": "Section 1304", 
                "url": "https://coppa.gov",
                "excerpt": "Parental consent required",
                "score": 9.0
            }],
            "query_used": "minors regulatory compliance legal requirements",
            "confidence_score": 0.85
        }
        
        state = {
            "feature_description": "Test feature for minors",
            "screening_analysis": {
                "risk_level": "HIGH",
                "age_sensitivity": True,
                "data_sensitivity": "T5",
                "geographic_scope": ["US"],
                "trigger_keywords": ["minors"],
                "compliance_required": True
            }
        }
        
        result = self.run_async(self.agent.process(state))
        
        self.assertTrue(result["research_completed"])
        self.assertGreater(len(result["research_evidence"]), 0)
        self.assertGreater(len(result["research_candidates"]), 0)

if __name__ == '__main__':
    # Run with: python -m pytest test_research_agent.py -v
    unittest.main()