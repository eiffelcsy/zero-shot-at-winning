# Shared pytest fixtures for RAG tools tests

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_query_processor():
    """Mock query processor for query enhancement."""
    processor = Mock()
    
    # Mock enhance_query method
    processor.enhance_query.return_value = "age verification minors under 13 COPPA parental consent requirements"
    
    return processor


@pytest.fixture
def mock_retriever():
    """Mock RAG retriever for document retrieval."""
    retriever = Mock()
    
    # Mock retrieve method with raw ChromaDB-style results
    retriever.retrieve.return_value = [
        {
            'id': 'coppa_doc_1',
            'document': 'COPPA requires verifiable parental consent for collection of personal information from children under 13.',
            'metadata': {
                'regulation_code': 'US_COPPA',
                'jurisdiction': 'US',
                'section': 'Section 312.5'
            },
            'distance': 0.15
        },
        {
            'id': 'ca_sb976_doc_1', 
            'document': 'California SB-976 requires social media platforms to disable algorithmic recommendations by default for users under 18.',
            'metadata': {
                'regulation_code': 'CA_SB976',
                'jurisdiction': 'CA',
                'section': 'Section 1'
            },
            'distance': 0.22
        }
    ]
    
    return retriever


@pytest.fixture
def sample_agent_queries():
    """Sample queries that research agents would ask."""
    return {
        "basic_compliance": "What are the age verification requirements for our platform?",
        "feature_specific": "Do we need parental consent for algorithmic recommendations for teens?",
        "jurisdiction_specific": "What are California's requirements for social media features for minors?",
        "data_handling": "What are the data collection restrictions for users under 13?"
    }

