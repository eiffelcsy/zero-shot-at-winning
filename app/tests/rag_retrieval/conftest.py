"""
Shared pytest fixtures for RAG retrieval tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection."""
    collection = Mock()
    collection.query.return_value = {
        'ids': [['doc_1', 'doc_2']],
        'documents': [['Sample document 1', 'Sample document 2']],
        'metadatas': [[{'source': 'doc1.pdf'}, {'source': 'doc2.pdf'}]],
        'distances': [[0.1, 0.3]]
    }
    return collection


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings instance."""
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_embeddings.embed_documents.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0]
    ]
    return mock_embeddings


@pytest.fixture
def mock_llm():
    """Mock LLM for query enhancement and processing."""
    llm = AsyncMock()
    llm.apredict.return_value = "Enhanced query with additional context and specificity."
    return llm


@pytest.fixture
def sample_queries():
    """Sample queries for testing query processing."""
    return {
        "basic_compliance": "What are the COPPA requirements for children's data?",
        "short_keyword": "GDPR compliance",
        "typo_laden": "Wat are the COPA requirments for childrens data?",
        "special_chars": "What's the §2258A reporting requirement?!@#$%",
        "multilingual": "¿Cuáles son los requisitos de GDPR para menores?",
        "empty": "",
        "whitespace_only": "   \n\t   "
    }
