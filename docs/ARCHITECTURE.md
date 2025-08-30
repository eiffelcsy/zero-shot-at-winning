# System Architecture
Our project is structured into three major layers: Frontend, Backend, and Data/Knowledge Base.

## 1. Frontend
- Framework: Streamlit
- Responsibilities:
  - Provides the user interface for compliance analysis.
  - Uploads regulation PDFs.
  - Displays detection results, analytics dashboard, and feedback forms.

## 2. Backend
- Framework: FastAPI
- Libraries: LangChain, ChromaDB, custom agents.
- Responsibilities:
  - Hosts compliance analysis endpoints.
  - Orchestrates multi-agent system for feature screening, research, validation, and learning.
  - Integrates retrieval-augmented generation (RAG) for regulation and policy references.

## 3. Data & Knowledge Base
- Stores user contributions, regulatory documents, terminology, and overrides.
- Vector database (Chroma) powers semantic retrieval.

## 4. Deployment
-  Containerization: Docker + Docker Compose
-  Flow:
    - `docker compose build` builds frontend and backend images.
    - `docker compose up` runs containers and orchestrates service communication.

## High Level Diagram 
```bash
+-------------+           +-----------------+           +-------------------+
|   Frontend  | <-------> |     Backend     | <-------> | Knowledge Sources |
| (Streamlit) |   HTTP    |    (FastAPI)    |    RAG    |  (Chroma + Data)  |
+-------------+           +-----------------+           +-------------------+
```

---

# Repository Structure
```bash
└── app
    ├── agents              # Multi-agent system (screening, validation, learning, research)
    │   ├── memory          # Custom memory modules
    │   ├── prompts         # Prompt templates for LLM agents
    │   ├── tools           # Tools for agents
    │   ├── base.py         # Base agent class
    │   ├── orchestrator.py # Orchestration logic for agents
    │   └── state.py        # Agent state management
    ├── api                 # FastAPI routes
    │   └── v1              # API v1 endpoints
    ├── chroma              # ChromaDB connection
    ├── data                # Knowledge base and datasets
    │   ├── kb              # User contributions
    │   ├── memory          # Few-shot examples
    │   ├── tiktok_terminology # Terminology dictionary
    ├── logs                # Logging configuration
    ├── rag                 # Retrieval-Augmented Generation modules
    │   ├── ingestion       # Document ingestion pipeline
    │   ├── retrieval       # Query processing and retrieval
    │   └── tools           # Retrieval tools
    ├── tests               # Backend test suites
    ├── config.py           # Configurations
    └── main.py             # Backend entrypoint
└── docs                    # Documentation
└── frontend                # Streamlit frontend
    ├── static              # UI assets (CSS)
    ├── Dockerfile          # Frontend Dockerfile
    └── ui.py               # Streamlit UI entrypoint
└── tests                   # Unit tests
└── docker-compose.yml      # Service orchestration
└── Dockerfile              # Backend Dockerfile
└── env.example             # Environment variables template
└── requirements.txt        # Python dependencies
```
