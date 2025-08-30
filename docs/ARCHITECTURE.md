# System Architecture
Our project is structured into two main layers: Frontend and Backend.

## 1. Frontend
- Framework: Streamlit
- Responsibilities:
  - Provides the user interface for interaction.
  - Sends requests to the backend for URL and text phishing detection.
  - Displays detection results and recommendations.

## 2. Backend
- Framework: FastAPI
- Libraries Used:
  - LangChain for prompt management and reasoning.
- Responsibilities:
  - Processes incoming requests from the frontend.
  - Runs phishing detection using LLMs and rules.
  - Returns structured responses (risk level, explanation, recommendation).

## 3. Deployment
-  Containerization: Docker + Docker Compose
-  Flow:
    - `docker compose build` builds both frontend and backend images.
    - `docker compose up` runs containers and orchestrates service communication.
