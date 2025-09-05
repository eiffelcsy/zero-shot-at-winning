# Zero Shot at Winning

**From Guesswork to Governance: Automating Geo-Regulation with LLM**

## Overview
TikTok operates globally, and each product feature must comply with multiple geographic regulations — from Brazil’s data localization laws to GDPR in Europe. Manual compliance checks are slow, error-prone, and risk legal exposure.

Our project introduces a prototype system that uses Large Language Models (LLMs) to automatically flag features requiring geo-specific compliance logic. The solution provides auditable outputs, enabling proactive legal guardrails and traceable evidence for regulatory audits.

## Problem Statement
Detect whether a TikTok feature needs geo-specific compliance logic based on feature artifacts such as titles, descriptions, and documents (PRD, TRD).

**Outputs:**
- Flag for geo-specific compliance requirement
- Clear reasoning
- Related regulations

## Key Features
- **Automated Screening** – Detects whether a feature requires region-specific compliance logic.
- **Explainable AI** – Produces clear reasoning, risk level, and related regulation references.
- **Multi-Agent Workflow** – Screening, Research, Validation, and Learning agents collaborate to improve accuracy.
- **Feedback Loop** – Human feedback continuously enhances the system’s precision.
- **Document Ingestion** – Upload PDFs of regulations, which are chunked, embedded, and stored for retrieval via ChromaDB.
- **Audit-Ready Evidence** – Generates outputs suitable for legal review and regulatory inquiries.

## Tech Stack
- **Backend:** Python 3.11+, FastAPI, LangChain
- **LLM Integration:** OpenAI GPT-4 with LangChain orchestration
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit
- **Infrastructure:** Docker, Docker Compose, PostgreSQL

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zero-shot-at-winning.git
cd zero-shot-at-winning
```
2. Environment Setup:

Copy .env.example into .env and configure API keys:
```bash
cp .env.example .env
```
3. Build and run with Docker Compose:
```bash
docker compose build
docker compose up
```
4. Run Frontend UI in another terminal:
```bash
streamlit run frontend/ui.py
```

## Repository Structure
See docs/ARCHITECTURE.md for full details.
 
## Demo Video
Watch our 3-minute demo: [YouTube Link](https://youtu.be/K7PVkRO0jCU)

## Focused Regulations
- EU Digital Service Act (DSA)
- California Protecting Our Kids from Social Media Addiction Act
- Florida Online Protections for Minors
- Utah Social Media Regulation Act
- US reporting requirements to NCMEC

## Team
- Eiffel Chong Shiang Yih
- Ryan Soh Jing Zhi
- Teo Fu Qiang
- Lim Yixuan
