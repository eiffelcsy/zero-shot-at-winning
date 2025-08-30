# Zero Shot at Winning

**From Guesswork to Governance: Automating Geo-Regulation with LLM**

## Overview
TikTok operates globally, and each product feature must comply with multiple geographic regulations — from Brazil’s data localization laws to GDPR in Europe. Manual compliance checks are slow, error-prone, and risk legal exposure.

Our project introduces a prototype system that uses Large Language Models (LLMs) to automatically flag features requiring geo-specific compliance logic. The solution provides auditable outputs, enabling proactive legal guardrails and traceable evidence for regulatory audits.

## Problem Statement
Detect whether a TikTok feature needs geo-specific compliance logic based on feature artifacts such as titles, descriptions, and documents (PRD, TRD).

**Outputs:**
- Flag for geo-specific compliance requirement
- Human-readable reasoning
- Optional: related regulations

**Example:**

| Feature Artifact | Flag | Notes |
|-----------------|------|-------|
| Reads user location to enforce France's copyright rules | ✅ | Legal compliance needed |
| Age gates for Indonesia’s Child Protection Law | ✅ | Legal compliance needed |
| Geofences feature rollout in US for market testing | ❌ | Business-driven, not legal |
| Video filter available globally except KR | ❓ | Requires human evaluation |

## Key Features
- **LLM-based Classification**: Analyzes feature artifacts to detect compliance needs
- **Automated Reasoning**: Provides human-readable explanations
- **Audit-Ready Outputs**: CSVs with compliance flags for regulatory review
- **Extensible Architecture**: Supports adding domain-specific knowledge or alternative detection methods

## Tech Stack
- **Backend:** Python 3.11+, FastAPI, LangChain, SQLAlchemy, Celery
- **LLM Integration:** OpenAI GPT-4 
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit
- **Infrastructure:** Docker, Docker Compose, Redis, PostgreSQL

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zero-shot-at-winning.git
cd zero-shot-at-winning
```
2. Environment Setup
3. Copy .env.example into .env and configure API keys:
```bash
cp .env.example .env
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
5. Build and run with Docker Compose:
```bash
docker compose build
docker compose up
```
5. Run Frontend UI in another terminal:
```bash
streamlit run frontend/ui.py
```

## Repository Structure
See docs/ARCHITECTURE.md for full details.
 
## Demo Video
Watch our 3-minute demo: YouTube Link

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
