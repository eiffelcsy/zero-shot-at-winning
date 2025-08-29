# app/tests/smoke_learning.py
import os
import asyncio
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg2
from dotenv import load_dotenv

# Put project root on sys.path BEFORE importing app.*
ROOT = Path(__file__).resolve().parents[2]   # .../zero-shot-at-winning
import sys
sys.path.insert(0, str(ROOT))

from app.agents.learning import LearningAgent  # uses PostgresMemoryStore inside

# ---------- env & bootstrap ----------

load_dotenv(override=True)

# Required env:
#   OPENAI_API_KEY=sk-...
#   PG_CONN_STRING=postgresql://postgres:postgres@localhost:5432/zero_shot
#   DB_NAME=zero_shot   (optional; if unset, derived from PG_CONN_STRING)

PG_CONN_STRING = os.getenv("PG_CONN_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("PG_CONN_STRING is not set. Put it in .env")

DB_NAME = os.getenv("DB_NAME")
if not DB_NAME:
    DB_NAME = urlparse(PG_CONN_STRING).path.lstrip("/") or "zero_shot"

# Derive an admin URL (same creds/host/port) but DB=postgres
parsed = urlparse(PG_CONN_STRING)
DB_ADMIN_URL = urlunparse(parsed._replace(path="/postgres"))

def ensure_db_exists():
    """Create DB if missing; fall back to TEMPLATE template0 if collation mismatch."""
    conn = psycopg2.connect(DB_ADMIN_URL)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            exists = cur.fetchone()
            if exists:
                print(f"[bootstrap] Database {DB_NAME} already exists.")
                return
            print(f"[bootstrap] Creating database {DB_NAME}...")
            try:
                cur.execute(f'CREATE DATABASE "{DB_NAME}";')
            except psycopg2.Error as e:
                if "collation version mismatch" in str(e):
                    print("[bootstrap] Collation mismatch; retrying with TEMPLATE template0...")
                    cur.execute(f'CREATE DATABASE "{DB_NAME}" TEMPLATE template0;')
                else:
                    raise
    finally:
        conn.close()

# ---------- fake state for the learning smoke ----------

def build_fake_state():
    # Minimal but realistic data flowing from previous agents + user feedback
    return {
        "feature_name": "Teen Location Sharing",
        "feature_description": "Under-18 users can share precise location with guardians.",
        "screening_analysis": {
            "agent": "ScreeningAgent",
            "risk_level": "HIGH",
            "compliance_required": True,
            "age_sensitivity": True,
            "data_sensitivity": "T5",
            "reasoning": "Precise location for minors triggers COPPA concerns."
        },
        "research_evidence": [
            {
                "jurisdiction": "US",
                "name": "Children's Online Privacy Protection Rule (COPPA)",
                "section": "16 CFR Part 312",
                "url": "https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa",
                "excerpt": "Verifiable parental consent required before collecting personal information from children under 13."
            }
        ],
        "final_decision": {
            "agent": "ValidationAgent",
            "needs_geo_logic": "YES",
            "reasoning": "Feature targets minors; COPPA + state privacy laws likely apply.",
            "related_regulations": [
                {
                    "name": "COPPA",
                    "jurisdiction": "US",
                    "section": "16 CFR Part 312",
                    "url": "https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa",
                    "evidence_excerpt": "Parental consent required."
                }
            ],
            "confidence": 0.78
        },
        # user feedback the LearningAgent should learn from:
        "user_feedback": {
            "is_correct": "no",
            "notes": "Decision ignored California-specific requirements (e.g., under-16 opt-in). Please add CA coverage and improve examples."
        }
    }

# ---------- main ----------

async def main():
    ensure_db_exists()

    # Choose where to store audit JSONL
    feedback_path = os.getenv("LEARNING_FEEDBACK_JSONL", "data/feedback.jsonl")

    # Instantiate LearningAgent (it reads PG_CONN_STRING and uses PostgresMemoryStore with use_vectors=False)
    agent = LearningAgent(feedback_file=feedback_path)

    state = build_fake_state()
    res = await agent.process(state)

    print("\n=== LearningAgent Result ===")
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))

    # Show where artifacts live
    print("\nJSONL audit file:", Path(feedback_path).resolve())
    print("Postgres:", PG_CONN_STRING)

if __name__ == "__main__":
    # Requires: OPENAI_API_KEY in env (real key)
    asyncio.run(main())
