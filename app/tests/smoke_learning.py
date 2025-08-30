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

from app.agents.feedback.learning import LearningAgent  # uses PostgresMemoryStore inside

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

        # LearningAgent reads research_analysis (dict)
        "research_analysis": {
            "regulations": [
                {
                    "jurisdiction": "US",
                    "name": "Children's Online Privacy Protection Rule (COPPA)",
                    "section": "16 CFR Part 312",
                    "url": "https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa",
                    "excerpt": "Verifiable parental consent required before collecting personal information from children under 13."
                },
                # Add a state-specific hook to provoke few-shots and glossary
                {
                    "jurisdiction": "US-CA",
                    "name": "California Privacy Rights Act (CPRA)",
                    "section": "1798.120 & 1798.140 (definitions/opt-in)",
                    "url": "https://oag.ca.gov/privacy/ccpa",
                    "excerpt": "Opt-in consent required for selling/sharing personal information of consumers under 16; parents/guardians may consent for under 13."
                }
            ]
        },

        # NEW ValidationOutput schema
        "validation_analysis": {
            "agent": "ValidationAgent",
            "feature_name": "Teen Location Sharing",
            "final_decision": "COMPLIANT",  # COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW
            "confidence_score": 0.52,
            "reasoning": "Assessment considers COPPA. California-specific under-16 opt-in and geofencing review need clearer handling.",
            "compliance_requirements": [
                "Obtain verifiable parental consent (VPC) for users under 13",
                "Honor CA under-16 opt-in for selling/sharing personal data",
                "Minimize precise geolocation retention; purpose-bind and set TTL",
                "Provide guardian dashboard for consent, revocation, and audit trail"
            ],
            "risk_assessment": "High risk due to precise geolocation of minors plus cross-jurisdictional obligations (federal + CA).",
            "recommendations": [
                "Add California under-16 opt-in logic and logging",
                "Gate precise location behind verified guardian consent",
                "Enforce TTL for location (e.g., 24h) and redact after aggregation",
                "Emit auditable events for consent lifecycle (grant/revoke/expire)"
            ],
            "tiktok_terminology_used": False
        },

        # User feedback to drive LEARNING (explicitly ask for glossary & few-shots)
        "user_feedback": {
            "is_correct": "no",
            "notes": (
                "Please correct the analysis to explicitly account for CA under-16 opt-in and precise geolocation TTL. "
                "Also, generate training examples for the ValidationAgent that show how to integrate COPPA + CA in one decision. "
                "We also need glossary definitions for unclear terms used across agents.\n\n"
                "GLOSSARY_CANDIDATES:\n"
                "- T5: Highest sensitivity tier for personal data incl. precise geolocation of minors.\n"
                "- VPC (Verifiable Parental Consent): Methods acceptable under COPPA to obtain parental consent.\n"
                "- Geo-logic vs Geofencing: 'Geo-logic' means jurisdictional decision rules that switch behavior by location; "
                "'Geofencing' is the technical mechanism to detect location boundaries.\n"
                "- TTL (Time-To-Live): Maximum retention window for raw precise location before redaction/aggregation.\n"
                "- Under-16 Opt-in (CA/CPRA): Consent requirement for selling/sharing personal data of consumers under 16.\n\n"
                "FEW_SHOT_REQUEST (ValidationAgent):\n"
                "BAD_EXAMPLE:\n"
                "{"
                "\"final_decision\":\"COMPLIANT\","
                "\"reasoning\":\"Only COPPA reviewed; CA rules ignored.\","
                "\"compliance_requirements\":[\"VPC for <13\"],"
                "\"recommendations\":[\"None\"]"
                "}\n"
                "GOOD_EXAMPLE:\n"
                "{"
                "\"final_decision\":\"NEEDS_REVIEW\","
                "\"reasoning\":\"COPPA VPC is required for <13 and CA under-16 opt-in applies. TTL for raw geo set to 24h; guardian audit enabled.\","
                "\"compliance_requirements\":["
                "\"VPC for <13 (COPPA)\","
                "\"CA under-16 opt-in for selling/sharing\","
                "\"Geo TTL <= 24h; aggregate thereafter\""
                "],"
                "\"recommendations\":["
                "\"Implement consent gating flow (parental/teen)\","
                "\"Log consent grant/revoke with timestamps\","
                "\"Block selling/sharing absent CA opt-in for 13-15\""
                "]"
                "}\n"
                "Please add at least one few-shot for 'COMPLIANT' and one for 'NON_COMPLIANT' with clear evidence linkage."
            )
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
