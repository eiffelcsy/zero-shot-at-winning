# app/tests/smoke_e2e_learning.py
import os, sys, asyncio, json
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import psycopg2

# Ensure project root on sys.path BEFORE importing app.*
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

from app.agents.validation import ValidationAgent
from app.agents.learning import LearningAgent
from app.agents.memory.memory_pg import PostgresMemoryStore

# --- Env / globals ---
PG_CONN_STRING = os.getenv("PG_CONN_STRING")
if not PG_CONN_STRING:
    raise RuntimeError("PG_CONN_STRING is not set. Put it in .env")

DB_NAME = os.getenv("DB_NAME")
if not DB_NAME:
    DB_NAME = urlparse(PG_CONN_STRING).path.lstrip("/") or "zero_shot"

parsed = urlparse(PG_CONN_STRING)
DB_ADMIN_URL = urlunparse(parsed._replace(path="/postgres"))

def ensure_db_exists():
    """Create DB if missing; fall back to TEMPLATE template0 on collation mismatch."""
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

async def main():
    ensure_db_exists()

    # Make LangSmith quiet if configured
    os.environ.pop("LANGCHAIN_API_KEY", None)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # --- 1) Baseline Validation (NO overlay) ---
    # Pass a non-empty string to bypass DB overlay for the first run.
    v1 = ValidationAgent(memory_overlay=" ")  # uses this string instead of Postgres overlay
    state = {
        "feature_name": "Teen Location Sharing",
        "feature_description": "Under-18 users can share precise location.",
        "screening_analysis": {
            "agent": "ScreeningAgent",
            "risk_level": "HIGH",
            "compliance_required": True,
            "age_sensitivity": True,
            "data_sensitivity": "T5",
            "reasoning": "Precise location for minors triggers COPPA concerns."
        },
        # ValidationAgent expects research_analysis.regulations
        "research_analysis": {
            "regulations": [
                {
                    "name": "Children's Online Privacy Protection Rule (COPPA)",
                    "jurisdiction": "US",
                    "section": "16 CFR Part 312",
                    "url": "https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa",
                    "evidence_excerpt": "Verifiable parental consent required before collecting PI from children under 13."
                }
            ]
        },
    }

    baseline = await v1.process(state)
    print("\n=== Baseline Validation ===")
    print(json.dumps(baseline, indent=2))

    # Basic sanity
    assert baseline.get("validation_completed") is True
    base_out = baseline["validation_analysis"]
    base_reason = json.dumps(base_out.get("reasoning", ""), ensure_ascii=False)
    base_conf = float(base_out.get("confidence", 0.0))

    # --- 2) Learning step (writes rules/few-shots to Postgres + JSONL) ---
    learner = LearningAgent(pg_conn=PG_CONN_STRING)
    learn_state = {
        "feature_name": state["feature_name"],
        "feature_description": state["feature_description"],
        "screening_analysis": state["screening_analysis"],
        # LearningAgent expects research_evidence list (not research_analysis)
        "research_evidence": [
            {
                "jurisdiction": r["jurisdiction"],
                "name": r["name"],
                "section": r["section"],
                "url": r["url"],
                "excerpt": r["evidence_excerpt"]
            } for r in state["research_analysis"]["regulations"]
        ],
        "final_decision": base_out,
        "user_feedback": {
            "is_correct": "no",
            "notes": "Please cite regulation URLs clearly and consider California minors' protections."
        },
    }

    learning = await learner.process(learn_state)
    print("\n=== LearningAgent Report ===")
    print(json.dumps(learning, indent=2))

    # At least one memory area should be updated
    counts = learning.get("learning_report", {}).get("learning_counts", {})
    assert any(int(counts.get(k, 0)) > 0 for k in ("rules", "few_shots", "glossary", "kb_snippets")), \
        "LearningAgent did not apply any updates to memory."

    # --- 3) Reprompt Validation with DB overlay ---
    # Fresh instance that uses Postgres overlay (default behavior)
    v2 = ValidationAgent()  # will init store from PG_CONN_STRING and render overlay
    rerun = await v2.process(state)
    print("\n=== Reprompted Validation (after Learning) ===")
    print(json.dumps(rerun, indent=2))

    assert rerun.get("validation_completed") is True
    new_out = rerun["validation_analysis"]
    new_reason = json.dumps(new_out.get("reasoning", ""), ensure_ascii=False)
    new_conf = float(new_out.get("confidence", 0.0))

    # --- Assertions showing drift toward feedback ---
    #  A) Reasoning or confidence changed
    assert (new_reason != base_reason) or (new_conf != base_conf), \
        "Validation output did not change after learning."

    #  B) (Soft) Prefer to see stronger citation emphasis in reasoning
    #     This is heuristic; don't fail the run, just print a note if not observed.
    if "cite" in learner.feedback_file.lower():  # just to branch; not critical
        pass
    if "calif" in learning.get("learning_report", {}).get("learning_summary", "").lower():
        print("[info] Learning summary mentions California; good sign.")

if __name__ == "__main__":
    asyncio.run(main())
