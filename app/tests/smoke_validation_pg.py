import os, sys
from urllib.parse import urlparse, urlunparse
import psycopg2
from pathlib import Path

# Put project root on sys.path BEFORE importing app.*
ROOT = Path(__file__).resolve().parents[2]   # .../zero-shot-at-winning
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(override=True)

from app.agents.memory.memory_pg import PostgresMemoryStore

PG_CONN_STRING = os.getenv("PG_CONN_STRING")
DB_NAME = os.getenv("DB_NAME")

# Derive admin URL from PG_CONN_STRING by replacing the database with "postgres"
parsed = urlparse(PG_CONN_STRING)
admin_path = "/postgres"
DB_ADMIN_URL = urlunparse(parsed._replace(path=admin_path))

def ensure_db_exists():
    conn = psycopg2.connect(DB_ADMIN_URL)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            exists = cur.fetchone()
            if not exists:
                print(f"Creating database {DB_NAME} from template0...")
                # Prefer template0 to bypass collation mismatch on template1
                cur.execute(f"CREATE DATABASE {DB_NAME} TEMPLATE template0;")
            else:
                print(f"Database {DB_NAME} already exists, skipping creation.")
    finally:
        conn.close()


def main():
    ensure_db_exists()
    use_vectors = False
    store = PostgresMemoryStore(os.getenv("PG_CONN_STRING"), use_vectors=use_vectors)

    # 1) Write a rule
    res1 = store.update_rules([
        {"agent": "validation", "rule_text": "Cite at least one regulation URL in final decisions."},
        {"agent": "screening",  "rule_text": "Flag features involving T5 data as HIGH risk."},
    ])
    print("Rules applied:", res1)

    # 2) Upsert few-shots
    res2 = store.add_few_shots([
        {"agent": "validation", "example": "Sample validation example A"},
        {"agent": "validation", "example": "Sample validation example B"},
    ])
    print("Few-shots applied:", res2)

    # 3) Render overlay (should include rules and two few-shots)
    txt = store.render_overlay_for("validation")
    print("\n--- OVERLAY ---\n", txt)

    # 4) Add KB snippets (vector indexed only if use_vectors=True)
    res3 = store.add_kb_snippets([
        {"url": "https://example.com/coppa", "section": "consent", "text": "Parental consent required...", "name": "COPPA"},
    ])
    print("KB applied:", res3)

if __name__ == "__main__":
    main()
