# app/agents/memory_pg.py
from __future__ import annotations
from typing import List, Dict, Any, Literal, Tuple, Optional
from dataclasses import dataclass
import json, hashlib, os

try:
    from langgraph.store.postgres import PostgresStore
except Exception:
    PostgresStore = None  # fallback later

try:
    from langchain.embeddings import init_embeddings  # LC v0.3+
    _init_embeddings = lambda: init_embeddings("openai:text-embedding-3-small")
except Exception:
    try:
        from langchain_openai import OpenAIEmbeddings  # older LC
        _init_embeddings = lambda: OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        _init_embeddings = lambda: None  # no vectors in tests

from dotenv import load_dotenv
load_dotenv()

def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]

@dataclass
class ApplyResult:
    applied: int
    namespace: Tuple[str, ...]


class _InMemoryStore:
    """Tiny drop-in used for tests; supports setup/get/put/search."""
    def __init__(self): self._d = {}
    def setup(self): pass
    def put(self, ns: Tuple[str, ...], key: str, value: Any, index: bool = False):
        self._d.setdefault(ns, {})[key] = value
    def get(self, ns: Tuple[str, ...], key: str) -> Optional[Any]:
        return self._d.get(ns, {}).get(key)
    def search(self, ns: Tuple[str, ...], query: str, limit: int = 50):
        class _Item:
            def __init__(self, v): self.value = v
        values = list(self._d.get(ns, {}).values())[-limit:]
        return [_Item(v) for v in values]


class PostgresMemoryStore:
    """
    Postgres-backed memory overlays; gracefully degrades to in-memory for tests.
    """

    def __init__(self, conn_string: str | None = None, use_vectors: bool = True):
        # Fallback if PostgresStore is unavailable
        if PostgresStore is None:
            self._store_ctx = None
            self.store = _InMemoryStore()
            self.store.setup()
            return

        conn = conn_string or os.getenv("PG_CONN_STRING", "postgresql://user:pass@localhost:5432/dbname")
        index = None
        if use_vectors:
            emb = _init_embeddings()
            if emb is not None:
                index = {"dims": 1536, "embed": emb, "fields": ["text", "excerpt", "section", "name"]}

        # Some LangGraph versions return a context manager here.
        obj = PostgresStore.from_conn_string(conn, index=index) if index else PostgresStore.from_conn_string(conn)
        if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
            # keep the context open on the instance, close in __del__
            self._store_ctx = obj
            self.store = obj.__enter__()
        else:
            self._store_ctx = None
            self.store = obj

        # call setup if available
        if hasattr(self.store, "setup"):
            self.store.setup()

    @classmethod
    def from_conn_string(cls, conn_string: str | None = None, use_vectors: bool = True) -> "PostgresMemoryStore":
        return cls(conn_string, use_vectors)

    def __del__(self):
        # close context cleanly if we opened one
        ctx = getattr(self, "_store_ctx", None)
        if ctx and hasattr(ctx, "__exit__"):
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    # ---------- Glossary ----------
    def update_glossary(self, items: List[Dict[str, Any]]) -> ApplyResult:
        ns = ("memory", "glossary")
        applied = 0
        for it in items or []:
            term = (it.get("term") or "").strip()
            expansion = (it.get("expansion") or "").strip()
            if not term or not expansion:
                continue
            self.store.put(ns, term, {"expansion": expansion, "hints": it.get("hints", [])})
            applied += 1
        return ApplyResult(applied=applied, namespace=ns)

    # ---------- KB snippets ----------
    def add_kb_snippets(self, snippets: List[Dict[str, Any]]) -> ApplyResult:
        ns = ("kb", "snippets"); applied = 0
        for snip in snippets or []:
            keyh = _hash({"u": snip.get("url"), "s": snip.get("section")})
            if self.store.get(ns, keyh) is not None:
                continue
            self.store.put(ns, keyh, snip)
            applied += 1
        return ApplyResult(applied=applied, namespace=ns)

    # ---------- Few-shot examples ----------
    def add_few_shots(self, examples: List[Dict[str, Any]]) -> ApplyResult:
        applied = 0
        ns_map = {"screening": ("fewshots", "screening"),
                  "research": ("fewshots", "research"),
                  "validation": ("fewshots", "validation")}
        last_ns = None
        for ex in examples or []:
            agent = (ex.get("agent") or "").lower()
            if agent not in ns_map:
                continue
            ns = ns_map[agent]
            keyh = _hash(ex)
            if self.store.get(ns, keyh) is None:
                self.store.put(ns, keyh, ex, index=False)
                applied += 1
            last_ns = ns
        return ApplyResult(applied=applied, namespace=last_ns or ("fewshots",))

    # ---------- Rules ----------
    def update_rules(self, rules: List[Dict[str, Any]]) -> ApplyResult:
        ns = ("memory", "rules"); applied = 0
        for r in rules or []:
            agent = (r.get("agent") or "").lower()
            text = (r.get("rule_text") or "").strip()
            if agent not in ("screening", "validation") or not text:
                continue
            item = {"agent": agent, "rule_text": text}
            keyh = _hash(item)
            if self.store.get(ns, keyh) is None:
                self.store.put(ns, keyh, item, index=False)
                applied += 1
        return ApplyResult(applied=applied, namespace=ns)

    # ---------- Overlay rendering ----------
    def render_overlay_for(self, agent: Literal["screening","validation"]) -> str:
        overlay: list[str] = []
        ns_rules = ("memory", "rules")
        rules = self.store.search(ns_rules, query=agent, limit=200)
        for item in rules:
            val = getattr(item, "value", item)
            if isinstance(val, dict) and val.get("agent") == agent:
                overlay.append(f"- RULE: {val.get('rule_text')}")

        ns_fs = ("fewshots", agent)
        few = self.store.search(ns_fs, query="", limit=2)
        if few:
            overlay.append("\nFEW-SHOT EXAMPLES:")
            for item in few:
                val = getattr(item, "value", item)
                overlay.append(json.dumps(val, ensure_ascii=False))

        return "MEMORY OVERLAY\n" + "\n".join(overlay) if overlay else ""
