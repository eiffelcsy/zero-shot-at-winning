"""
ResearchAgent
-------------
Purpose:
  Given:
    - normalized feature text (from Terminology Resolver)
    - ScreeningAgent analysis (risk, jurisdictions, age sensitivity, etc.)
  Find:
    - The most relevant regulations (from the 5 targeted regs)
    - Top evidence snippets (title/section/url/excerpt) per regulation
  Output:
    - Compact payload for the Validator Agent (not implemented here)

Data dependencies:
  - Local curated KB files at data/kb/*.jsonl
    Each line: {
      "jurisdiction": "US|EU|CA|UT|FL",
      "reg_code": "US_2258A | EU_DSA | CA_SB976 | UT_MINORS | FL_MINORS",
      "name": "18 USC §2258A",
      "section": "(a) Reporting requirements",
      "url": "https://...",
      "excerpt": "short, precise passage ..."
    }

MVP Retrieval:
  - In-memory TF-IDF-ish scoring (no external DB)
  - Two-pass retrieval:
      (1) Focused: search only within predicted regs
      (2) Global: low-k "safety net" search over all docs
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Iterable
from dataclasses import dataclass
import os, json, glob, math, re
from collections import Counter, defaultdict
from hashlib import sha256

# If you have a BaseComplianceAgent class, inherit it; otherwise a minimal stub is fine.
try:
    from .base import BaseComplianceAgent  # optional
except Exception:
    class BaseComplianceAgent:
        def __init__(self, name: str, llm=None):
            self.name = name
            self.llm = llm
        def _log_interaction(self, inp, out):  # no-op for MVP
            pass


# ------------------------------
# Constants & tiny helpers
# ------------------------------

REG_EU_DSA     = "EU_DSA"
REG_CA_SB976   = "CA_SB976"
REG_FL_MINORS  = "FL_MINORS"
REG_UT_MINORS  = "UT_MINORS"
REG_US_2258A   = "US_2258A"

SUPPORTED_REGS = {REG_EU_DSA, REG_CA_SB976, REG_FL_MINORS, REG_UT_MINORS, REG_US_2258A}

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
STOP = set(["the","a","an","of","and","for","to","in","on","with","by","or","is","are","be"])

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOP]


@dataclass
class KBRecord:
    jurisdiction: str
    reg_code: str
    name: str
    section: str
    url: str
    excerpt: str
    # cached tokens
    tokens: List[str] | None = None


# ------------------------------
# Research Agent
# ------------------------------

class ResearchAgent(BaseComplianceAgent):
    """
    Research Agent:
      - Loads local KB (JSONL)
      - Builds IDF over excerpts
      - Scores documents against a query using sum(IDF(term)) over intersection
      - Returns top-k per candidate regulation + a small global safety net
    """

    def __init__(self, kb_dir: str = "data/kb", top_k_per_reg: int = 3, top_k_global: int = 2, llm=None):
        super().__init__("ResearchAgent", llm)
        self.kb_dir = kb_dir
        self.top_k_per_reg = top_k_per_reg
        self.top_k_global = top_k_global

        self.kb: List[KBRecord] = self._load_kb(self.kb_dir)
        self.df = self._build_df(self.kb)         # document frequency
        self.N = max(1, len(self.kb))
        self.idf = {term: math.log((self.N + 1) / (df + 1)) + 1.0 for term, df in self.df.items()}

    # ---------- Public entrypoint ----------

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected input_data:
          {
            "feature_name": "...",
            "feature_description": "...",
            "normalized_text": "...",      # optional; fallback to feature_description
            "screening": {                 # ScreeningAgent.analysis
               "risk_level": "...",
               "compliance_required": true/false,
               "confidence": 0.xx,
               "trigger_keywords": [...],
               "regulatory_indicators": [...],
               "reasoning": "...",
               "needs_research": true/false,
               "geographic_scope": ["CA", "US"] or ["unknown"],
               "age_sensitivity": true/false
            }
          }
        Returns:
          {
            "agent": "ResearchAgent",
            "query": "...",          # internal query used
            "candidates": [          # candidate regs chosen by heuristics
              {"reg": "CA_SB976", "why": "...", "score": 0.82}
            ],
            "evidence": [            # ranked docs to pass to validator
              {
                "reg": "CA_SB976",
                "jurisdiction": "CA",
                "name": "CA SB-976",
                "section": "(...)",
                "url": "https://...",
                "excerpt": "....",
                "score": 7.43
              },
              ...
            ],
            "next_step": "validation"
          }
        """
        try:
            norm_text = input_data.get("normalized_text") or input_data.get("feature_description", "")
            screening = input_data.get("screening", {})

            query = self._form_query(norm_text, screening)
            candidates = self._select_candidate_regs(screening, norm_text)

            # Focused retrieval per candidate reg
            focused = []
            for reg, why in candidates:
                hits = self._retrieve(query, restrict_reg=reg, k=self.top_k_per_reg)
                for rec, score in hits:
                    focused.append(self._rec_to_ev(rec, score, reg_override=reg))

            # Global safety net (small k)
            global_hits = self._retrieve(query, restrict_reg=None, k=self.top_k_global)
            global_ev = [self._rec_to_ev(rec, score) for rec, score in global_hits]

            evidence = self._merge_and_dedupe(focused + global_ev)

            result = {
                "agent": self.name,
                "query": query,
                "candidates": [{"reg": r, "why": why, "score": self._candidate_prior(r, screening)} for r, why in candidates],
                "evidence": evidence,
                "next_step": "validation"
            }
            self._log_interaction(input_data, result)
            return result

        except Exception as e:
            return {
                "agent": self.name,
                "error": f"research_failed: {e}",
                "evidence": [],
                "next_step": "human_review"
            }

    # ---------- Retrieval internals ----------

    def _load_kb(self, kb_dir: str) -> List[KBRecord]:
        kb: List[KBRecord] = []
        for path in glob.glob(os.path.join(kb_dir, "*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        rec = KBRecord(
                            jurisdiction=obj.get("jurisdiction",""),
                            reg_code=obj.get("reg_code",""),
                            name=obj.get("name",""),
                            section=obj.get("section",""),
                            url=obj.get("url",""),
                            excerpt=obj.get("excerpt",""),
                        )
                        rec.tokens = tokenize(" ".join([rec.name, rec.section, rec.excerpt]))
                        kb.append(rec)
                    except json.JSONDecodeError:
                        continue
        return kb

    def _build_df(self, kb: List[KBRecord]) -> Dict[str, int]:
        seen_per_doc = defaultdict(int)
        for rec in kb:
            if not rec.tokens: continue
            for term in set(rec.tokens):
                seen_per_doc[term] += 1
        return dict(seen_per_doc)

    def _score(self, query_terms: List[str], doc_terms: List[str]) -> float:
        """Simple sum of IDF for overlapping terms; fast and explainable for MVP."""
        if not query_terms or not doc_terms:
            return 0.0
        qset = set(query_terms)
        dset = set(doc_terms)
        overlap = qset & dset
        return sum(self.idf.get(t, 0.0) for t in overlap)

    def _retrieve(self, query: str, restrict_reg: str | None, k: int) -> List[Tuple[KBRecord, float]]:
        q_terms = tokenize(query)
        scored: List[Tuple[KBRecord, float]] = []
        for rec in self.kb:
            if restrict_reg and rec.reg_code != restrict_reg:
                continue
            s = self._score(q_terms, rec.tokens or [])
            if s > 0:
                scored.append((rec, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _merge_and_dedupe(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prefer the highest score per (reg, url, section) tuple."""
        best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for ev in items:
            key = (ev.get("reg",""), ev.get("url",""), ev.get("section",""))
            if key not in best or ev.get("score",0) > best[key].get("score",0):
                best[key] = ev
        return sorted(best.values(), key=lambda x: x.get("score",0), reverse=True)

    def _rec_to_ev(self, rec: KBRecord, score: float, reg_override: str | None = None) -> Dict[str, Any]:
        return {
            "reg": reg_override or rec.reg_code,
            "jurisdiction": rec.jurisdiction,
            "name": rec.name,
            "section": rec.section,
            "url": rec.url,
            "excerpt": rec.excerpt,
            "score": round(score, 4),
            "doc_hash": sha256((rec.url + rec.section).encode("utf-8")).hexdigest()[:12]
        }

    # ---------- Candidate selection (no training; pure heuristics) ----------

    def _select_candidate_regs(self, screening: Dict[str, Any], text: str) -> List[Tuple[str, str]]:
        """
        Decide which regs to search first, based on:
          - geographic_scope (CA, UT, FL, US, EU, unknown)
          - age_sensitivity (True/False)
          - trigger/regulatory keywords from ScreeningAgent
        Returns: list of (reg_code, why_string)
        """
        geo = [g.upper() for g in screening.get("geographic_scope", []) or []]
        age = bool(screening.get("age_sensitivity", False))
        triggers = " ".join(screening.get("trigger_keywords", []) or []).lower()
        indicators = " ".join(screening.get("regulatory_indicators", []) or []).lower()
        full_text = f"{text}\n{triggers}\n{indicators}".lower()

        cands: List[Tuple[str, str]] = []

        # US NCMEC reporting
        if re.search(r"\b(ncmec|child\s*(sexual)?\s*abuse|csam|report)\b", full_text) or "us" in geo:
            cands.append((REG_US_2258A, "US provider reporting to NCMEC / child exploitation"))

        # California minors personalization defaults
        if ("california" in geo or "ca" in geo or "sb-976" in full_text or "sb976" in full_text) and age:
            cands.append((REG_CA_SB976, "CA minors + PF defaults / parental opt-in"))

        # Utah minors curfew/age verification
        if ("utah" in geo or "ut" in geo) and age:
            cands.append((REG_UT_MINORS, "UT minors curfew/age restrictions"))

        # Florida minors online protections
        if ("florida" in geo or "fl" in geo) and age:
            cands.append((REG_FL_MINORS, "FL online protections for minors"))

        # EU DSA (ads to minors, transparency, recommender systems)
        if ("eu" in geo or "gdpr" in full_text or "digital services act" in full_text or "dsa" in full_text):
            cands.append((REG_EU_DSA, "EU Digital Services Act relevance"))

        # Fallback: if nothing classified, search all but still prioritize minors/US if cues exist
        if not cands:
            # If minors cues
            if re.search(r"\b(minor|under\s*18|teen|age[-\s]*gate)\b", full_text):
                cands.extend([
                    (REG_CA_SB976, "Ambiguous minors: include CA as candidate"),
                    (REG_FL_MINORS, "Ambiguous minors: include FL as candidate"),
                    (REG_UT_MINORS, "Ambiguous minors: include UT as candidate"),
                    (REG_EU_DSA, "Ambiguous minors: include EU-DSA minors controls"),
                ])
            else:
                # Broad US/EU coverage
                cands.extend([
                    (REG_US_2258A, "No clear signal: include US reporting"),
                    (REG_EU_DSA, "No clear signal: include EU DSA"),
                ])

        # Deduplicate while preserving order
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for reg, why in cands:
            if reg in SUPPORTED_REGS and reg not in seen:
                uniq.append((reg, why)); seen.add(reg)
        return uniq

    def _candidate_prior(self, reg_code: str, screening: Dict[str, Any]) -> float:
        """A tiny prior score used for UI/debug; does NOT affect document ranking."""
        base = 0.5
        geo = [g.upper() for g in screening.get("geographic_scope", []) or []]
        age = bool(screening.get("age_sensitivity", False))
        if reg_code == REG_US_2258A and ("US" in geo or "UNITED STATES" in geo):
            base += 0.2
        if reg_code in (REG_CA_SB976, REG_FL_MINORS, REG_UT_MINORS) and age:
            base += 0.2
        if reg_code == REG_EU_DSA and ("EU" in geo):
            base += 0.2
        return min(1.0, base)

    # ---------- Query formulation ----------

    def _form_query(self, norm_text: str, screening: Dict[str, Any]) -> str:
        """
        Build a compact query string from normalized feature text and salient screening fields.
        Keep it short to reduce noise for TF-IDF style matching.
        """
        parts = [norm_text]
        if screening.get("age_sensitivity"):
            parts.append("minors under 18 age gate")
        geo = screening.get("geographic_scope") or []
        if isinstance(geo, list) and geo:
            parts.append(" ".join(geo))
        # include 2–3 top keywords if present
        for k in (screening.get("trigger_keywords") or [])[:3]:
            parts.append(k)
        # short indicators
        for k in (screening.get("regulatory_indicators") or [])[:2]:
            parts.append(k)
        return " ".join(parts)
