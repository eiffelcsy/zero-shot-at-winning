"""
ValidatorAgent
--------------
Final decision-maker. Consumes:
  - ScreeningAgent analysis (risk, confidence, geographic_scope, age_sensitivity, etc.)
  - ResearchAgent evidence (ranked snippets with name/section/url/excerpt)

Produces (strict JSON):
  - needs_geo_logic: "YES" | "NO" | "REVIEW"
  - reasoning: short, clear paragraph citing evidence
  - related_regulations: [{name, jurisdiction, section, url, evidence_excerpt}]

Principles:
  - Evidence-gated: prefer REVIEW instead of guessing when evidence is thin.
  - Traceable: includes citations from ResearchAgent (not free-text web searching).
  - Minimal: one LLM call + a tiny validator.
"""

from __future__ import annotations
from typing import Dict, Any, List, Literal
from dataclasses import dataclass
import json
import os

from pydantic import BaseModel, Field, HttpUrl, ValidationError
from openai import OpenAI  # pip install openai

Decision = Literal["YES", "NO", "REVIEW"]

# ---------- Pydantic schema for strict parsing ----------

class RelatedReg(BaseModel):
    name: str
    jurisdiction: str
    section: str
    url: HttpUrl
    evidence_excerpt: str

class ValidatorJSON(BaseModel):
    needs_geo_logic: Decision
    reasoning: str = Field(min_length=10, max_length=1200)
    related_regulations: List[RelatedReg] = Field(default_factory=list)


# ---------- The Agent ----------

@dataclass
class ValidatorAgent:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self):
        # Uses OPENAI_API_KEY from environment
        self.client = OpenAI()

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload = {
          "feature_name": "...",
          "normalized_text": "...",  # optional; falls back to feature_description
          "feature_description": "...",
          "screening": {...},        # ScreeningAgent.analysis
          "research": {              # ResearchAgent output
             "candidates": [{"reg": "CA_SB976", "why": "...", "score": 0.82}, ...],
             "evidence": [
                {"reg": "CA_SB976","jurisdiction":"CA","name":"CA SB-976","section":"...","url":"...","excerpt":"...","score":7.43},
                ...
             ]
          }
        }
        returns:
          {
            "agent": "ValidatorAgent",
            "decision": "YES|NO|REVIEW",
            "reasoning": "...",
            "related_regulations": [...],
            "raw": {...}  # raw model JSON for audit
          }
        """
        # Precondition: if absolutely no evidence, short-circuit to REVIEW.
        research = (payload or {}).get("research", {})
        evidence: List[Dict[str, Any]] = research.get("evidence") or []
        if not evidence:
            return {
                "agent": "ValidatorAgent",
                "decision": "REVIEW",
                "reasoning": "No evidence snippets were provided by ResearchAgent; cannot substantiate a legal requirement.",
                "related_regulations": [],
                "raw": {"note": "empty_evidence"}
            }

        system = (
            "You are a compliance validator for product features. "
            "Only use the EVIDENCE provided below. Do not invent laws or URLs. "
            "If evidence does not entail a jurisdiction-specific duty, output REVIEW (not NO). "
            "Keep the reasoning short, specific, and cite the regulation names/sections you used."
        )

        # Keep the user content compact: pass only what’s needed for determination.
        user_obj = {
            "FEATURE": {
                "name": payload.get("feature_name"),
                "text": payload.get("normalized_text") or payload.get("feature_description", "")
            },
            "SCREENING": payload.get("screening", {}),
            "EVIDENCE": [
                {
                    "reg": e.get("reg"),
                    "jurisdiction": e.get("jurisdiction"),
                    "name": e.get("name"),
                    "section": e.get("section"),
                    "url": e.get("url"),
                    "excerpt": e.get("excerpt"),
                    "score": e.get("score"),
                } for e in evidence[:12]  # cap to keep tokens low
            ],
            "DECISION_SCHEMA": {
                "needs_geo_logic": "YES | NO | REVIEW",
                "reasoning": "short paragraph (2–5 sentences) citing evidence",
                "related_regulations": [
                    {"name":"...", "jurisdiction":"...", "section":"...", "url":"...", "evidence_excerpt":"..."}
                ]
            },
            "RULES": [
                "If the feature enforces different behavior by region due to law (e.g., minors defaults in CA, curfew in UT, reporting to NCMEC in US, minors protections in FL, DSA minors/transparency in EU), it NEEDS geo logic.",
                "Market tests or business-only geofencing are NOT legal requirements → NO.",
                "If intent is ambiguous or evidence does not entail a legal obligation, output REVIEW."
            ]
        }

        # Use JSON mode so we always get a single JSON object back.
        # (OpenAI JSON/structured outputs docs: see OpenAI “Structured model outputs” & API overview.) 
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {
                  "role": "user",
                  "content": (
                      "Return ONLY valid JSON matching DECISION_SCHEMA. "
                      "Analyze the FEATURE using SCREENING and EVIDENCE.\n\n"
                      + json.dumps(user_obj, ensure_ascii=False)
                  ),
                },
            ],
        )

        raw_json = resp.choices[0].message.content

        # Parse + validate against our Pydantic schema
        try:
            parsed = ValidatorJSON.model_validate_json(raw_json)
        except ValidationError as ve:
            # Fallback: if model returns slightly off JSON, attempt a gentle repair
            repaired = self._best_effort_repair(raw_json)
            parsed = ValidatorJSON.model_validate(repaired)

        # Minimal post-conditions to keep the model honest:
        # 1) YES must include at least one related regulation
        # 2) Every related regulation URL should exist in the evidence set
        if parsed.needs_geo_logic == "YES" and not parsed.related_regulations:
            parsed.needs_geo_logic = "REVIEW"
            parsed.reasoning = (
                parsed.reasoning + " | Adjusted to REVIEW: no related_regulations were supplied."
            )

        evidence_urls = {e.get("url") for e in evidence if e.get("url")}
        filtered_regs = []
        for r in parsed.related_regulations:
            if r.url in evidence_urls:
                filtered_regs.append(r)
        # If the model cited URLs not in evidence, drop them.
        if filtered_regs and len(filtered_regs) != len(parsed.related_regulations):
            parsed.related_regulations = filtered_regs

        # If nothing remains but decision is YES, downgrade to REVIEW
        if parsed.needs_geo_logic == "YES" and not parsed.related_regulations:
            parsed.needs_geo_logic = "REVIEW"
            parsed.reasoning = (
                parsed.reasoning + " | Adjusted to REVIEW: citations did not match provided evidence."
            )

        return {
            "agent": "ValidatorAgent",
            "decision": parsed.needs_geo_logic,
            "reasoning": parsed.reasoning.strip(),
            "related_regulations": [r.model_dump() for r in parsed.related_regulations],
            "raw": json.loads(raw_json),
        }

    # --- helpers ---

    def _best_effort_repair(self, raw: str) -> Dict[str, Any]:
        """
        Very small fixer for common JSON issues (e.g., trailing commas or missing array).
        We keep it conservative for MVP.
        """
        try:
            obj = json.loads(raw)
            # Ensure fields exist
            obj.setdefault("needs_geo_logic", "REVIEW")
            obj.setdefault("reasoning", "Model output did not strictly match schema; defaulting to REVIEW.")
            obj.setdefault("related_regulations", [])
            return obj
        except Exception:
            return {
                "needs_geo_logic": "REVIEW",
                "reasoning": "Unparseable model output; defaulting to REVIEW.",
                "related_regulations": []
            }
