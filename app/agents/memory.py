from __future__ import annotations
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
import os, json, hashlib


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]


@dataclass
class ApplyResult:
    applied: int
    file: str


class MemoryStore:
    """File-backed memory overlays for agents."""

    def __init__(self):
        self.paths = {
            "glossary": "data/memory/glossary_overrides.json",
            "kb": "data/kb/user_contributions.jsonl",
            "few_screening": "data/memory/few_shots/screening.jsonl",
            "few_research": "data/memory/few_shots/research.jsonl",
            "few_validation": "data/memory/few_shots/validation.jsonl",
            "rules": "data/memory/rules_overrides.json",
        }
        for p in self.paths.values():
            _ensure_dir(p)
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    if p.endswith(".json"):
                        # Initialize JSON files to an empty object/list
                        f.write("{}" if "glossary" in p else "[]")
                    else:
                        # jsonl files can start empty
                        pass

    # ---------- Glossary (Screening/Resolver) ----------
    def update_glossary(self, items: List[Dict[str, Any]]) -> ApplyResult:
        path = self.paths["glossary"]
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        applied = 0
        for it in items or []:
            term = (it.get("term") or "").strip()
            expansion = (it.get("expansion") or "").strip()
            hints = it.get("hints", [])
            if not term or not expansion:
                continue
            # merge/overwrite
            data[term] = {"expansion": expansion, "hints": hints}
            applied += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return ApplyResult(applied=applied, file=path)

    # ---------- KB snippets (Research) ----------
    def add_kb_snippets(self, snippets: List[Dict[str, Any]]) -> ApplyResult:
        path = self.paths["kb"]
        applied = 0
        # Deduplicate by (url, section) short hash
        existing_hashes = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        existing_hashes.add(_hash({"u": obj.get("url"), "s": obj.get("section")}))
        except FileNotFoundError:
            pass

        with open(path, "a", encoding="utf-8") as f:
            for snip in snippets or []:
                keyh = _hash({"u": snip.get("url"), "s": snip.get("section")})
                if keyh in existing_hashes:
                    continue
                f.write(json.dumps(snip, ensure_ascii=False) + "\n")
                existing_hashes.add(keyh)
                applied += 1
        return ApplyResult(applied=applied, file=path)

    # ---------- Few-shot examples (all agents) ----------
    def add_few_shots(self, examples: List[Dict[str, Any]]) -> ApplyResult:
        # Route by agent field
        applied = 0
        for ex in examples or []:
            agent = (ex.get("agent") or "").lower()
            if agent not in ("screening", "research", "validation"):
                continue
            path = self.paths[f"few_{agent}"]
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                applied += 1
        # return a synthetic “directory” path for visibility
        return ApplyResult(applied=applied, file="data/memory/few_shots/")

    # ---------- Rule overlays (Screening/Validation prompts) ----------
    def update_rules(self, rules: List[Dict[str, Any]]) -> ApplyResult:
        path = self.paths["rules"]
        try:
            rule_list = json.load(open(path, "r", encoding="utf-8"))
            if not isinstance(rule_list, list):
                rule_list = []
        except Exception:
            rule_list = []

        applied = 0
        for r in rules or []:
            agent = (r.get("agent") or "").lower()
            text = (r.get("rule_text") or "").strip()
            if agent not in ("screening", "validation") or not text:
                continue
            item = {"agent": agent, "rule_text": text}
            if item not in rule_list:
                rule_list.append(item)
                applied += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rule_list, f, ensure_ascii=False, indent=2)
        return ApplyResult(applied=applied, file=path)

    # ---------- Rendering overlays for prompts (optional) ----------
    def render_overlay_for(self, agent: Literal["screening", "validation"]) -> str:
        """Generate a small block that agents can append to their prompt."""
        overlay: List[str] = []

        # Rules first
        try:
            rules = json.load(open(self.paths["rules"], "r", encoding="utf-8"))
            for r in rules:
                if r.get("agent") == agent:
                    overlay.append(f"- RULE: {r.get('rule_text')}")
        except Exception:
            pass

        # Few-shots: include last 2 only to keep prompts small
        few_path = self.paths[f"few_{agent}"]
        last_two = []
        try:
            lines = open(few_path, "r", encoding="utf-8").read().strip().splitlines()
            for line in lines[-2:]:
                if line.strip():
                    last_two.append(json.loads(line))
        except Exception:
            pass
        if last_two:
            overlay.append("\nFEW-SHOT EXAMPLES:")
            for ex in last_two:
                overlay.append(json.dumps(ex, ensure_ascii=False))

        return "MEMORY OVERLAY\n" + "\n".join(overlay) if overlay else ""
