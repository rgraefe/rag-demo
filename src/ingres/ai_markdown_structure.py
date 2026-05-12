import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompts.structure_profile import (
    STRUCTURE_PROFILE_PROMPT,
    STRUCTURE_PROFILE_REVIEW_PROMPT,
)
from models.model_factory import ModelFactory, Modeltypes

logger = logging.getLogger(__name__)


STATIC_GENERIC_PROFILE: Dict[str, Any] = {
    "document_type": "generic converted markdown",
    "document_class": "generic",
    "confidence": 1.0,
    "rules": [
        {
            "type": "regex_replace",
            "pattern": r"^>\s?",
            "replacement": "",
            "description": "Remove fake PDF blockquote marker at line start",
            "source": "static_generic",
            "priority": 10,
        },
        {
            "type": "regex_replace",
            "pattern": r"^(\s*)\(?(\d+)\)\s+",
            "replacement": r"\1\2. ",
            "description": "Normalize numbered list styles like 1) or (1) to markdown style 1.",
            "source": "static_generic",
            "priority": 20,
        },
    ],
}


STATIC_EU_REGULATION_PROFILE: Dict[str, Any] = {
    "document_type": "EU regulation baseline",
    "document_class": "eu_regulation",
    "confidence": 1.0,
    "rules": [
        {
            "type": "heading",
            "pattern": r"^CHAPTER\s+[IVXLCM\d]+\b",
            "level": 1,
            "description": "EU regulation chapter heading",
            "source": "static_eu_regulation",
            "priority": 110,
        },
        {
            "type": "heading",
            "pattern": r"^\s*Article\s+\d+\s*",
            "level": 2,
            "description": "Detect standalone EU regulation article headings",
            "source": "static_eu_regulation",
            "priority": 130,
        },
    ],
}


STATIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "generic": STATIC_GENERIC_PROFILE,
    "eu_regulation": STATIC_EU_REGULATION_PROFILE,
}


class MarkdownStructureProfileGenerator:
    """
    Uses an LLM to infer document class and additional structure rules.

    Flow:
      1. Ask LLM for candidate profile.
      2. Merge static + AI candidate rules.
      3. Ask LLM to review the merged profile and remove semantic duplicates.
      4. Validate final profile.
    """

    def __init__(
        self,
        modeltype: Modeltypes = Modeltypes.OPENAI,
        max_chars: int = 24000,
    ) -> None:
        self.llm = ModelFactory.getLlmModel(modeltype)
        self.max_chars = max_chars

    def generate_profile(
        self,
        md_text: str,
        rules: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        sample = self._select_representative_sample(md_text)

        existing_rules = rules or STATIC_GENERIC_PROFILE["rules"]
        existing_rules_json = json.dumps(existing_rules, indent=2)

        prompt = STRUCTURE_PROFILE_PROMPT.format(
            document_text=sample,
            existing_rules_json=existing_rules_json,
        )

        response = self.llm.complete(prompt)
        raw = getattr(response, "text", str(response)).strip()

        ai_profile = self._parse_json(raw)
        self._validate_ai_profile(ai_profile)

        merged = StructureProfileMerger.merge(ai_profile)
        self._validate_final_profile(merged)

        reviewed = self._review_profile_with_llm(merged)
        self._validate_final_profile(reviewed)

        return reviewed

    def _review_profile_with_llm(
        self,
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        profile_json = json.dumps(profile, indent=2)

        prompt = STRUCTURE_PROFILE_REVIEW_PROMPT.format(
            profile_json=profile_json,
        )

        response = self.llm.complete(prompt)
        raw = getattr(response, "text", str(response)).strip()

        reviewed = self._parse_json(raw)

        # Keep defensive fallback metadata if the review prompt omitted it.
        reviewed.setdefault("document_type", profile.get("document_type", "unknown"))
        reviewed.setdefault("document_class", profile.get("document_class", "generic"))
        reviewed.setdefault("confidence", profile.get("confidence", 0.0))
        reviewed.setdefault("ai_profile", profile.get("ai_profile", {}))

        return reviewed

    def save_profile(self, profile: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def _select_representative_sample(self, md_text: str) -> str:
        lines = md_text.splitlines()

        interesting_patterns = [
            r"^\s*>?\s*Article\s+\d+\b",
            r"^\s*>?\s*CHAPTER\s+[IVXLCM\d]+\b",
            r"^\s*>?\s*Section\s+\d+\b",
            r"^\s*\(?\d+\)\s+",
            r"^\s*\d+\.\s+",
            r"^\s*>",
            r"^\s*#{1,6}\s+",
            r"^Whereas:$",
            r"^[A-Z][A-Z\s,\-()0-9]{10,}$",
        ]

        interesting: List[str] = []
        for line in lines:
            if any(re.search(p, line) for p in interesting_patterns):
                interesting.append(line)

        n = len(lines)
        head = "\n".join(lines[:250])
        middle = "\n".join(lines[max(0, n // 2 - 125): n // 2 + 125])
        tail = "\n".join(lines[-250:])
        signal = "\n".join(interesting[:500])

        sample = (
            "=== BEGINNING ===\n"
            + head
            + "\n\n=== STRUCTURAL SIGNALS ===\n"
            + signal
            + "\n\n=== MIDDLE ===\n"
            + middle
            + "\n\n=== END ===\n"
            + tail
        )

        return sample[: self.max_chars]

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON returned by LLM:\n%s", raw)
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    def _validate_ai_profile(self, profile: Dict[str, Any]) -> None:
        if not isinstance(profile, dict):
            raise ValueError("AI profile must be a JSON object.")

        if "document_type" not in profile:
            raise ValueError("AI profile missing 'document_type'.")

        if "document_class" not in profile:
            raise ValueError("AI profile missing 'document_class'.")

        if "discovered_rules" not in profile:
            raise ValueError("AI profile missing 'discovered_rules'.")

        if not isinstance(profile["discovered_rules"], list):
            raise ValueError("'discovered_rules' must be a list.")

        for i, rule in enumerate(profile["discovered_rules"]):
            self._validate_rule(rule, i)

    def _validate_final_profile(self, profile: Dict[str, Any]) -> None:
        if not isinstance(profile, dict):
            raise ValueError("Final profile must be a JSON object.")

        if "rules" not in profile or not isinstance(profile["rules"], list):
            raise ValueError("Final profile missing 'rules' list.")

        for i, rule in enumerate(profile["rules"]):
            self._validate_rule(rule, i)

    def _validate_rule(self, rule: Dict[str, Any], index: int) -> None:
        allowed_types = {"heading", "regex_replace"}

        if not isinstance(rule, dict):
            raise ValueError(f"Rule {index} must be an object.")

        rule_type = rule.get("type")
        if rule_type not in allowed_types:
            raise ValueError(f"Rule {index} has invalid type: {rule_type}")

        pattern = rule.get("pattern")
        if not pattern:
            raise ValueError(f"Rule {index} missing pattern.")

        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Rule {index} has invalid regex: {exc}") from exc

        if rule_type == "heading":
            level = rule.get("level")
            if not isinstance(level, int) or level < 1 or level > 6:
                raise ValueError(f"Heading rule {index} needs level 1..6.")

        if rule_type == "regex_replace":
            if "replacement" not in rule:
                raise ValueError(f"regex_replace rule {index} missing replacement.")

            replacement = rule.get("replacement", "")
            if "$" in replacement:
                raise ValueError(
                    f"Rule {index} uses $1-style replacement. "
                    "Use Python replacement syntax like \\\\1 instead."
                )

            if re.search(r"\\d\+|\\d", pattern) and replacement == "":
                raise ValueError(
                    f"Rule {index} may delete numbering: {rule}"
                )


class StructureProfileMerger:
    """
    Mechanically combines static rules with AI-discovered rules.

    Semantic duplicate/conflict cleanup is handled by the second LLM review pass.
    """

    @staticmethod
    def merge(ai_profile: Dict[str, Any]) -> Dict[str, Any]:
        document_class = ai_profile.get("document_class", "generic")

        merged: Dict[str, Any] = {
            "document_type": ai_profile.get("document_type", "unknown"),
            "document_class": document_class,
            "confidence": ai_profile.get("confidence", 0.0),
            "rules": [],
            "ai_profile": ai_profile,
        }

        StructureProfileMerger._append_rules(
            merged["rules"],
            STATIC_GENERIC_PROFILE["rules"],
        )

        if document_class in STATIC_PROFILES and document_class != "generic":
            StructureProfileMerger._append_rules(
                merged["rules"],
                STATIC_PROFILES[document_class]["rules"],
            )

        ai_rules = [
            StructureProfileMerger._sanitize_ai_rule(rule)
            for rule in ai_profile.get("discovered_rules", [])
        ]

        StructureProfileMerger._append_rules(
            merged["rules"],
            ai_rules,
        )

        merged["rules"] = sorted(
            merged["rules"],
            key=lambda r: int(r.get("priority", 1000)),
        )

        return merged

    @staticmethod
    def _append_rules(
        target: List[Dict[str, Any]],
        rules: List[Dict[str, Any]],
    ) -> None:
        for rule in rules:
            target.append(deepcopy(rule))

    @staticmethod
    def _sanitize_ai_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
        clean = deepcopy(rule)
        clean["source"] = "ai"
        clean.setdefault("priority", 1000)

        if clean.get("type") == "regex_replace":
            replacement = clean.get("replacement", "")

            # Convert common JS-style LLM replacement syntax to Python syntax.
            if "$" in replacement:
                clean["replacement"] = re.sub(
                    r"\$(\d+)",
                    r"\\\1",
                    replacement,
                )

            pattern = clean.get("pattern", "")
            replacement = clean.get("replacement", "")

            if re.search(r"\\d\+|\\d", pattern) and replacement == "":
                raise ValueError(
                    f"Unsafe AI rule would delete numbering: {clean}"
                )

        return clean


class MarkdownStructureRuleApplier:
    """
    Applies a validated structure profile deterministically to complete md_text.
    """

    def __init__(self, profile: Dict[str, Any]) -> None:
        self.profile = profile
        self.rules = profile.get("rules", [])

    def apply(
        self,
        md_text: str,
        output_path: Optional[Path] = None,
        write_if_larger_than: int = 1_000,
    ) -> str:
        lines = md_text.splitlines()
        lines = self._apply_line_rules(lines)

        result = "\n".join(lines)

        if output_path is not None and len(result) >= write_if_larger_than:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result, encoding="utf-8")
            logger.info("Wrote normalized markdown to %s", output_path)

        return result

    def _apply_line_rules(self, lines: List[str]) -> List[str]:
        out: List[str] = []

        regex_rules = [
            r for r in self.rules if r.get("type") == "regex_replace"
        ]
        heading_rules = [
            r for r in self.rules if r.get("type") == "heading"
        ]

        for line in lines:
            line = self._apply_regex_replace(line, regex_rules)
            line = self._apply_heading(line, heading_rules)
            out.append(line)

        out = self._normalize_blank_lines(out)
        out = self._insert_blank_lines_before_headings(out)

        return out

    def _apply_regex_replace(
        self,
        line: str,
        rules: List[Dict[str, Any]],
    ) -> str:
        for rule in rules:
            line = re.sub(
                rule["pattern"],
                rule.get("replacement", ""),
                line,
            )
        return line

    def _apply_heading(
        self,
        line: str,
        heading_rules: List[Dict[str, Any]],
    ) -> str:
        stripped = line.strip()

        if not stripped:
            return line

        if re.match(r"^#{1,6}\s+", stripped):
            return line

        for rule in heading_rules:
            if re.search(rule["pattern"], stripped):
                level = int(rule["level"])
                return f"{'#' * level} {stripped}"

        return line

    def _normalize_blank_lines(self, lines: List[str]) -> List[str]:
        result: List[str] = []
        previous_blank = False

        for line in lines:
            current_blank = not line.strip()

            if current_blank and previous_blank:
                continue

            result.append(line)
            previous_blank = current_blank

        return result

    def _insert_blank_lines_before_headings(self, lines: List[str]) -> List[str]:
        result: List[str] = []

        for line in lines:
            is_heading = bool(re.match(r"^#{1,6}\s+", line.strip()))

            if is_heading and result and result[-1].strip():
                result.append("")

            result.append(line)

        return result