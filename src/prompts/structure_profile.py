from llama_index.core import PromptTemplate

STRUCTURE_PROFILE_REVIEW_PROMPT = PromptTemplate(
    """
You are reviewing a JSON document-structure parsing profile.

Remove duplicates in rules.
By duplicate I mean not only identical strings but having the same purpose. 
Keep those that are not part of ai_profile. Only return the fixed json.

The returned JSON must have this structure:

{
  "document_type": "...",
  "document_class": "...",
  "confidence": 0.0,
  "rules": [],
  "ai_profile": {}
}

Here is the profile to review:

{profile_json}

"""
)

STRUCTURE_PROFILE_PROMPT = PromptTemplate(
    """
You are an expert in document structure analysis and OCR/PDF cleanup.

Your task is to analyze a markdown document converted from PDF or Word and infer structural parsing rules.

IMPORTANT:
You do NOT rewrite the document.
You do NOT remove meaningful legal text.
You do NOT remove numbering.
You only return a JSON profile that software can apply deterministically.

Return ONLY valid JSON.
No markdown fences.
No explanation outside JSON.

Allowed JSON structure:

{
  "document_type": "short descriptive name",
  "document_class": "generic | eu_regulation | contract | technical_manual | unknown",
  "confidence": 0.0,
  "discovered_rules": [
    {
      "type": "heading",
      "pattern": "Python-compatible regex",
      "level": 1,
      "description": "what this detects"
    },
    {
      "type": "regex_replace",
      "pattern": "Python-compatible regex",
      "replacement": "replacement string",
      "description": "what this normalizes"
    }
  ]
}

Allowed rule types:
- heading
- regex_replace

Do NOT create rules of these types:
- remove_line
- remove_prefix
- merge_with_next
- split_before

Those are handled by deterministic code.

Heading guidance:
- Existing markdown headings beginning with # should not be duplicated.
- Numbered recitals like "1)" are NOT headings.
- Introductory legal clauses are usually NOT headings.
- For EU regulations:
  - "CHAPTER I", "CHAPTER II" are usually level 1 headings.
  - "Section 1", "Section 2" are usually level 2 headings.
  - "Article 5", "Article 12" are usually level 3 headings.
  - "Whereas:" may be level 1 or level 2 depending on document structure.
  - Recitals like "1)", "2)", "10)" are numbered list items, not headings.

Normalization guidance:
- Fake PDF blockquote markers like "> Article 5" may be normalized by removing leading ">".
- Numbered items like "1)" should become "1.", never be deleted.
- Do not remove legal clauses such as "Having regard to..." or "Whereas:".
- Do not generate overly specific title-only rules unless they reveal real document structure.
- Prefer reusable structural patterns over exact long text.

Here is the document excerpt:

{document_text}

Existing rules already handled by the system:

{existing_rules_json}

Do NOT return rules that duplicate or approximate these existing rules.
Do NOT return alternative regexes for the same concept.
Only return rules for structural patterns that are not already covered.
If no additional rules are needed, return "discovered_rules": [].
"""
)