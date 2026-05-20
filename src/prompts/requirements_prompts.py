from llama_index.core import PromptTemplate

REQUIREMENT_EXTRACTION_PROMPT = PromptTemplate(
"""
You are extracting compliance requirements from a legal or regulatory text.

Extract only concrete requirements, obligations, prohibitions, duties,
conditions, deadlines, responsibilities, or documentation requirements.

Do not invent requirements. If the text only contains definitions or context,
return an empty requirements list.

Return JSON only in this schema:

{
  "requirements": [
    {
      "id": "REQ-...",
      "topic": "...",
      "requirement": "...",
      "must_cover": ["...", "..."],
      "conditions": ["...", "..."],
      "deadline": "... or null",
      "responsible_party": "... or null",
      "severity": "low|medium|high|critical",
      "source_section": "...",
      "source_quote": "..."
    }
  ]
}

Article ID:
{article_id}

Header:
{header_path}

Legal text:
{article_text}
"""
)