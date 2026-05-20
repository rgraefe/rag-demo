import uuid
import hashlib
from IPython.display import Markdown, display
from llama_index.core.schema import BaseNode, Document, TextNode
from difflib import SequenceMatcher
from typing import Any, Dict, List
import re

def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))
        
def document_to_node(doc: Document):
    node = TextNode(id_= doc.id_,
                    embedding=doc.embedding,
                    extra_info=doc.extra_info,
                    excluded_llm_metadata_keys=doc.excluded_embed_metadata_keys,
                    excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                    relationships=doc.relationships,
                    text = doc.text,
                    text_template=doc.text_template,
                    metadata_template=doc.metadata_template,
                    metadata_separator=doc.metadata_separator)
    return node

def node_to_document(node: BaseNode) -> Document:
    if isinstance(node, Document):
        return node

    return Document(
        text=node.get_content(),
        metadata=dict(node.metadata or {}),
        id_=node.node_id,
    )
    


def _norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


def deduplicate_requirements(
    requirements: List[Dict[str, Any]],
    similarity_threshold: float = 0.88,
) -> List[Dict[str, Any]]:
    """
    Simple deterministic deduplication.

    Keeps the first requirement and merges source references from later
    near-duplicates.
    """
    deduped: List[Dict[str, Any]] = []

    for req in requirements:
        req_text = req.get("requirement", "")

        if not req_text:
            continue

        matched_existing = None

        for existing in deduped:
            existing_text = existing.get("requirement", "")

            if _similarity(req_text, existing_text) >= similarity_threshold:
                matched_existing = existing
                break

        if matched_existing is None:
            req = dict(req)
            req["source_refs"] = [
                {
                    "source_name": req.get("source_name"),
                    "source_article_id": req.get("source_article_id"),
                    "source_section": req.get("source_section"),
                    "source_quote": req.get("source_quote"),
                }
            ]
            deduped.append(req)
        else:
            matched_existing.setdefault("source_refs", []).append(
                {
                    "source_name": req.get("source_name"),
                    "source_article_id": req.get("source_article_id"),
                    "source_section": req.get("source_section"),
                    "source_quote": req.get("source_quote"),
                }
            )

            matched_existing["must_cover"] = sorted(
                set(matched_existing.get("must_cover", []))
                | set(req.get("must_cover", []))
            )

            matched_existing["conditions"] = sorted(
                set(matched_existing.get("conditions", []))
                | set(req.get("conditions", []))
            )

    return deduped