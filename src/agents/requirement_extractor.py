import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from prompts.requirements_prompts import REQUIREMENT_EXTRACTION_PROMPT


CACHE_DIR = Path("requirement_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _extract_json_object(text: str) -> dict:
    """
    Robustly extract JSON from an LLM response.
    Handles accidental ```json fences.
    """
    cleaned = text.strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            raise

        return json.loads(match.group(0))


def load_json(path: Path, default: Any) -> Any:
    """
    Load JSON safely.
    """
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """
    Atomic JSON save.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def extract_requirements_with_llm(
    llm: Any,
    article_id: str,
    article_text: str,
    header_path: str | None = None,
    source_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Extract structured requirements from one complete legal article/section.

    Works with common LlamaIndex LLM interfaces that expose .complete().
    """

    prompt = REQUIREMENT_EXTRACTION_PROMPT.format(
        article_id=article_id,
        header_path=header_path or "",
        article_text=article_text,
    )

    response = llm.complete(prompt)
    raw_text = str(response)

    data = _extract_json_object(raw_text)

    requirements = data.get("requirements", [])

    normalized: List[Dict[str, Any]] = []

    for i, req in enumerate(requirements, start=1):
        requirement_text = str(req.get("requirement", "")).strip()

        if not requirement_text:
            continue

        req_id = req.get("id") or f"{article_id}-REQ-{i:03d}"

        normalized.append(
            {
                "id": req_id,
                "source_name": source_name,
                "source_article_id": article_id,
                "source_section": req.get("source_section") or header_path,
                "topic": req.get("topic", "").strip(),
                "requirement": requirement_text,
                "must_cover": req.get("must_cover") or [],
                "conditions": req.get("conditions") or [],
                "deadline": req.get("deadline"),
                "responsible_party": req.get("responsible_party"),
                "severity": req.get("severity", "medium"),
                "source_quote": req.get("source_quote", ""),
            }
        )

    return normalized


def assemble_article_text(article, all_nodes) -> tuple[str, list]:
    """
    Assemble full hierarchical article text from:
    - parent article node
    - sibling paragraph children
    """

    article_id = article.node_id

    children = [
        n for n in all_nodes
        if n.metadata.get("parent_article") == article_id
        and n.metadata.get("node_level") == "paragraph"
    ]

    children = sorted(
        children,
        key=lambda n: n.metadata.get("article_child_index", 10**9),
    )

    full_article_text = "\n\n".join(
        [article.get_content()] +
        [c.get_content() for c in children]
    )

    return full_article_text, children


def extract_requirements_from_legal_corpus(
    docstore,
    llm: Any,
    deduplicate_fn,
    source_name: str = "GDPR",
    cache_name: str = "legal_requirements",
    force_reprocess: bool = False,
):
    """
    Exhaustively process all parent/article nodes from a legal corpus.

    Features:
    - per-article caching
    - resumable extraction
    - coverage logging
    - final JSON persistence
    """

    article_cache_file = (
        CACHE_DIR /
        f"{cache_name}_article_cache.json"
    )

    final_output_file = (
        CACHE_DIR /
        f"{cache_name}_final.json"
    )

    coverage_log_file = (
        CACHE_DIR /
        f"{cache_name}_coverage_log.json"
    )

    all_nodes = list(docstore.docs.values())

    article_nodes = [
        n for n in all_nodes
        if n.metadata.get("node_level") == "article"
    ]

    article_nodes = sorted(
        article_nodes,
        key=lambda n: (
            n.metadata.get("header_path", ""),
            n.node_id,
        ),
    )

    article_cache = load_json(article_cache_file, {})
    coverage_log = load_json(coverage_log_file, {})

    print("=" * 100)
    print(f"Found {len(article_nodes)} article nodes.")
    print(f"Cached nodes: {len(article_cache)}")
    print("=" * 100)

    all_requirements: List[Dict[str, Any]] = []

    for index, article in enumerate(article_nodes, start=1):
        article_id = article.node_id
        header_path = article.metadata.get("header_path")

        print("=" * 100)
        print(f"[{index}/{len(article_nodes)}]")
        print(f"Article: {header_path or article_id}")

        if article_id in article_cache and not force_reprocess:
            print("Using cached extraction.")

            cached_requirements = (
                article_cache[article_id]
                .get("requirements", [])
            )

            all_requirements.extend(cached_requirements)

            continue

        full_article_text, children = assemble_article_text(
            article,
            all_nodes,
        )

        try:
            extracted = extract_requirements_with_llm(
                llm=llm,
                article_id=article_id,
                article_text=full_article_text,
                header_path=header_path,
                source_name=source_name,
            )

            article_cache[article_id] = {
                "article_id": article_id,
                "header_path": header_path,
                "source_name": source_name,
                "child_count": len(children),
                "text_length": len(full_article_text),
                "requirements": extracted,
                "processed_at":
                    datetime.utcnow().isoformat() + "Z",
                "status": "success",
            }

            coverage_log[article_id] = {
                "article_id": article_id,
                "header_path": header_path,
                "source_name": source_name,
                "child_count": len(children),
                "text_length": len(full_article_text),
                "requirements_extracted": len(extracted),
                "status": "success",
                "processed_at":
                    datetime.utcnow().isoformat() + "Z",
            }

            all_requirements.extend(extracted)

            save_json(article_cache_file, article_cache)
            save_json(coverage_log_file, coverage_log)

            print(f"Extracted requirements: {len(extracted)}")

        except Exception as e:
            print(f"ERROR processing article {article_id}")
            print(str(e))

            coverage_log[article_id] = {
                "article_id": article_id,
                "header_path": header_path,
                "source_name": source_name,
                "child_count":
                    len(children)
                    if "children" in locals()
                    else None,
                "text_length":
                    len(full_article_text)
                    if "full_article_text" in locals()
                    else None,
                "requirements_extracted": 0,
                "status": "error",
                "error": str(e),
                "processed_at":
                    datetime.utcnow().isoformat() + "Z",
            }

            save_json(coverage_log_file, coverage_log)

            continue

    print("=" * 100)
    print("Deduplicating requirements...")
    print("=" * 100)

    deduped_requirements = deduplicate_fn(
        all_requirements
    )

    final_payload = {
        "source_name": source_name,
        "created_at":
            datetime.utcnow().isoformat() + "Z",
        "article_nodes_total":
            len(article_nodes),
        "article_nodes_cached_or_processed":
            len(article_cache),
        "requirements_raw_count":
            len(all_requirements),
        "requirements_deduped_count":
            len(deduped_requirements),
        "requirements":
            deduped_requirements,
    }

    save_json(
        final_output_file,
        final_payload,
    )

    print(f"Raw requirements: {len(all_requirements)}")
    print(f"Deduped requirements: {len(deduped_requirements)}")
    print(f"Saved final JSON to: {final_output_file}")

    return deduped_requirements